import random
import torch
import torch.nn.functional as F
import numpy as np
from wav_minigrid.dataset import MiniGridDynamicsDataset, NormalizedDataset
from wav_minigrid.models import SparseIDM
from torch.utils.data import DataLoader, Subset


def load_video_pairs(path, device, role):
    """Read action-free pairs; training roles are 'groups' and 'policy'."""
    if role not in ("groups", "policy"):
        raise ValueError(f"Unknown video data role: {role}")
    fields = {"states", "carried", "next_states", "next_carried"}
    with np.load(path) as data:
        expected = fields | {"is_group"}
        if set(data.files) != expected:
            raise ValueError(f"Expected state-only fields and is_group, not action labels: {path}")
        arrays = {k: data[k] for k in fields}
        flag = data["is_group"]
        if flag.dtype != np.bool_ or flag.shape != (len(arrays["states"]),):
            raise ValueError("is_group must be one boolean per transition")
        selected = flag if role == "groups" else ~flag
        arrays = {k: v[selected] for k, v in arrays.items()}
    n = len(arrays["states"])
    expected = {"states": (n, 6, 6, 3), "carried": (n, 2),
                "next_states": (n, 6, 6, 3), "next_carried": (n, 2)}
    if not n or any(arrays[k].shape != shape for k, shape in expected.items()):
        raise ValueError(f"Invalid video transition shapes: {path}")
    if role != "groups":
        return {k: torch.as_tensor(v, device=device).float() for k, v in arrays.items()}
    current = np.concatenate([arrays["states"].reshape(n, 108), arrays["carried"]], 1).astype(np.int16)
    target = np.concatenate([arrays["next_states"].reshape(n, 108), arrays["next_carried"]], 1).astype(np.int16)
    groups = {}
    for state, successor in zip(current, target):
        groups.setdefault(state.tobytes(), []).append(successor.tolist())
    if any(len(v) != 7 for v in groups.values()):
        raise ValueError("Expected seven observations/input, retaining duplicate outcomes.")
    keys = sorted(groups)
    current = np.stack([np.frombuffer(k, dtype=np.int16) for k in keys])
    target = np.asarray([sorted(groups[k]) for k in keys], dtype=np.int16)
    return tuple(torch.as_tensor(v, device=device).float()
                 for v in (current[:, :108].reshape(-1, 6, 6, 3), current[:, 108:], target))


def video_observations(states, carried):
    return dict(frame=states, carried_col=carried[:, :1], carried_obj=carried[:, 1:])


def video_all_codes(model, inputs):
    """Decode every anonymous code through the unchanged original WorldModel."""
    features = model._extract_spatial_features(inputs)
    n, k = len(features), model.num_actions
    # Pair training explicitly recomputes a prior attached to the encoder.
    prior = model.prior_net(features.detach())
    features = features.repeat_interleave(k, 0)
    codes = model.vq_layer._embedding(torch.arange(k, device=features.device).repeat(n))
    gamma, beta = model.film_gen(codes).chunk(2, -1)
    delta = (1 + gamma[:, :, None, None]) * model.dyn_conv(features) + beta[:, :, None, None]
    next_features = features + delta
    decoded = model.decoder_shared(next_features)
    carried = model.carried_head(next_features)
    return dict(logits_obj=model.head_obj_cls(decoded), logits_col=model.head_col_cls(decoded),
                logits_state=model.head_state_cls(decoded), carried_col=carried[:, :1],
                carried_obj=carried[:, 1:], prior_logits=prior)


def video_pair_cost(out, current, target, carried_weight=10., changed_weight=1.):
    """Reconstruction costs [input, latent, observed successor]; no action labels."""
    n, t = target.shape[:2]
    k = len(out["logits_obj"]) // n
    grid = target[:, :, :108].reshape(n, t, 36, 3).long()
    weights = 1 + (changed_weight - 1) * (current.reshape(n, 1, 36, 3) != grid).any(-1).float()
    cost = 0
    for channel, key in enumerate(("logits_obj", "logits_col", "logits_state")):
        logp = F.log_softmax(out[key], 1).flatten(2).permute(0, 2, 1).reshape(n, k, 36, -1)
        labels = grid[:, :, :, channel][:, None, :, :, None].expand(n, k, t, 36, 1)
        ce = -logp[:, :, None].expand(n, k, t, 36, logp.shape[-1]).gather(-1, labels).squeeze(-1)
        cost = cost + (ce * weights[:, None]).mean(-1)
    predicted_carried = torch.cat([out["carried_col"], out["carried_obj"]], 1).reshape(n, k, 2)
    return cost + carried_weight * (predicted_carried[:, :, None] - target[:, None, :, 108:]).square().sum(-1)


def video_matching_loss(out, current, target, config):
    from scipy.optimize import linear_sum_assignment

    cost = video_pair_cost(out, current, target, config["FORWARD_CARRIED_LOSS_WEIGHT"],
                           config["CHANGED_PIXEL_WEIGHT"])
    choices = np.stack([linear_sum_assignment(v)[1] for v in cost.detach().cpu().numpy()])
    assignment = torch.as_tensor(choices, device=cost.device)
    agent_mass = out["logits_obj"].softmax(1)[:, 10].sum((1, 2))
    return (cost.gather(2, assignment[:, :, None]).mean()
            + config["CONSISTENCY_WEIGHT"] * (agent_mass - 1).square().mean())


def video_pair_losses(model, batch, config):
    """Dynamic anonymous targets: prior mass loss and weighted pair reconstruction."""
    inputs = video_observations(batch["states"], batch["carried"])
    target = torch.cat([batch["next_states"].flatten(1), batch["next_carried"]], 1)[:, None]
    out = video_all_codes(model, inputs)
    cost = video_pair_cost(out, batch["states"], target).squeeze(-1)
    with torch.no_grad():
        grid = torch.stack([out[k].argmax(1) for k in
                            ("logits_obj", "logits_col", "logits_state")], -1).flatten(1)
        carried = torch.cat([out["carried_col"], out["carried_obj"]], 1).round()
        decoded = torch.cat([grid, carried], 1).reshape(len(target), model.num_actions, 110)
        mask = (decoded == target).all(-1)
        exact = mask.any(-1)
        mask[~exact] = F.one_hot(cost[~exact].argmin(1), model.num_actions).bool()
        weights = torch.where(exact, 1., config["UNMATCHED_WEIGHT"])
    # Probability gradients reach the shared encoder, as in the verified experiment.
    logits = model.prior_net(model._extract_spatial_features(inputs)) / config["PRIOR_TEMPERATURE"]
    logmass = logits.log_softmax(-1).masked_fill(~mask, -1e9).logsumexp(-1)
    prior = -(weights * logmass).sum() / weights.sum()
    reconstruction = (weights * cost.masked_fill(~mask, 1e9).min(-1).values).sum() / weights.sum()
    return prior, reconstruction, exact.float().mean()


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_batch_inputs(batch, device):
    """Prepare batch data for model input by permuting dimensions.

    Args:
        batch: Dictionary containing 'frame', 'carried_col', 'carried_obj', and optionally 'action'
        device: Target device for tensors

    Returns:
        inputs: Dictionary with permuted inputs
        actions: Action tensor if present, else None
    """
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'action'}
    inputs['frame'] = inputs['frame'].permute(1, 0, 2, 3, 4)  # [T, B, H, W, C]
    inputs['carried_col'] = inputs['carried_col'].permute(1, 0, 2)  # [T, B, 1]
    inputs['carried_obj'] = inputs['carried_obj'].permute(1, 0, 2)  # [T, B, 1]
    actions = batch.get('action')
    if actions is not None:
        actions = actions.to(device)
    return inputs, actions


def get_dataloaders_with_validation(data_path, batch_size, train_ratio=0.8, seed=None, normalize=True):
    """Create train and validation dataloaders from a single dataset.

    Args:
        data_path: Path to the dataset file
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data to use for training (rest for validation)
        seed: Random seed for splitting
        normalize: Whether to normalize the dataset

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    print(f"Loading dataset from: {data_path}")
    full_dataset = MiniGridDynamicsDataset(data_path)
    dataset_size = len(full_dataset)
    print(f"Dataset size: {dataset_size}")

    # Normalize dataset if requested
    if normalize:
        dataset = NormalizedDataset(full_dataset)
    else:
        dataset = full_dataset

    # Split into train and validation
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        indices = torch.randperm(dataset_size).tolist()
        torch.set_rng_state(rng_state)
    else:
        indices = torch.randperm(dataset_size).tolist()

    train_size = int(dataset_size * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    return train_loader, val_loader


def evaluate_idm(model, dataloader, device, num_actions=7, verbose=False):
    """Evaluate IDM model accuracy. Returns action accuracies dict and average accuracy.

    Args:
        model: The IDM model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        num_actions: Number of possible actions
        verbose: Whether to print per-action accuracies

    Returns:
        action_accuracies: Dictionary mapping action index to accuracy
        avg_acc: Average accuracy across all actions
    """
    model.eval()
    correct = 0
    total = 0
    action_correct = torch.zeros(num_actions, dtype=torch.long)
    action_total = torch.zeros(num_actions, dtype=torch.long)

    with torch.no_grad():
        for batch in dataloader:
            inputs, actions = prepare_batch_inputs(batch, device)

            outputs = model(inputs)
            if isinstance(model, SparseIDM):
                logits, _, _ = outputs
            else:
                logits = outputs
            pred = torch.argmax(logits, dim=1)

            correct_batch = (pred == actions)
            correct += correct_batch.sum().item()
            total += actions.size(0)

            for a in range(num_actions):
                mask = (actions == a)
                action_total[a] += mask.sum().item()
                if mask.any():
                    action_correct[a] += (correct_batch & mask).sum().item()

    acc = correct / total if total > 0 else 0.0
    action_accuracies = {}

    if verbose:
        print("\nPer-Action Accuracy:")
        for a in range(num_actions):
            count = action_total[a].item()
            if count > 0:
                p = action_correct[a].item() / count
                action_accuracies[a] = p
                print(f"  Action {a}: {p*100:6.2f}% ({action_correct[a]}/{count})")
            else:
                action_accuracies[a] = 0.0
                print(f"  Action {a}: N/A (0 samples)")
    else:
        for a in range(num_actions):
            count = action_total[a].item()
            if count > 0:
                action_accuracies[a] = action_correct[a].item() / count
            else:
                action_accuracies[a] = 0.0

    return action_accuracies, acc


def load_model_checkpoint(model, checkpoint_path, device, strict=False):
    """Load model checkpoint with error handling.

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        strict: Whether to strictly enforce that the keys match

    Returns:
        model: Model with loaded weights
    """
    print(f"Loading model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=strict)

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    missing_keys = checkpoint_keys - model_keys
    unexpected_keys = model_keys - checkpoint_keys
    if missing_keys:
        print(f"  [Warning] Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  [Warning] Unexpected keys: {len(unexpected_keys)}")

    return model

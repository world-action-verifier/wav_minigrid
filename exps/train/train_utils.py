import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from asim_minigrid.dataset import MiniGridDynamicsDataset, NormalizedDataset
from torch.utils.data import DataLoader, Subset, Dataset

def compute_loss_vp(pred_out, inputs, criterion_ce_none, criterion_mse,
                         aux_action_weight=40.0,
                         consistency_weight=10.0,
                         carried_weight=10.0,
                         vq_loss_weight=10.0,
                         prior_weight=50.0):
    """Compute loss for WorldModel with physics alignment."""
    if not hasattr(criterion_ce_none, "reduction") or criterion_ce_none.reduction != "none":
        criterion_ce_none = nn.CrossEntropyLoss(reduction="none")
    
    device = pred_out['logits_obj'].device
    curr_frame = inputs['frame'][0].long()
    target_frame = inputs['frame'][1].long()
    gt_obj, gt_col, gt_state = target_frame[..., 0], target_frame[..., 1], target_frame[..., 2]
    target_c_col = inputs['carried_col'][1].float()
    target_c_obj = inputs['carried_obj'][1].float()
    
    diff_mask = (curr_frame != target_frame).any(dim=-1).float()
    
    loss_map_obj = criterion_ce_none(pred_out['logits_obj'], gt_obj)
    loss_map_col = criterion_ce_none(pred_out['logits_col'], gt_col)
    loss_map_state = criterion_ce_none(pred_out['logits_state'], gt_state)
    loss_recon_frame = (loss_map_obj + loss_map_col + loss_map_state).mean()
    
    loss_recon_carried = carried_weight * (
        criterion_mse(pred_out['carried_col'], target_c_col) +
        criterion_mse(pred_out['carried_obj'], target_c_obj)
    )
    
    loss_vq = pred_out['vq_loss']
    prior_logits = pred_out['prior_logits']
    target_indices = pred_out['target_indices']
    criterion_ce = nn.CrossEntropyLoss()
    loss_prior = criterion_ce(prior_logits, target_indices)
    
    loss_aux_action = torch.tensor(0.0, device=device)
    action_acc = None
    if pred_out.get('gt_actions') is not None:
        gt_actions = pred_out['gt_actions']
        if gt_actions.dtype != torch.long:
            gt_actions = gt_actions.long()
        if gt_actions.dim() > 1:
            gt_actions = gt_actions.squeeze(-1)
        loss_aux_action = criterion_ce(prior_logits, gt_actions)
        action_acc = (prior_logits.argmax(dim=1) == gt_actions).float().mean()

    probs_obj = F.softmax(pred_out['logits_obj'], dim=1)
    agent_mass = probs_obj[:, 10, :, :].sum(dim=[1, 2])
    loss_consistency = F.mse_loss(agent_mass, torch.ones_like(agent_mass))

    total_loss = (loss_recon_frame + loss_recon_carried +
                  vq_loss_weight * loss_vq +
                  prior_weight * loss_prior +
                  aux_action_weight * loss_aux_action +
                  consistency_weight * loss_consistency)
    
    num_dyn = diff_mask.sum()
    loss_dyn_pixel = ((loss_map_obj + loss_map_col + loss_map_state) * diff_mask).sum() / (num_dyn + 1e-6)
    prior_acc = (prior_logits.argmax(dim=1) == target_indices).float().mean()
                 
    stats = {
        'total': total_loss.item(),
        'recon_frame': loss_recon_frame.item(),
        'recon_carried': loss_recon_carried.item(),
        'vq_loss': loss_vq.item(),
        'prior_cls': loss_prior.item(),
        'prior_acc': prior_acc.item(),
        'aux_action': loss_aux_action.item(),
        'consistency': loss_consistency.item(),
        'loss_dyn_pixel': loss_dyn_pixel.item(),
        'num_dyn_pixels': num_dyn.item()
    }
    if action_acc is not None:
        stats['action_acc'] = action_acc.item()
                 
    return total_loss, stats
    
def evaluate(
    model,
    dataloader,
    *,
    device: torch.device,
    forward_carried_loss_weight: float,
    use_random_base_model: bool,
    is_round_0: bool = False,
):
    """Evaluate model and return average loss."""
    model.eval()

    criterion_mse = nn.MSELoss(reduction="none")
    criterion_ce = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "action"}
            inputs["frame"] = inputs["frame"].permute(1, 0, 2, 3, 4)
            inputs["carried_col"] = inputs["carried_col"].permute(1, 0, 2)
            inputs["carried_obj"] = inputs["carried_obj"].permute(1, 0, 2)

            batch_actions = batch["action"].to(device)
            actions = batch_actions if (not is_round_0 or use_random_base_model) else None

            if is_round_0 and not use_random_base_model:
                pred = model(inputs, mode="inference")
            else:
                pred = model(inputs, mode="predict_with_action", gt_actions=actions)

            gt_next_frame = inputs["frame"][1].long()
            gt_obj = gt_next_frame[..., 0]
            gt_col = gt_next_frame[..., 1]
            gt_state = gt_next_frame[..., 2]

            loss_obj = criterion_ce(pred["logits_obj"], gt_obj)
            loss_col = criterion_ce(pred["logits_col"], gt_col)
            loss_state = criterion_ce(pred["logits_state"], gt_state)
            pixel_loss_per_sample = (loss_obj + loss_col + loss_state).mean(dim=(1, 2))  # [B]

            gt_carried_col = inputs["carried_col"][1].float()
            gt_carried_obj = inputs["carried_obj"][1].float()
            loss_c_col = criterion_mse(pred["carried_col"], gt_carried_col).mean(dim=1)  # [B]
            loss_c_obj = criterion_mse(pred["carried_obj"], gt_carried_obj).mean(dim=1)  # [B]
            carried_loss_per_sample = forward_carried_loss_weight * (loss_c_col + loss_c_obj)  # [B]

            per_sample_loss = pixel_loss_per_sample + carried_loss_per_sample
            total_loss += float(per_sample_loss.sum().item())
            total_samples += int(per_sample_loss.shape[0])

    avg_loss = total_loss / max(1, total_samples)
    return {"mse": float(avg_loss)}

def get_dataloaders(data_path, batch_size, split_ratio=0.5, seed=42):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    full_dataset = MiniGridDynamicsDataset(data_path)
    dataset_size = len(full_dataset)
    
    
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    indices = torch.randperm(dataset_size).tolist()
    torch.set_rng_state(rng_state)
    
    stage1_indices = indices

    train_size = int(0.8 * len(stage1_indices))
    train_indices = stage1_indices[:train_size]
    test_indices = stage1_indices[train_size:]
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Total Data: {dataset_size}")
    print(f"Stage 1 Data (Used): {len(stage1_indices)}")
    
    return train_loader, test_loader, full_dataset.states.shape[1:]


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
            
            logits = model(inputs)
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


def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Create a learning rate scheduler with warmup and cosine annealing.
    
    Args:
        optimizer: Optimizer to attach scheduler to
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
    
    Returns:
        scheduler: LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

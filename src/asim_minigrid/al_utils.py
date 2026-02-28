import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from asim_minigrid.evaluate_generation import MiniGridPhysicsOracle
from asim_minigrid.models import SparseIDM


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


def compute_loss_for_pool(
    model,
    dataset,
    pool_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
    forward_carried_loss_weight: float,
) -> np.ndarray:
    """Compute per-sample loss over pool (for loss-based selection)."""
    model.eval()

    criterion_mse = nn.MSELoss(reduction="none")
    criterion_ce = nn.CrossEntropyLoss(reduction="none")

    loader = DataLoader(Subset(dataset, list(pool_indices)), batch_size=batch_size, shuffle=False)
    losses: List[float] = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "action"}
            inputs["frame"] = inputs["frame"].permute(1, 0, 2, 3, 4)
            inputs["carried_col"] = inputs["carried_col"].permute(1, 0, 2)
            inputs["carried_obj"] = inputs["carried_obj"].permute(1, 0, 2)

            actions = batch["action"].to(device)
            pred_out = model(inputs, mode="predict_with_action", gt_actions=actions)

            gt_next_frame = inputs["frame"][1].long()
            gt_obj = gt_next_frame[..., 0]
            gt_col = gt_next_frame[..., 1]
            gt_state = gt_next_frame[..., 2]

            loss_obj = criterion_ce(pred_out["logits_obj"], gt_obj)
            loss_col = criterion_ce(pred_out["logits_col"], gt_col)
            loss_state = criterion_ce(pred_out["logits_state"], gt_state)
            pixel_loss = (loss_obj + loss_col + loss_state).mean(dim=(1, 2))  # [B]

            loss_c_col = criterion_mse(pred_out["carried_col"], inputs["carried_col"][1].float()).mean(dim=1)
            loss_c_obj = criterion_mse(pred_out["carried_obj"], inputs["carried_obj"][1].float()).mean(dim=1)
            sample_losses = pixel_loss + forward_carried_loss_weight * (loss_c_col + loss_c_obj)
            losses.extend(sample_losses.detach().cpu().numpy().tolist())

    return np.asarray(losses, dtype=np.float32)


def compute_uncertainty_for_pool(
    model,
    dataset,
    pool_indices: Sequence[int],
    batch_size: int,
    seed: int,
    n_samples: int,
    compute_uncertainty_via_mcdropout_fn,
) -> np.ndarray:
    """Compute per-sample uncertainty over pool (for uncertainty-based selection)."""
    return compute_uncertainty_via_mcdropout_fn(
        model,
        dataset,
        list(pool_indices),
        batch_size,
        seed=seed,
        n_samples=n_samples,
    )

def compute_uncertainty_via_mcdropout(model, dataset, pool_indices, batch_size, seed=None, n_samples=10):
    """
    Compute per-sample uncertainty via MC Dropout.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    device = next(model.parameters()).device
    was_training = model.training

    def freeze_bn(m):
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.eval()

    def enable_dropout(m):
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()

    def bald_score(probs_list):
        # probs_list: list([B, C, H, W])
        if not probs_list:
            return None
        probs = torch.stack(probs_list, dim=0)  # [N, B, C, H, W]
        mean_p = probs.mean(dim=0)              # [B, C, H, W]
        entropy_mean = -torch.sum(mean_p * torch.log(mean_p + 1e-8), dim=1)  # [B, H, W]
        entropy_ind = -torch.sum(probs * torch.log(probs + 1e-8), dim=2)     # [N, B, H, W]
        mean_entropy = entropy_ind.mean(dim=0)                                # [B, H, W]
        bald = entropy_mean - mean_entropy                                    # [B, H, W]
        b, h, w = bald.shape
        k = max(1, int(h * w * 0.05))
        flat = bald.view(b, -1)
        topk, _ = torch.topk(flat, k, dim=1)
        return topk.mean(dim=1)  # [B]

    def variance_score(values_list):
        # values_list: list([B, 1])
        if not values_list:
            return None
        vals = torch.stack(values_list, dim=0)  # [N, B, 1]
        return torch.var(vals, dim=0).squeeze(-1)  # [B]

    loader = DataLoader(Subset(dataset, pool_indices), batch_size=batch_size, shuffle=False)
    uncertainties = []

    try:
        model.train()
        model.apply(freeze_bn)
        model.apply(enable_dropout)

        with torch.no_grad():
            for batch in loader:
                frames = batch['frame'].to(device)
                carried_col = batch['carried_col'].to(device)
                carried_obj = batch['carried_obj'].to(device)
                actions = batch['action'].to(device)

                # Use t=0 as current state.
                if frames.dim() == 5:
                    frames0 = frames[:, 0].float()
                else:
                    frames0 = frames.float()
                if carried_col.dim() == 3:
                    carried_col0 = carried_col[:, 0].float()
                else:
                    carried_col0 = carried_col.float()
                if carried_obj.dim() == 3:
                    carried_obj0 = carried_obj[:, 0].float()
                else:
                    carried_obj0 = carried_obj.float()
                if actions.dim() > 1:
                    actions0 = actions[:, 0]
                else:
                    actions0 = actions

                inputs = {
                    'frame': frames0,
                    'carried_col': carried_col0,
                    'carried_obj': carried_obj0,
                }

                samples = {
                    'probs_obj': [],
                    'probs_col': [],
                    'probs_state': [],
                    'val_carried_col': [],
                    'val_carried_obj': [],
                }

                for _ in range(n_samples):
                    out = model(inputs, mode='predict_with_action', gt_actions=actions0)
                    samples['probs_obj'].append(F.softmax(out['logits_obj'], dim=1))
                    samples['probs_col'].append(F.softmax(out['logits_col'], dim=1))
                    samples['probs_state'].append(F.softmax(out['logits_state'], dim=1))
                    samples['val_carried_col'].append(out['carried_col'])
                    samples['val_carried_obj'].append(out['carried_obj'])

                u_obj = bald_score(samples['probs_obj'])
                u_col = bald_score(samples['probs_col'])
                u_state = bald_score(samples['probs_state'])
                u_c_col = variance_score(samples['val_carried_col'])
                u_c_obj = variance_score(samples['val_carried_obj'])

                bsz = frames0.shape[0]
                zeros = torch.zeros(bsz, device=device)
                u_obj = u_obj if u_obj is not None else zeros
                u_col = u_col if u_col is not None else zeros
                u_state = u_state if u_state is not None else zeros
                u_c_col = u_c_col if u_c_col is not None else zeros
                u_c_obj = u_c_obj if u_c_obj is not None else zeros

                final_score = (u_obj + u_col + u_state) + 10.0 * (u_c_col + u_c_obj)
                uncertainties.extend(final_score.detach().cpu().numpy())
    finally:
        model.train(was_training)
        if not was_training:
            model.eval()

    return np.array(uncertainties)

def query_strategy(
    strategy_name: str,
    model,
    dataset,
    pool_indices: Sequence[int],
    n_select: int,
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    forward_carried_loss_weight: float,
    compute_uncertainty_via_mcdropout_fn,
    uncertainty_n_samples: int = 25,
    round_idx: int = None,
    prev_losses_map: Dict[int, float] = None,
) -> Tuple[List[int], Dict, Dict]:
    """Select indices from pool based on a strategy.
    
    Returns:
        selected_indices: List of selected indices
        pseudo_actions: Dict mapping indices to pseudo actions (if any)
        current_loss_map: Dict mapping indices to current loss values (for Progress strategy)
    """
    pool_indices = list(pool_indices)
    if len(pool_indices) <= n_select:
        return pool_indices, {}, {}

    if strategy_name == "Random":
        selected = np.random.choice(pool_indices, size=n_select, replace=False).tolist()
        return selected, {}, {}

    if strategy_name in {"Hard-Oracle", "Simple-Oracle", "Uniform-Oracle"}:
        losses = compute_loss_for_pool(
            model=model,
            dataset=dataset,
            pool_indices=pool_indices,
            batch_size=batch_size,
            device=device,
            forward_carried_loss_weight=forward_carried_loss_weight,
        )
        loss_with_indices = [(float(losses[i]), pool_indices[i]) for i in range(len(pool_indices))]

        if strategy_name == "Hard-Oracle":
            loss_with_indices.sort(key=lambda x: (-x[0], x[1]))
            return [idx for _, idx in loss_with_indices[:n_select]], {}, {}

        if strategy_name == "Simple-Oracle":
            loss_with_indices.sort(key=lambda x: (x[0], x[1]))
            return [idx for _, idx in loss_with_indices[:n_select]], {}, {}

        # Uniform-Oracle: sample across bins of sorted losses.
        loss_with_indices.sort(key=lambda x: (x[0], x[1]))
        n_bins = min(10, len(loss_with_indices))
        bin_size = max(1, len(loss_with_indices) // n_bins)
        samples_per_bin = n_select // n_bins
        remainder = n_select % n_bins

        selected: List[int] = []
        for bin_idx in range(n_bins):
            start = bin_idx * bin_size
            end = start + bin_size if bin_idx < n_bins - 1 else len(loss_with_indices)
            bin_data = loss_with_indices[start:end]
            k = samples_per_bin + (1 if bin_idx < remainder else 0)
            k = min(k, len(bin_data))
            if k <= 0:
                continue
            step = max(1, len(bin_data) // k)
            for j in range(0, len(bin_data), step):
                if len(selected) >= n_select:
                    break
                selected.append(bin_data[j][1])
        if len(selected) < n_select:
            selected_set = set(selected)
            remaining = [idx for _, idx in loss_with_indices if idx not in selected_set]
            if remaining:
                fill = np.random.choice(remaining, size=min(n_select - len(selected), len(remaining)), replace=False).tolist()
                selected.extend(fill)
        return selected[:n_select], {}, {}

    if strategy_name == "Uncertainty":
        uncertainties = compute_uncertainty_for_pool(
            model=model,
            dataset=dataset,
            pool_indices=pool_indices,
            batch_size=batch_size,
            seed=seed,
            n_samples=uncertainty_n_samples,
            compute_uncertainty_via_mcdropout_fn=compute_uncertainty_via_mcdropout_fn,
        )
        u_min = float(np.min(uncertainties))
        u_max = float(np.max(uncertainties))
        if u_max - u_min > 1e-6:
            u_norm = (uncertainties - u_min) / (u_max - u_min)
        else:
            u_norm = np.zeros_like(uncertainties)

        temperature = 0.8
        weights = np.exp(u_norm / temperature)
        probs = weights / np.sum(weights)
        selected_local = np.random.choice(len(pool_indices), size=n_select, replace=False, p=probs)
        return [pool_indices[i] for i in selected_local], {}, {}
    
    if strategy_name == "Progress":
        losses = compute_loss_for_pool(
            model=model,
            dataset=dataset,
            pool_indices=pool_indices,
            batch_size=batch_size,
            device=device,
            forward_carried_loss_weight=forward_carried_loss_weight,
        )
        
        current_loss_map = {idx: float(losses[i]) for i, idx in enumerate(pool_indices)}

        if round_idx == 1 or prev_losses_map is None:
            scores = losses
        else:
            scores = []
            for i, idx in enumerate(pool_indices):
                loss_curr = float(losses[i])
                loss_prev = prev_losses_map.get(idx, loss_curr)
                progress = loss_prev - loss_curr
                scores.append(progress)
            scores = np.array(scores)

        score_with_indices = [(float(scores[i]), pool_indices[i]) for i in range(len(scores))]
        score_with_indices.sort(key=lambda x: (-x[0], x[1]))
        
        selected_indices = [idx for _, idx in score_with_indices[:n_select]]
        return selected_indices, {}, current_loss_map
    
    raise ValueError(f"Unknown strategy: {strategy_name}")

def select_and_collect_consistency_data(
    video_gen_model,
    current_world_model,
    inverse_model,
    dataset,
    pool_indices,
    env_name,  # kept for compatibility; not used
    n_select,
    batch_size=64,
    device="cuda",
    data_mode="oracle",  # "oracle" or "model"
    seed=None,
    use_random_mix=False,
    random_mix_ratio=0.3,
):
    """
    Score each candidate by disagreement between:
      - video_gen_model (action-free) predicted next state
      - current_world_model (conditioned on inverse_model inferred action) predicted next state

    Then collect transitions using either:
      - data_mode="oracle": MiniGridPhysicsOracle.step(...)
      - data_mode="model": use video_gen_model predicted next state as supervision

    Returns:
      selected_indices: list[int]
      new_collected_data: list[dict] with keys state/carried/action/next_state/next_carried
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        fixed_pool = pool_indices.copy()
        np.random.shuffle(fixed_pool)
        pool_indices = fixed_pool
    device_t = torch.device(device) if not isinstance(device, torch.device) else device

    # Local import to keep this module lightweight.
    if data_mode == "oracle":
        oracle = MiniGridPhysicsOracle()
    else:
        oracle = None

    video_gen_model.eval()
    current_world_model.eval()
    inverse_model.eval()

    loader = DataLoader(Subset(dataset, pool_indices), batch_size=batch_size, shuffle=False)

    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_mse = torch.nn.MSELoss(reduction='none')

    losses = []
    idx_order = []
    s_gen_cache = {}   
    action_cache = {}  

    with torch.no_grad():
        base = 0
        for batch in tqdm(loader, desc="Scoring Candidates"):
            bsz = batch['frame'].shape[0]
            batch_indices = [pool_indices[base + i] for i in range(bsz)]
            base += bsz

            curr_frame = batch['frame'][:, 0].to(device_t).float()
            c_col = batch['carried_col'][:, 0].to(device_t).float()
            c_obj = batch['carried_obj'][:, 0].to(device_t).float()

            inputs_flat = {
                'frame': curr_frame,
                'carried_col': c_col,
                'carried_obj': c_obj,
            }

            # 1. Video Gen Model Predicts Next State (Action-Free)
            gen_out = video_gen_model(inputs_flat, mode='inference')
            gen_obj = torch.argmax(gen_out['logits_obj'], dim=1).float()
            gen_col = torch.argmax(gen_out['logits_col'], dim=1).float()
            gen_state = torch.argmax(gen_out['logits_state'], dim=1).float()
            s_gen_frame = torch.stack([gen_obj, gen_col, gen_state], dim=-1)
            s_gen_c_col = torch.round(gen_out['carried_col'])
            s_gen_c_obj = torch.round(gen_out['carried_obj'])

            # 2. Inverse Model Predicts Action (based on current state + gen next state)
            inv_inputs = {
                'frame': torch.stack([curr_frame, s_gen_frame], dim=0),
                'carried_col': torch.stack([c_col, s_gen_c_col], dim=0),
                'carried_obj': torch.stack([c_obj, s_gen_c_obj], dim=0),
            }
            if isinstance(inverse_model, SparseIDM):
                output, _, _ = inverse_model(inv_inputs)
            else:
                output = inverse_model(inv_inputs)

            if isinstance(output, tuple):
                act_logits = output[0]
            else:
                act_logits = output
            pred_actions = torch.argmax(act_logits, dim=1) 

            # 3. World Model Predicts Next State (conditioned on Predicted Action)
            wm_out = current_world_model(inputs_flat, mode='predict_with_action', gt_actions=pred_actions)

            gt_next_long = s_gen_frame.long()
            gt_obj = gt_next_long[..., 0]
            gt_col = gt_next_long[..., 1]
            gt_state = gt_next_long[..., 2]

            loss_obj = criterion_ce(wm_out['logits_obj'], gt_obj)
            loss_col = criterion_ce(wm_out['logits_col'], gt_col)
            loss_state = criterion_ce(wm_out['logits_state'], gt_state)
            frame_loss = (loss_obj + loss_col + loss_state).mean(dim=(1, 2))

            loss_c_col = criterion_mse(wm_out['carried_col'], s_gen_c_col.float()).mean(dim=1)
            loss_c_obj = criterion_mse(wm_out['carried_obj'], s_gen_c_obj.float()).mean(dim=1)

            score = frame_loss + 10.0 * (loss_c_col + loss_c_obj)

            for i, idx in enumerate(batch_indices):
                losses.append(float(score[i].item()))
                idx_order.append(idx)
                
                action_cache[idx] = int(pred_actions[i].item())

                if data_mode == "model":
                    s_gen_cache[idx] = (
                        s_gen_frame[i].detach().cpu().numpy(),
                        float(s_gen_c_col[i].item()),
                        float(s_gen_c_obj[i].item()),
                    )

    if not idx_order:
        return [], []

    # Rank by score desc, tie-break by idx asc for determinism.
    ranked = sorted(zip(losses, idx_order), key=lambda x: (-x[0], x[1]))

    if use_random_mix:
        n_consistency = int(n_select * (1.0 - random_mix_ratio))
        n_random = max(0, n_select - n_consistency)

        top_pool = ranked[: max(1, n_consistency * 2)]
        chosen_consistency = random.sample(top_pool, k=min(len(top_pool), n_consistency))
        selected = [idx for _, idx in chosen_consistency]

        remaining = [idx for _, idx in ranked if idx not in set(selected)]
        if n_random > 0 and remaining:
            selected.extend(np.random.choice(remaining, size=min(n_random, len(remaining)), replace=False).tolist())
        selected_indices = selected[:n_select]
    else:
        selected_indices = [idx for _, idx in ranked[:n_select]]

    new_collected_data = []
    for idx in tqdm(selected_indices, desc="Collecting Data"):
        sample = dataset[idx]

        frame = sample['frame']
        if torch.is_tensor(frame):
            s_t_np = frame[0].detach().cpu().numpy()
        else:
            s_t_np = np.asarray(frame[0])

        c_col_scalar = float(sample['carried_col'][0].item() if torch.is_tensor(sample['carried_col']) else sample['carried_col'][0])
        c_obj_scalar = float(sample['carried_obj'][0].item() if torch.is_tensor(sample['carried_obj']) else sample['carried_obj'][0])
        
        if idx not in action_cache:
             raise KeyError(f"Index {idx} not found in action_cache. Logic Error.")
        
        if data_mode == "oracle":
            action_to_take = int(sample['action'].item() if torch.is_tensor(sample['action']) else sample['action'])
            s_next_np, c_col_next, c_obj_next = oracle.step(s_t_np, c_col_scalar, c_obj_scalar, action_to_take)
        elif data_mode == "model":
            action_to_take = action_cache[idx] 
            if idx not in s_gen_cache:
                raise KeyError("Missing cached model prediction for selected index.")
            s_next_np, c_col_next, c_obj_next = s_gen_cache[idx]
        else:
            raise ValueError("data_mode must be 'oracle' or 'model'")

        new_collected_data.append({
            'state': s_t_np,
            'carried': np.array([c_col_scalar, c_obj_scalar]),
            'action': action_to_take,
            'next_state': s_next_np,
            'next_carried': np.array([c_col_next, c_obj_next]),
        })

    return selected_indices, new_collected_data

def evaluate(
    model,
    dataloader,
    *,
    device: torch.device,
    forward_carried_loss_weight: float,
    use_random_base_model: bool,
    is_round_0: bool = False,
) -> Dict[str, float]:
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


def train_one_round(
    model,
    train_loader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    forward_carried_loss_weight: float,
    train_from_scratch: bool,
    freeze_model_for_active_learning_fn,
) -> None:
    """Fine-tune world model for one active-learning round."""
    if not train_from_scratch:
        freeze_model_for_active_learning_fn(model)

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(trainable_params, lr=lr)

    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss(reduction="none")  

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "action"}
            inputs["frame"] = inputs["frame"].permute(1, 0, 2, 3, 4)
            inputs["carried_col"] = inputs["carried_col"].permute(1, 0, 2)
            inputs["carried_obj"] = inputs["carried_obj"].permute(1, 0, 2)

            actions = batch["action"].to(device)
            optimizer.zero_grad()
            pred = model(inputs, mode="predict_with_action", gt_actions=actions)

            target_frame = inputs["frame"][1].long()
            gt_obj = target_frame[..., 0]
            gt_col = target_frame[..., 1]
            gt_state = target_frame[..., 2]

            loss_obj = criterion_ce(pred["logits_obj"], gt_obj)
            loss_col = criterion_ce(pred["logits_col"], gt_col)
            loss_state = criterion_ce(pred["logits_state"], gt_state)
         
            ce_map = loss_obj + loss_col + loss_state  # [B, H, W]
            loss_frame_cls = ce_map.mean(dim=(1, 2)).mean() 

            target_c_col = inputs["carried_col"][1].float()
            target_c_obj = inputs["carried_obj"][1].float()
            loss_c_col = criterion_mse(pred["carried_col"], target_c_col)
            loss_c_obj = criterion_mse(pred["carried_obj"], target_c_obj)

            loss = loss_frame_cls + forward_carried_loss_weight * (loss_c_col + loss_c_obj)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1


"""
Utility functions for training and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm

def train_world_model(
    model, 
    train_loader, 
    epochs, 
    lr, 
    device, 
    freeze_func, 
    forward_carried_loss_weight=1.0
):
    """
    Fine-tune the world model.
    
    Args:
        model: The World Model instance.
        train_loader: DataLoader for training data.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: torch.device (cpu or cuda).
        freeze_func: Function to freeze specific model parameters.
        forward_carried_loss_weight: Weight for the carried object loss.
    """
    # Freeze parameters (train only adapter and dynamics)
    freeze_func(model)
    
    # Filter trainable parameters
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(trainable_params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    
    model.train()
    best_loss = float('inf')
    
    print(f"Start training World Model for {epochs} epochs")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    for _ in tqdm(range(epochs), desc="Training World Model"):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Data preparation
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'action'}
            inputs['frame'] = inputs['frame'].permute(1, 0, 2, 3, 4)  # [T, B, H, W, C]
            inputs['carried_col'] = inputs['carried_col'].permute(1, 0, 2)  # [T, B, 1]
            inputs['carried_obj'] = inputs['carried_obj'].permute(1, 0, 2)  # [T, B, 1]
            actions = batch['action'].to(device)  # [B]
            
            # Forward pass
            pred = model(inputs, mode='predict_with_action', gt_actions=actions)
            
            # Calculate pixel losses
            gt_next_frame = inputs['frame'][1]  # [B, H, W, C]
            gt_next_long = gt_next_frame.long()
            gt_obj = gt_next_long[..., 0]   # [B, H, W]
            gt_col = gt_next_long[..., 1]   # [B, H, W]
            gt_state = gt_next_long[..., 2] # [B, H, W]
            
            loss_obj = criterion_ce(pred['logits_obj'], gt_obj)
            loss_col = criterion_ce(pred['logits_col'], gt_col)
            loss_state = criterion_ce(pred['logits_state'], gt_state)
            pixel_loss = (loss_obj + loss_col + loss_state).mean()
            
            # Calculate carried object losses
            gt_carried_col = inputs['carried_col'][1].float()  # [B, 1]
            gt_carried_obj = inputs['carried_obj'][1].float()  # [B, 1]
            loss_c_col = criterion_mse(pred['carried_col'], gt_carried_col)
            loss_c_obj = criterion_mse(pred['carried_obj'], gt_carried_obj)
            carried_loss = forward_carried_loss_weight * (loss_c_col + loss_c_obj)
            
            # Total loss
            loss = pixel_loss + carried_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    print(f"World Model training finished. Best loss: {best_loss:.6f}")
    return model


def train_inverse_model(
    train_loader, 
    num_actions, 
    epochs, 
    lr, 
    device, 
    model_class,
):
    """
    Train the Inverse Model from scratch.
    
    Args:
        train_loader: DataLoader for training data.
        num_actions: Number of possible actions.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: torch.device.
        model_class: The class constructor for IDM.
    """
    print(f"Start training Inverse Model for {epochs} epochs")
    
    # Initialize model
    model = model_class(num_actions=num_actions).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    best_loss = float('inf')
    
    for _ in tqdm(range(epochs), desc="Training Inverse Model"):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Data preparation
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'action'}
            inputs['frame'] = inputs['frame'].permute(1, 0, 2, 3, 4)
            inputs['carried_col'] = inputs['carried_col'].permute(1, 0, 2)
            inputs['carried_obj'] = inputs['carried_obj'].permute(1, 0, 2)
            actions = batch['action'].to(device)
            
            # Forward pass
            output = model(inputs)
            logits = output
            # Cross Entropy Loss
            ce_loss = criterion(logits, actions)

            loss = ce_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    print(f"Inverse Model training finished. Best loss: {best_loss:.6f}")
    return model

def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_model_for_active_learning(model):
    """
    Freeze specific modules of WorldModel for fine-tuning.
    Only the dynamics adapter and decoder are trainable.
    """
    modules_to_freeze = [
        model.encoder_cnn,
        model.obj_embedding,
        model.col_embedding,
        model.state_embedding,
        model.scalar_embed,
        model.fusion_conv,
        model.vq_encoder_net,
        model.vq_layer,
        model.prior_net,
    ]
    
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False
    
    print(f"Frozen {len(modules_to_freeze)} modules for fine-tuning.")


def prepare_batch_for_model(batch, device):
    """
    Prepare batch data for model input.
    Converts batch format to model expected format.
    """
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'action'}
    inputs['frame'] = inputs['frame'].permute(1, 0, 2, 3, 4)  # [T, B, H, W, C]
    inputs['carried_col'] = inputs['carried_col'].permute(1, 0, 2)  # [T, B, 1]
    inputs['carried_obj'] = inputs['carried_obj'].permute(1, 0, 2)  # [T, B, 1]
    actions = batch['action'].to(device)  # [B]
    return inputs, actions


def test_world_model(model, test_loader, forward_carried_loss_weight=10.0, device=None):
    """
    Test world model and compute dynamic accuracy.
    
    Args:
        model: Trained world model
        test_loader: Test data loader
        forward_carried_loss_weight: Weight for carried loss
        device: Device to run on
    
    Returns:
        Dictionary with dynamic accuracy metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    stats = {
        'total_dyn_pixels': 0,
        'correct_dyn_pixels': 0,
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing World Model"):
            inputs, actions = prepare_batch_for_model(batch, device)
            
            # Forward pass
            pred = model(inputs, mode='predict_with_action', gt_actions=actions)
            
            gt_next_frame = inputs['frame'][1]  # [B, H, W, C]
            prev_frame = inputs['frame'][0].float()
            
            # Get carried states
            prev_carried_col = inputs['carried_col'][0]
            prev_carried_obj = inputs['carried_obj'][0]
            gt_next_carried_col = inputs['carried_col'][1]
            gt_next_carried_obj = inputs['carried_obj'][1]
            
            # Compute dynamics mask
            diff_map = torch.abs(prev_frame - gt_next_frame.float()).sum(dim=-1)
            diff_mask = (diff_map > 0.01)  # [B, H, W]
            
            # Check if carried state changed
            carried_col_changed = (prev_carried_col != gt_next_carried_col).squeeze(-1)
            carried_obj_changed = (prev_carried_obj != gt_next_carried_obj).squeeze(-1)
            carried_changed = carried_col_changed | carried_obj_changed
            
            # Compute predictions
            gt_next_long = gt_next_frame.long()
            pred_obj = torch.argmax(pred['logits_obj'], dim=1)
            pred_col = torch.argmax(pred['logits_col'], dim=1)
            pred_state = torch.argmax(pred['logits_state'], dim=1)
            pred_int = torch.stack([pred_obj, pred_col, pred_state], dim=-1).long()
            
            # Pixel accuracy
            pixel_correct_mask = (pred_int == gt_next_long).all(dim=-1)
            
            # Carried accuracy
            pred_carried_col = torch.round(pred['carried_col']).long()
            pred_carried_obj = torch.round(pred['carried_obj']).long()
            gt_carried_col_long = gt_next_carried_col.long()
            gt_carried_obj_long = gt_next_carried_obj.long()
            carried_col_correct = (pred_carried_col == gt_carried_col_long).squeeze(-1)
            carried_obj_correct = (pred_carried_obj == gt_carried_obj_long).squeeze(-1)
            carried_correct = carried_col_correct & carried_obj_correct
            
            # Dynamic pixel accuracy
            if diff_mask.sum() > 0:
                correct_dyn = (pixel_correct_mask & diff_mask)
                stats['correct_dyn_pixels'] += correct_dyn.sum().item()
                stats['total_dyn_pixels'] += diff_mask.sum().item()
            
            # Carried state changes count as dynamic pixels
            if carried_changed.sum() > 0:
                correct_carried_dyn = carried_correct[carried_changed].sum().item()
                total_carried_dyn = carried_changed.sum().item()
                stats['correct_dyn_pixels'] += correct_carried_dyn
                stats['total_dyn_pixels'] += total_carried_dyn
    
    # Compute final metrics
    dyn_acc = stats['correct_dyn_pixels'] / max(1, stats['total_dyn_pixels'])
    
    return {
        'dyn_acc': dyn_acc,
        'correct_dyn_pixels': stats['correct_dyn_pixels'],
        'total_dyn_pixels': stats['total_dyn_pixels'],
    }


def test_inverse_model(inverse_model, oracle, test_loader, device=None):
    """
    Test inverse model and compute dynamic accuracy.
    
    Args:
        inverse_model: Trained inverse model
        oracle: MiniGridPhysicsOracle instance
        test_loader: Test data loader
        device: Device to run on
    
    Returns:
        Dictionary with dynamic accuracy metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inverse_model.eval()
    
    stats = {
        'total_dyn_pixels': 0,
        'correct_dyn_pixels': 0,
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Inverse Model"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'action'}
            frames = inputs['frame']  # [B, T, H, W, C]
            carried_col = inputs['carried_col']  # [B, T, 1]
            carried_obj = inputs['carried_obj']  # [B, T, 1]
            
            curr_frame = frames[:, 0]
            next_frame_gt = frames[:, 1]
            curr_c_col = carried_col[:, 0]
            curr_c_obj = carried_obj[:, 0]
            next_c_col_gt = carried_col[:, 1]
            next_c_obj_gt = carried_obj[:, 1]
            
            # Prepare inverse model input
            inv_frames = torch.stack([curr_frame, next_frame_gt], dim=0)
            inv_c_col = torch.stack([curr_c_col, next_c_col_gt], dim=0)
            inv_c_obj = torch.stack([curr_c_obj, next_c_obj_gt], dim=0)
            
            inv_inputs = {
                'frame': inv_frames,
                'carried_col': inv_c_col,
                'carried_obj': inv_c_obj
            }
            
            # Predict action
            output = inverse_model(inv_inputs)
            if isinstance(output, tuple):
                pred_action_logits, _ = output
            else:
                pred_action_logits = output
            pred_actions = torch.argmax(pred_action_logits, dim=1)
            
            # Simulate next state using oracle
            batch_size = curr_frame.shape[0]
            pred_next_frames = []
            pred_next_c_cols = []
            pred_next_c_objs = []
            
            for i in range(batch_size):
                s_t_np = curr_frame[i].cpu().numpy()
                c_col_scalar = int(curr_c_col[i].item())
                c_obj_scalar = int(curr_c_obj[i].item())
                action_to_take = int(pred_actions[i].item())
                
                s_next_np, c_col_next, c_obj_next = oracle.step(
                    s_t_np, c_col_scalar, c_obj_scalar, action_to_take
                )
                
                pred_next_frames.append(s_next_np)
                pred_next_c_cols.append(c_col_next)
                pred_next_c_objs.append(c_obj_next)
            
            pred_next_frame = torch.from_numpy(np.stack(pred_next_frames)).to(device)
            pred_next_c_col = torch.tensor(pred_next_c_cols, dtype=torch.float32).unsqueeze(1).to(device)
            pred_next_c_obj = torch.tensor(pred_next_c_objs, dtype=torch.float32).unsqueeze(1).to(device)
            
            # Compute dynamics mask
            curr_frame_f = curr_frame.float()
            next_frame_gt_f = next_frame_gt.float()
            diff_map = torch.abs(curr_frame_f - next_frame_gt_f).sum(dim=-1)
            diff_mask = (diff_map > 0.01)
            
            # Check carried state changes
            carried_col_changed = (curr_c_col != next_c_col_gt).squeeze(-1)
            carried_obj_changed = (curr_c_obj != next_c_obj_gt).squeeze(-1)
            carried_changed = carried_col_changed | carried_obj_changed
            
            # Pixel accuracy
            pred_next_long = pred_next_frame.long()
            gt_next_long = next_frame_gt.long()
            pixel_correct_mask = (pred_next_long == gt_next_long).all(dim=-1)
            
            # Carried accuracy
            pred_carried_col = pred_next_c_col.long()
            pred_carried_obj = pred_next_c_obj.long()
            gt_carried_col_long = next_c_col_gt.long()
            gt_carried_obj_long = next_c_obj_gt.long()
            carried_col_correct = (pred_carried_col == gt_carried_col_long).squeeze(-1)
            carried_obj_correct = (pred_carried_obj == gt_carried_obj_long).squeeze(-1)
            carried_correct = carried_col_correct & carried_obj_correct
            
            # Dynamic pixel accuracy
            if diff_mask.sum() > 0:
                correct_dyn = (pixel_correct_mask & diff_mask)
                stats['correct_dyn_pixels'] += correct_dyn.sum().item()
                stats['total_dyn_pixels'] += diff_mask.sum().item()
            
            # Carried state changes count as dynamic pixels
            if carried_changed.sum() > 0:
                correct_carried_dyn = carried_correct[carried_changed].sum().item()
                total_carried_dyn = carried_changed.sum().item()
                stats['correct_dyn_pixels'] += correct_carried_dyn
                stats['total_dyn_pixels'] += total_carried_dyn
    
    # Compute final metrics
    dyn_accuracy = stats['correct_dyn_pixels'] / max(1, stats['total_dyn_pixels'])
    
    return {
        'dyn_accuracy': dyn_accuracy,
        'correct_dyn_pixels': stats['correct_dyn_pixels'],
        'total_dyn_pixels': stats['total_dyn_pixels'],
    }
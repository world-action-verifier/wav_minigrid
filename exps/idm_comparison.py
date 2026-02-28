import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from asim_minigrid.models import DenseIDM, SparseIDM
from asim_minigrid.dataset import NormalizedDataset, MiniGridDynamicsDataset
from asim_minigrid.config import IDM_COMPARISON, DEVICE

# Configuration
TRAIN_DATA_PATH = IDM_COMPARISON["TRAIN_DATA_PATH"]
TEST_DATA_PATH = IDM_COMPARISON["TEST_DATA_PATH"]
BATCH_SIZE = IDM_COMPARISON["BATCH_SIZE"]
LR = IDM_COMPARISON["LR"]
EPOCHS = IDM_COMPARISON["EPOCHS"]
NUM_ACTIONS = IDM_COMPARISON["NUM_ACTIONS"]
SEED = IDM_COMPARISON["SEED"]
SEED = 46
SAVE_MODEL = False
# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def get_dataloaders(train_path, test_path, batch_size=64):
    """Create train and test dataloaders."""
    train_dataset = NormalizedDataset(MiniGridDynamicsDataset(train_path))
    test_dataset = NormalizedDataset(MiniGridDynamicsDataset(test_path))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def evaluate_model(model, dataloader, device, verbose=False):
    """Evaluate model accuracy. Returns action accuracies dict and average accuracy."""
    model.eval()
    correct = 0
    total = 0
    action_correct = torch.zeros(NUM_ACTIONS, dtype=torch.long)
    action_total = torch.zeros(NUM_ACTIONS, dtype=torch.long)
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'action'}
            inputs['frame'] = inputs['frame'].permute(1, 0, 2, 3, 4)
            inputs['carried_col'] = inputs['carried_col'].permute(1, 0, 2)
            inputs['carried_obj'] = inputs['carried_obj'].permute(1, 0, 2)
            actions = batch['action'].to(device)
            
            outputs = model(inputs)
            if isinstance(model, SparseIDM):
                logits, _, _ = outputs
            else:
                logits = outputs
            pred = torch.argmax(logits, dim=1)
            
            correct_batch = (pred == actions)
            correct += correct_batch.sum().item()
            total += actions.size(0)
            
            for a in range(NUM_ACTIONS):
                mask = (actions == a)
                action_total[a] += mask.sum().item()
                if mask.any():
                    action_correct[a] += (correct_batch & mask).sum().item()
    
    acc = correct / total
    action_accuracies = {}
    
    if verbose:
        print("\nPer-Action Accuracy:")
        for a in range(NUM_ACTIONS):
            count = action_total[a].item()
            if count > 0:
                p = action_correct[a].item() / count
                action_accuracies[a] = p
                print(f"  Action {a}: {p*100:6.2f}% ({action_correct[a]}/{count})")
            else:
                action_accuracies[a] = 0.0
    else:
        for a in range(NUM_ACTIONS):
            count = action_total[a].item()
            if count > 0:
                action_accuracies[a] = action_correct[a].item() / count
            else:
                action_accuracies[a] = 0.0
    
    return action_accuracies, acc


def train_model(model, train_loader, test_loader, model_name, device, epochs=200, lr=1e-4):
    """Train a model and evaluate on test set. Returns model, action accuracies, and average accuracy."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    num_batches = len(train_loader)
    total_steps = epochs * num_batches if num_batches > 0 else 1

    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            obs_inputs = {k: v.to(device) for k, v in batch.items() if k != 'action'}
            obs_inputs['frame'] = obs_inputs['frame'].permute(1, 0, 2, 3, 4)
            obs_inputs['carried_col'] = obs_inputs['carried_col'].permute(1, 0, 2)
            obs_inputs['carried_obj'] = obs_inputs['carried_obj'].permute(1, 0, 2)
            target_actions = batch['action'].to(device)
            
            optimizer.zero_grad()
            outputs = model(obs_inputs)
            
            if isinstance(model, SparseIDM):
                action_logits, mask, mask_logits = outputs

                ce_loss = F.cross_entropy(action_logits, target_actions)

                target_cells = 2.0
                cells_selected = mask.sum(dim=(1, 2, 3))
                # sparsity_loss = torch.mean((cells_selected - target_cells) ** 2)
                sparsity_loss = torch.mean(torch.abs(cells_selected - target_cells))

                # Sparse Loss warmup: Lambda increases linearly from 0 to the target value.
                target_lambda = 0.1
                current_step = epoch * num_batches + batch_idx
                warmup_lambda = target_lambda * min(float(current_step) / max(total_steps - 1, 1), 1.0)

                loss = ce_loss + warmup_lambda * sparsity_loss
                logits_for_metrics = action_logits
            else:
                logits_for_metrics = outputs
                loss = criterion(logits_for_metrics, target_actions)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(logits_for_metrics, dim=1)
            total_correct += (pred == target_actions).sum().item()
            total_samples += target_actions.size(0)
            num_batches += 1
        
    # Final evaluation on test set
    # print(f"\nEvaluating {model_name} on test set...")
    if SAVE_MODEL:
        torch.save(model.state_dict(), f"{model_name}.pth")
        
    action_accuracies, avg_acc = evaluate_model(model, test_loader, device, verbose=False)
    
    return model, action_accuracies, avg_acc

def main():
    print(f"Device: {DEVICE}")
    print(f"Training data: {TRAIN_DATA_PATH}")
    print(f"Test data: {TEST_DATA_PATH}")
    
    # Load data
    train_loader, test_loader = get_dataloaders(TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE)
    
    sample_batch = next(iter(train_loader))
    frame_sample = sample_batch["frame"]  # 形状通常为 [B, T, H, W, 3]
    grid_h, grid_w = frame_sample.shape[2], frame_sample.shape[3]
    
    results = {}
    
    # Train and evaluate DenseIDM
    dense_model = DenseIDM(num_actions=NUM_ACTIONS)
    dense_model, dense_action_acc, dense_avg_acc = train_model(
        dense_model, train_loader, test_loader, 
        "DenseIDM", DEVICE, EPOCHS, LR
    )
    results['DenseIDM'] = {'action_acc': dense_action_acc, 'avg_acc': dense_avg_acc}
    
    # Train and evaluate SparseIDM
    sparse_model = SparseIDM(grid_h=grid_h, grid_w=grid_w, num_actions=NUM_ACTIONS)
    sparse_model, sparse_action_acc, sparse_avg_acc = train_model(
        sparse_model, train_loader, test_loader,
        "SparseIDM", DEVICE, EPOCHS, LR
    )
    results['SparseIDM'] = {'action_acc': sparse_action_acc, 'avg_acc': sparse_avg_acc}
    
    # Print comparison results - interaction actions (3, 4, 5, 6) and average
    interaction_actions = [3, 4, 5, 6]  # Pickup, Drop, Toggle, Done
    
    print(f"\n{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Average':<12} {'Pickup':<12} {'Drop':<12} {'Toggle':<12} {'Done':<12}")
    print(f"{'-'*60}")
    for model_name, result in results.items():
        action_acc = result['action_acc']
        # Calculate average only over interaction actions
        interaction_accs = [action_acc[a] for a in interaction_actions]
        avg_acc = np.mean(interaction_accs)
        acc_str = f"{avg_acc*100:.2f}%"
        action_strs = [f"{action_acc[a]*100:.2f}%" for a in interaction_actions]
        print(f"{model_name:<15} {acc_str:<12} " + " ".join(f"{s:<12}" for s in action_strs))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

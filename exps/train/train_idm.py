#!/usr/bin/env python3
"""
Train SparseIDM from scratch.
This script trains a SparseIDM model on the base dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from asim_minigrid.models import SparseIDM
from config import DEVICE, IDM_TRAINING
from train_utils import (
    set_all_seeds,
    get_dataloaders_with_validation,
    evaluate_idm,
    prepare_batch_inputs,
)

# Configuration from config.py
BASE_DATA_PATH = IDM_TRAINING["BASE_DATA_PATH"]
BATCH_SIZE = IDM_TRAINING["BATCH_SIZE"]
LR = IDM_TRAINING["LR"]
EPOCHS_FIRST_ROUND = IDM_TRAINING["EPOCHS_FIRST_ROUND"]
NUM_ACTIONS = IDM_TRAINING["NUM_ACTIONS"]
SEED = IDM_TRAINING["SEED"]


def train_idm(args):
    """Train SparseIDM from scratch."""
    # Set random seeds
    set_all_seeds(SEED)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders_with_validation(
        BASE_DATA_PATH, 
        BATCH_SIZE, 
        train_ratio=args.train_ratio,
        seed=SEED
    )
    
    # Initialize model
    print(f"\n=== Initializing SparseIDM ===")
    model = SparseIDM(num_actions=NUM_ACTIONS).to(DEVICE)
    print(f"Model initialized with {NUM_ACTIONS} actions")
    
    # Evaluate initial model (random weights)
    print("\n=== Initial Evaluation ===")
    initial_action_acc, initial_avg_acc = evaluate_idm(model, val_loader, DEVICE, num_actions=NUM_ACTIONS, verbose=True)
    print(f"Initial Average Accuracy: {initial_avg_acc*100:.2f}%")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\n=== Training for {EPOCHS_FIRST_ROUND} epochs ===")
    for epoch in tqdm(range(EPOCHS_FIRST_ROUND), desc="Training SparseIDM"):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch in train_loader:
            inputs, actions = prepare_batch_inputs(batch, DEVICE)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            epoch_correct += (pred == actions).sum().item()
            epoch_total += actions.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == EPOCHS_FIRST_ROUND - 1:
            val_action_acc, val_avg_acc = evaluate_idm(model, val_loader, DEVICE, num_actions=NUM_ACTIONS, verbose=False)
            
            if val_avg_acc > best_val_acc:
                best_val_acc = val_avg_acc
                best_epoch = epoch + 1
                # Save best model
                save_path = os.path.join(args.save_dir, "best_sparse_idm.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'val_acc': val_avg_acc,
                    'val_action_acc': val_action_acc,
                    'num_actions': NUM_ACTIONS,
                }, save_path)
            
            if (epoch + 1) % args.eval_freq == 0:
                print(f"Epoch {epoch+1}/{EPOCHS_FIRST_ROUND} | "
                      f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                      f"Val Acc: {val_avg_acc*100:.2f}%")
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_action_acc, final_avg_acc = evaluate_idm(model, val_loader, DEVICE, num_actions=NUM_ACTIONS, verbose=True)
    print(f"Final Average Accuracy: {final_avg_acc*100:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    
    # Save final model
    final_save_path = os.path.join(args.save_dir, "final_sparse_idm.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': EPOCHS_FIRST_ROUND,
        'val_acc': final_avg_acc,
        'val_action_acc': final_action_acc,
        'initial_acc': initial_avg_acc,
        'num_actions': NUM_ACTIONS,
        'lr': LR,
        'epochs': EPOCHS_FIRST_ROUND,
    }, final_save_path)
    print(f"\nFinal model saved to: {final_save_path}")
    
    print("\n=== Training Complete ===")
    print(f"Initial Accuracy: {initial_avg_acc*100:.2f}%")
    print(f"Final Accuracy: {final_avg_acc*100:.2f}%")
    print(f"Improvement: {(final_avg_acc - initial_avg_acc)*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SparseIDM from scratch")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(project_root, "checkpoints", "sparse_idm"),
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (rest for validation)"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=50,
        help="Frequency of evaluation (in epochs)"
    )
    
    args = parser.parse_args()
    train_idm(args)


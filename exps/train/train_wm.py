#!/usr/bin/env python3
"""
Fine-tune World Model from video model checkpoint.
This script loads a pretrained video model and fine-tunes it on the base dataset.
"""

import torch
from torch.utils.data import DataLoader
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from asim_minigrid.models import WorldModel
from asim_minigrid.dataset import MiniGridDynamicsDataset, NormalizedDataset
from asim_minigrid.utils import freeze_model_for_active_learning
from asim_minigrid.al_utils import (
    train_one_round,
    evaluate,
)
from config import DEVICE, WM_FINETUNING
from train_utils import (
    set_all_seeds,
    load_model_checkpoint,
)

# Configuration from config.py
VIDEO_STAGE1_CKPT = WM_FINETUNING["VIDEO_STAGE1_CKPT"]
BASE_DATA_PATH = WM_FINETUNING["BASE_DATA_PATH"]
BATCH_SIZE = WM_FINETUNING["BATCH_SIZE"]
LR = WM_FINETUNING["LR"]
EPOCHS_FIRST_ROUND = WM_FINETUNING["EPOCHS_FIRST_ROUND"]
FORWARD_CARRIED_LOSS_WEIGHT = WM_FINETUNING["FORWARD_CARRIED_LOSS_WEIGHT"]
TRAIN_FROM_SCRATCH = WM_FINETUNING["TRAIN_FROM_SCRATCH"]
SEED = WM_FINETUNING["SEED"]


def load_video_model(model_path, obs_shape, num_actions=7):
    """Load video model checkpoint."""
    set_all_seeds(SEED)
    model = WorldModel(obs_shape, num_actions).to(DEVICE)
    model = load_model_checkpoint(model, model_path, DEVICE, strict=False)
    model.eval()
    return model


def train_world_model(args):
    """Fine-tune world model from video model checkpoint."""
    # Set random seeds
    set_all_seeds(SEED)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading training dataset from: {BASE_DATA_PATH}")
    train_dataset = MiniGridDynamicsDataset(BASE_DATA_PATH)
    obs_shape = train_dataset.states.shape[1:]
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Observation shape: {obs_shape}")
    
    # Normalize dataset
    train_dataset_normalized = NormalizedDataset(train_dataset)
    train_loader = DataLoader(train_dataset_normalized, batch_size=BATCH_SIZE, shuffle=True)
    
    # Load video model checkpoint
    model = load_video_model(VIDEO_STAGE1_CKPT, obs_shape, num_actions=7)
    
    # Evaluate initial model
    print("\n=== Initial Evaluation ===")
    initial_metrics = evaluate(
        model,
        train_loader,
        device=DEVICE,
        forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT,
        use_random_base_model=False,
        is_round_0=False,
    )
    print(f"Initial MSE: {initial_metrics['mse']:.5f}")
    
    # Fine-tune the model
    print(f"\n=== Training for {EPOCHS_FIRST_ROUND} epochs ===")
    train_one_round(
        model,
        train_loader,
        device=DEVICE,
        epochs=EPOCHS_FIRST_ROUND,
        lr=LR,
        forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT,
        train_from_scratch=TRAIN_FROM_SCRATCH,
        freeze_model_for_active_learning_fn=freeze_model_for_active_learning,
    )
    
    # Evaluate final model
    print("\n=== Final Evaluation ===")
    final_metrics = evaluate(
        model,
        train_loader,
        device=DEVICE,
        forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT,
        use_random_base_model=False,
        is_round_0=False,
    )
    print(f"Final MSE: {final_metrics['mse']:.5f}")
    
    # Save model
    save_path = os.path.join(args.save_dir, "fine_tuned_world_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'initial_mse': initial_metrics['mse'],
        'final_mse': final_metrics['mse'],
        'epochs': EPOCHS_FIRST_ROUND,
        'lr': LR,
        'forward_carried_loss_weight': FORWARD_CARRIED_LOSS_WEIGHT,
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    
    print("\n=== Training Complete ===")
    print(f"Initial MSE: {initial_metrics['mse']:.5f}")
    print(f"Final MSE: {final_metrics['mse']:.5f}")
    print(f"Improvement: {initial_metrics['mse'] - final_metrics['mse']:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune World Model from video model checkpoint")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(project_root, "checkpoints", "fine_tuned_world_model"),
        help="Directory to save the fine-tuned model"
    )
    
    args = parser.parse_args()
    train_world_model(args)


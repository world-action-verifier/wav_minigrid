"""
Script to investigate the impact of different training data sizes on World Model and Inverse Model performance.
"""

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys
import argparse
import json

from asim_minigrid.models import WorldModel, SparseIDM
from asim_minigrid.dataset import MiniGridDynamicsDataset, NormalizedDataset
from asim_minigrid.evaluate_generation import MiniGridPhysicsOracle
from asim_minigrid.utils import (
    freeze_model_for_active_learning,
    train_world_model,
    train_inverse_model,
    test_world_model,
    test_inverse_model,
    set_all_seeds
)
from asim_minigrid.config import DATA_EFFICIENCY_GAP, DEVICE

# Configuration
DATA_PATH = DATA_EFFICIENCY_GAP["DATA_PATH"]
TEST_SET_PATH = DATA_EFFICIENCY_GAP["TEST_SET_PATH"]
VIDEO_STAGE1_CKPT = DATA_EFFICIENCY_GAP["VIDEO_STAGE1_CKPT"]
BATCH_SIZE = DATA_EFFICIENCY_GAP["BATCH_SIZE"]
LR = DATA_EFFICIENCY_GAP["LR"]
EPOCHS = DATA_EFFICIENCY_GAP["EPOCHS"]
INVERSE_MODEL_EPOCHS = DATA_EFFICIENCY_GAP["INVERSE_MODEL_EPOCHS"]
FORWARD_CARRIED_LOSS_WEIGHT = DATA_EFFICIENCY_GAP["FORWARD_CARRIED_LOSS_WEIGHT"]
TARGET_PIXELS = DATA_EFFICIENCY_GAP["TARGET_PIXELS"]
MASK_L1_LAMBDA = DATA_EFFICIENCY_GAP["MASK_L1_LAMBDA"]
TRAIN_RATIOS = DATA_EFFICIENCY_GAP["TRAIN_RATIOS"]
SEED = DATA_EFFICIENCY_GAP["SEED"]

def load_pretrained_world_model(model_path, obs_shape, num_actions):
    """Load pretrained video model as initial weights for world model."""
    print(f"Loading pretrained model from: {model_path}")
    model = WorldModel(obs_shape, num_actions).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    print("Pretrained model loaded successfully")
    return model

def run_experiment(train_ratio, train_dataset, test_loader, obs_shape, num_actions, skip_world_model=False):
    """Run one experiment with given training data ratio."""
    print(f"\n{'='*80}")
    print(f"Experiment: Training data ratio = {train_ratio*100:.0f}%")
    print(f"{'='*80}")
    
    set_all_seeds(SEED)
    
    dataset_size = len(train_dataset)
    num_samples = max(1, int(dataset_size * train_ratio))
    indices = np.random.choice(dataset_size, size=num_samples, replace=False)
    subset = Subset(train_dataset, indices)
    train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training set size: {len(subset)} / {dataset_size} ({train_ratio*100:.0f}%)")
    
    if not skip_world_model:
        print("\n--- Training World Model ---")
        world_model = load_pretrained_world_model(VIDEO_STAGE1_CKPT, obs_shape, num_actions)
        world_model = train_world_model(
            world_model,
            train_loader,
            epochs=EPOCHS,
            lr=LR,
            device=DEVICE,
            freeze_func=freeze_model_for_active_learning,
            forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT
        )
    else:
        print("\n--- Skipping World Model training and testing, only training Inverse Model ---")
        world_model = None
    
    print("\n--- Training Inverse Model ---")
    inverse_model = train_inverse_model(
        train_loader,
        num_actions,
        epochs=INVERSE_MODEL_EPOCHS,
        lr=LR,
        device=DEVICE,
        model_class=SparseIDM,
    )
    
    if not skip_world_model:
        print("\n--- Testing World Model ---")
        world_model.eval()
        world_model_results = test_world_model(
            world_model,
            test_loader,
            forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT,
            device=DEVICE
        )
    else:
        world_model_results = {
            'dyn_acc': 0.0,
        }
    
    print("\n--- Testing Inverse Model ---")
    inverse_model.eval()
    oracle = MiniGridPhysicsOracle()
    inverse_model_results = test_inverse_model(
        inverse_model,
        oracle,
        test_loader,
        device=DEVICE
    )
    
    result = {
        'train_ratio': train_ratio,
        'train_size': len(subset),
        'world_model': {
            'dyn_acc': world_model_results.get('dyn_acc', 0.0),
        },
        'inverse_model': {
            'dyn_accuracy': inverse_model_results.get('dyn_accuracy', 0.0),
        },
    }

    result['skip_world_model'] = skip_world_model
    return result

def print_summary(results_list):
    """Print experiment results summary."""
    print(f"\n{'='*80}")
    print("Experiment Results Summary")
    print(f"{'='*80}")
    
    print(f"\n{'Ratio':<12} {'Train Size':<12} {'WM Dyn Acc':<15} {'IM Dyn Acc':<15}")
    print("-" * 60)
    
    for result in results_list:
        ratio = result['train_ratio']
        size = result['train_size']
        wm_dyn_acc = result['world_model']['dyn_acc']
        im_dyn_acc = result['inverse_model']['dyn_accuracy']
        
        print(f"{ratio*100:>6.0f}%      {size:>10}    {wm_dyn_acc:>13.4f}  {im_dyn_acc:>13.4f}")
    
    print(f"\n{'='*80}")

def save_results(results_list, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f"\nResults saved to: {output_path}")

def main():
    global EPOCHS, LR, BATCH_SIZE, SEED, TRAIN_RATIOS
    
    parser = argparse.ArgumentParser(description='Investigate the impact of different training data sizes on model performance')
    parser.add_argument('--train_ratios', type=float, nargs='+', default=TRAIN_RATIOS,
                        help='Training data ratio list (e.g., 0.2 0.4 0.6 0.8 1.0)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LR,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='data_ratio_experiment_results.json',
                        help='Output path for results')
    parser.add_argument('--skip_world_model', action='store_true',
                        help='Only train and test Inverse Model, skip World Model')
    
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size
    SEED = args.seed
    TRAIN_RATIOS = args.train_ratios
    
    print("="*80)
    print("Experiment: Impact of Training Data Size on Model Performance")
    print("="*80)
    print(f"Training set path: {DATA_PATH}")
    print(f"Test set path: {TEST_SET_PATH}")
    print(f"Pretrained model path: {VIDEO_STAGE1_CKPT}")
    print(f"Training data ratios: {TRAIN_RATIOS}")
    print(f"Training epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Random seed: {SEED}")
    print(f"Skip World Model: {args.skip_world_model}")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    print("\nLoading datasets...")
    train_dataset_full = MiniGridDynamicsDataset(DATA_PATH)
    train_dataset_full = NormalizedDataset(train_dataset_full)
    test_dataset = MiniGridDynamicsDataset(TEST_SET_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    obs_shape = test_dataset.states.shape[1:]
    num_actions = 7
    
    print(f"Full training set size: {len(train_dataset_full)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    
    results_list = []
    for i, ratio in enumerate(TRAIN_RATIOS):
        result = run_experiment(
            ratio,
            train_dataset_full,
            test_loader,
            obs_shape,
            num_actions,
            skip_world_model=args.skip_world_model,
        )
        results_list.append(result)
    
    print_summary(results_list)
    save_results(results_list, args.output)
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()


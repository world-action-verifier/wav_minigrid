"""
Script to investigate the impact of different background noise levels (1-4 noise floors)
on World Model and Inverse Model performance.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import json

from asim_minigrid.models import WorldModel, SparseIDM, DenseIDM
from asim_minigrid.dataset import MiniGridDynamicsDataset, NormalizedDataset
from asim_minigrid.evaluate_generation import MiniGridPhysicsOracle
from asim_minigrid.utils import (
    train_world_model,
    train_inverse_model,
    test_world_model,
    test_inverse_model,
    set_all_seeds
)
from asim_minigrid.config import NOISE_ROBUSTNESS, DEVICE

NOISE_EXPERIMENTS = NOISE_ROBUSTNESS["EXPERIMENTS"]
SAVE_MODEL = NOISE_ROBUSTNESS["SAVE_MODEL"]
VIDEO_STAGE1_CKPT = NOISE_ROBUSTNESS["VIDEO_STAGE1_CKPT"]
BATCH_SIZE = NOISE_ROBUSTNESS["BATCH_SIZE"]
LR = NOISE_ROBUSTNESS["LR"]
EPOCHS = NOISE_ROBUSTNESS["EPOCHS"]
INVERSE_MODEL_EPOCHS = NOISE_ROBUSTNESS["INVERSE_MODEL_EPOCHS"]
FORWARD_CARRIED_LOSS_WEIGHT = NOISE_ROBUSTNESS["FORWARD_CARRIED_LOSS_WEIGHT"]
SEED = NOISE_ROBUSTNESS["SEED"]

def load_pretrained_world_model(model_path, obs_shape, num_actions):
    model = WorldModel(obs_shape, num_actions).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    return model

def run_noise_experiment(exp_cfg, skip_world_model=False, skip_inverse_model=False):
    """运行特定噪声等级的实验"""
    print(f"\n{'#'*80}")
    print(f" Running Experiment: {exp_cfg['name']} ")
    print(f"{'#'*80}")
    
    set_all_seeds(SEED)
    
    # 1. 加载数据集
    print(f"Loading Train: {exp_cfg['train']}")
    train_dataset = NormalizedDataset(MiniGridDynamicsDataset(exp_cfg['train']))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Loading Test: {exp_cfg['test']}")
    test_dataset = MiniGridDynamicsDataset(exp_cfg['test']) # 测试集通常不需要增强
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    obs_shape = test_dataset.states.shape[1:]
    num_actions = 7
    
    # 2. 训练 World Model
    world_model_results = {'dyn_acc': 0.0}
    if not skip_world_model:
        print("\n--- [World Model] Training ---")
        world_model = load_pretrained_world_model(VIDEO_STAGE1_CKPT, obs_shape, num_actions)
        world_model = train_world_model(
            world_model,
            train_loader,
            epochs=EPOCHS,
            lr=LR,
            device=DEVICE,
            freeze_func=None,
            forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT
        )
        
        print("--- [World Model] Testing ---")
        world_model.eval()
        world_model_results = test_world_model(
            world_model,
            test_loader,
            device=DEVICE
        )
    else:
        world_model_results = {'dyn_acc': 0.0}
    
    # 3. 训练 Inverse Model
    if not skip_inverse_model:
        print("\n--- [Inverse Model] Training ---")
        inverse_model = train_inverse_model(
            train_loader,
            num_actions,
            epochs=INVERSE_MODEL_EPOCHS,
            lr=LR,
            device=DEVICE,
            model_class=SparseIDM,
        )
        
        print("--- [Inverse Model] Testing ---")
        inverse_model.eval()
        oracle = MiniGridPhysicsOracle()
        inverse_model_results = test_inverse_model(
            inverse_model,
            oracle,
            test_loader,
            device=DEVICE
        )
    else:
        inverse_model_results = {'dyn_accuracy': 0.0}

    if SAVE_MODEL:
        torch.save(world_model.state_dict(), f"world_model_{exp_cfg['name']}.pth")
        torch.save(inverse_model.state_dict(), f"dense_inverse_model_{exp_cfg['name']}.pth")

    return {
        'name': exp_cfg['name'],
        'wm_dyn_acc': world_model_results.get('dyn_acc', 0.0),
        'im_dyn_acc': inverse_model_results.get('dyn_accuracy', 0.0)
    }

def print_final_summary(results):
    print(f"\n{'='*80}")
    print(f"{'Noise Level':<20} | {'WM Dyn Acc':<15} | {'IM Dyn Acc':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} | {r['wm_dyn_acc']:>15.4f} | {r['im_dyn_acc']:>15.4f}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Investigate Background Noise Impact')
    parser.add_argument('--skip_wm', action='store_true', help='Skip World Model training')
    parser.add_argument('--output', type=str, default='noise_impact_results.json')
    args = parser.parse_args()

    results_list = []
    
    for exp_cfg in NOISE_EXPERIMENTS:
        # 检查文件是否存在
        if not os.path.exists(exp_cfg['train']) or not os.path.exists(exp_cfg['test']):
            print(f"Warning: Skipping {exp_cfg['name']} because files were not found.")
            continue
            
        res = run_noise_experiment(exp_cfg)
        results_list.append(res)
    
    print_final_summary(results_list)
    
    with open(args.output, 'w') as f:
        json.dump(results_list, f, indent=2)

if __name__ == "__main__":
    main()
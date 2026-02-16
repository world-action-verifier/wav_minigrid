import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os
import copy
import csv
import json
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from asim_minigrid.models import WorldModel, SparseIDM
from asim_minigrid.dataset import MiniGridDynamicsDataset, PseudoLabeledSubset, NormalizedDataset, MemoryDynamicsDataset
from asim_minigrid.utils import freeze_model_for_active_learning
from asim_minigrid.al_utils import (
    set_all_seeds,
    query_strategy,
    compute_uncertainty_via_mcdropout,
    select_and_collect_consistency_data,
    evaluate,
    train_one_round,
)
from asim_minigrid.config import WM_ACTIVE_LEARNING, DEVICE

# Configuration
DATA_PATH = WM_ACTIVE_LEARNING["DATA_PATH"]
BASE_DATA_PATH = WM_ACTIVE_LEARNING["BASE_DATA_PATH"]
STAGE1_CKPT = WM_ACTIVE_LEARNING["STAGE1_CKPT"]
INVERSE_MODEL_PATH = WM_ACTIVE_LEARNING["INVERSE_MODEL_PATH"]
VIDEO_STAGE1_CKPT = WM_ACTIVE_LEARNING["VIDEO_STAGE1_CKPT"]
SAVE_DIR = os.path.join(project_root, "checkpoints/stage2_active_learning")
LOG_FILE = WM_ACTIVE_LEARNING["LOG_FILE"]
TEST_SET_PATH = WM_ACTIVE_LEARNING["TEST_SET_PATH"]
BATCH_SIZE = WM_ACTIVE_LEARNING["BATCH_SIZE"]
LR = WM_ACTIVE_LEARNING["LR"]
EPOCHS_FIRST_ROUND = WM_ACTIVE_LEARNING["EPOCHS_FIRST_ROUND"]
EPOCHS_PER_ROUND = WM_ACTIVE_LEARNING["EPOCHS_PER_ROUND"]
NUM_ROUNDS = WM_ACTIVE_LEARNING["NUM_ROUNDS"]
ADD_COUNT_FIRST_ROUND = WM_ACTIVE_LEARNING["ADD_COUNT_FIRST_ROUND"]
ADD_COUNT_PER_ROUND = WM_ACTIVE_LEARNING["ADD_COUNT_PER_ROUND"]
FORWARD_CARRIED_LOSS_WEIGHT = WM_ACTIVE_LEARNING["FORWARD_CARRIED_LOSS_WEIGHT"]
NUMS_SAMPLES_PER_STATE = WM_ACTIVE_LEARNING["NUMS_SAMPLES_PER_STATE"]
CONSISTENCY_MODE = WM_ACTIVE_LEARNING["CONSISTENCY_MODE"]
STRATEGIES = WM_ACTIVE_LEARNING["STRATEGIES"].copy()  # Make a copy to allow modification
USE_BASE_DATA = WM_ACTIVE_LEARNING["USE_BASE_DATA"]
USE_RANDOM_BASE_MODEL = WM_ACTIVE_LEARNING["USE_RANDOM_BASE_MODEL"]
TRAIN_FROM_SCRATCH = WM_ACTIVE_LEARNING["TRAIN_FROM_SCRATCH"]
SEED = WM_ACTIVE_LEARNING["SEED"]

# Allow environment variable overrides
if 'AL_STRATEGIES' in os.environ:
    STRATEGIES = os.environ['AL_STRATEGIES'].split(',')
if 'AL_SEED' in os.environ:
    SEED = int(os.environ['AL_SEED'])

set_all_seeds(SEED)

def load_stage1_model(model_path, obs_shape, num_actions):
    """Load stage-1 world model checkpoint."""
    print(f"Loading Stage 1 Model from: {model_path}")
    set_all_seeds(SEED)
    
    model = WorldModel(obs_shape, num_actions).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    missing_keys = checkpoint_keys - model_keys
    unexpected_keys = model_keys - checkpoint_keys
    if missing_keys:
        print(f"  [Warning] Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  [Warning] Unexpected keys: {len(unexpected_keys)}")
    
    return model

 

def run_active_learning():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    full_dataset = MiniGridDynamicsDataset(DATA_PATH)
    obs_shape = full_dataset.states.shape[1:]
    
    inverse_model = SparseIDM(num_actions=7).to(DEVICE)
    inverse_model.load_state_dict(torch.load(INVERSE_MODEL_PATH))
    
    dataset_size = len(full_dataset)
    all_permuted_indices = torch.randperm(dataset_size).tolist()
    pool_indices_full = all_permuted_indices
    print(f"Pool Size: {len(pool_indices_full)}")
    
    print(f"Loading test set from: {TEST_SET_PATH}")
    test_dataset = MiniGridDynamicsDataset(TEST_SET_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    train_pool_indices = pool_indices_full
    print(f"AL Pool Size: {len(train_pool_indices)} | Test Size: {len(test_dataset)}")
    
    with open(os.path.join(SAVE_DIR, LOG_FILE), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Strategy', 'Round', 'Labeled_Size', 'Test_MSE'])
    
    all_results = {}
    for strategy in STRATEGIES:
        print(f"\n=================== Strategy: {strategy} ===================")
        
        set_all_seeds(SEED)
        model = load_stage1_model(STAGE1_CKPT, obs_shape, num_actions=7)
        model.eval()
        
        pretrained_video_gen = None
        if strategy == "ASIM":
            pretrained_video_gen = load_stage1_model(VIDEO_STAGE1_CKPT, obs_shape, num_actions=7)
            pretrained_video_gen.eval()
        
        current_pool = copy.deepcopy(train_pool_indices)
        current_labeled = []
        pseudo_label_bank = {}
        collected_real_data = []
        prev_losses_map = None

        metrics = evaluate(
            model,
            test_loader,
            device=DEVICE,
            forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT,
            use_random_base_model=USE_RANDOM_BASE_MODEL,
            is_round_0=True,
        )
        print(f"[Round 0] MSE: {metrics['mse']:.4f}")
        
        all_results[strategy] = [{
            'round': 0,
            'labeled': len(current_labeled),
            'mse': metrics['mse'],
        }]
        
        with open(os.path.join(SAVE_DIR, LOG_FILE), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([strategy, 0, len(current_labeled), metrics['mse']])
        
        for round_idx in range(1, NUM_ROUNDS + 1):
            n_select = ADD_COUNT_FIRST_ROUND if round_idx == 1 else ADD_COUNT_PER_ROUND
            
            if strategy == "ASIM":
                selected_idx, new_data_list = select_and_collect_consistency_data(
                    video_gen_model=pretrained_video_gen,
                    current_world_model=model,
                    inverse_model=inverse_model,
                    dataset=full_dataset,
                    pool_indices=current_pool,
                    env_name="MiniGrid-Empty-Interact-6x6-o3-v0",
                    n_select=n_select,
                    device=DEVICE,
                    data_mode=CONSISTENCY_MODE, # "oracle", "model"
                    seed=SEED + round_idx,  
                    use_random_mix=False,
                    random_mix_ratio=0.3,
                )
                current_pool = list(set(current_pool) - set(selected_idx))
                
                collected_real_data.extend(new_data_list)
                print(f"Round {round_idx}: Collected {len(new_data_list)} new real interactions.")
                print(f"Round {round_idx}: Total collected data: {len(collected_real_data)} samples.")
                
                part_b = MemoryDynamicsDataset(collected_real_data)
                if USE_BASE_DATA:
                    part_a = MiniGridDynamicsDataset(BASE_DATA_PATH)
                    part_a_normalized = NormalizedDataset(part_a)
                    part_b_normalized = NormalizedDataset(part_b)
                    train_dataset = ConcatDataset([part_a_normalized, part_b_normalized])
                else:
                    train_dataset = part_b
                
                current_labeled.extend(selected_idx)
                
            else:
                pool_before = len(current_pool)
                labeled_before = len(current_labeled)
                selected_idx, selected_pseudo_actions, current_loss_map = query_strategy(
                    strategy,
                    model,
                    full_dataset,
                    current_pool,
                    n_select,
                    device=DEVICE,
                    seed=SEED,
                    batch_size=BATCH_SIZE,
                    forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT,
                    compute_uncertainty_via_mcdropout_fn=compute_uncertainty_via_mcdropout,
                    round_idx=round_idx,
                    prev_losses_map=prev_losses_map,
                )
                pool_after = pool_before - len(selected_idx)
                
                if strategy == "Progress":
                    prev_losses_map = current_loss_map
                
                current_labeled.extend(selected_idx)
                selected_set = set(selected_idx)
                current_pool = [idx for idx in current_pool if idx not in selected_set]
                if selected_pseudo_actions:
                    pseudo_label_bank.update(selected_pseudo_actions)
                
                print(f"Round {round_idx}: Selected {len(selected_idx)}. Labeled Total: {len(current_labeled)}")
                
                part_b = PseudoLabeledSubset(full_dataset, current_labeled, {})
                if USE_BASE_DATA:
                    part_a = MiniGridDynamicsDataset(BASE_DATA_PATH)
                    part_a_normalized = NormalizedDataset(part_a)
                    part_b_normalized = NormalizedDataset(part_b)
                    train_dataset = ConcatDataset([part_a_normalized, part_b_normalized])
                else:
                    train_dataset = part_b
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            print(f"Round {round_idx}: Training dataset size: {len(train_dataset)}")
            
            train_one_round(
                model,
                train_loader,
                device=DEVICE,
                epochs=EPOCHS_PER_ROUND if round_idx > 1 else EPOCHS_FIRST_ROUND,
                lr=LR,
                forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT,
                train_from_scratch=TRAIN_FROM_SCRATCH,
                freeze_model_for_active_learning_fn=freeze_model_for_active_learning,
            )
            
            metrics = evaluate(
                model,
                test_loader,
                device=DEVICE,
                forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT,
                use_random_base_model=USE_RANDOM_BASE_MODEL,
            )
            print(f"Round {round_idx} Eval: MSE={metrics['mse']:.5f}")
            
            with open(os.path.join(SAVE_DIR, LOG_FILE), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([strategy, round_idx, len(current_labeled), metrics['mse']])
            
            all_results[strategy].append({
                'round': round_idx,
                'labeled': len(current_labeled),
                'mse': metrics['mse'],
            })

    print("\n============= All Strategies Summary =============")
    for strategy, results in all_results.items():
        print(f"\n>>> Strategy: {strategy}")
        for r in results:
            print(
                f"  Round {r['round']:>2d} | "
                f"Labeled {r['labeled']:>5d} | "
                f"MSE {r['mse']:.5f}"
            )

if __name__ == "__main__":
    run_active_learning()


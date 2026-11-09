"""
Configuration file for training and evaluation experiments.
All hyperparameters and paths are defined here.
"""

import os
import torch

# Common paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MINIGRID_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# IDM Comparison Experiment Configuration
# ============================================================================
IDM_COMPARISON = {
    "TRAIN_DATA_PATH": os.path.join(
        MINIGRID_DIR, 
        "data", 
        "MiniGrid-Empty-Interact-6x6-o6-3-noise-v0_train_few_sample.npz"
    ),
    "TEST_DATA_PATH": os.path.join(
        MINIGRID_DIR, 
        "data", 
        "MiniGrid-Empty-Interact-6x6-o6-3-noise-v0_test_few_sample.npz"
    ),
    "BATCH_SIZE": 64,
    "LR": 1e-3,
    "EPOCHS": 500,
    "NUM_ACTIONS": 7,
    "SEED": 42,
}

# ============================================================================
# Data Efficiency Gap Experiment Configuration
# ============================================================================
DATA_EFFICIENCY_GAP = {
    "DATA_PATH": os.path.join(
        MINIGRID_DIR, 
        "data", 
        "MiniGrid-Empty-Interact-6x6-o6-v0_train.npz"
    ),
    "TEST_SET_PATH": os.path.join(
        MINIGRID_DIR, 
        "data", 
        "MiniGrid-Empty-Interact-6x6-o6-v0_test.npz"
    ),
    "VIDEO_STAGE1_CKPT": os.path.join(
        MINIGRID_DIR, 
        "checkpoints", 
        "pretrained_video_model.pth"
    ),
    "BATCH_SIZE": 64,
    "LR": 1e-3,
    "EPOCHS": 500,
    "INVERSE_MODEL_EPOCHS": 300,
    "FORWARD_CARRIED_LOSS_WEIGHT": 10.0,
    "TARGET_PIXELS": 4.0,
    "MASK_L1_LAMBDA": 0.1,
    "ADDED": 400 / 2000,
    "TRAIN_RATIOS": None,  # Will be computed as [ADDED*1, ADDED*2, ADDED*3, ADDED*4, ADDED*5]
    "SEED": 42,
}

# Compute TRAIN_RATIOS based on ADDED
DATA_EFFICIENCY_GAP["TRAIN_RATIOS"] = [
    DATA_EFFICIENCY_GAP["ADDED"] * i 
    for i in range(1, 6)
]

# ============================================================================
# State Complexity Gap Experiment Configuration
# ============================================================================
STATE_COMPLEXITY_GAP = {
    "DEFAULT_TEST_SETS": [
        os.path.join(MINIGRID_DIR, "data", "MiniGrid-Empty-Interact-6x6-o6-v0_test.npz"),
        os.path.join(MINIGRID_DIR, "data", "MiniGrid-Empty-Interact-6x6-o8-v0_test.npz"),
        os.path.join(MINIGRID_DIR, "data", "MiniGrid-Empty-Interact-6x6-o10-v0_test.npz"),
        os.path.join(MINIGRID_DIR, "data", "MiniGrid-Empty-Interact-6x6-o12-v0_test.npz"),
        os.path.join(MINIGRID_DIR, "data", "MiniGrid-Empty-Interact-6x6-o14-v0_test.npz"),
    ],
    "DEFAULT_WORLD_MODEL_PATH": os.path.join(
        MINIGRID_DIR, 
        "checkpoints", 
        "pretrained_world_model_for_state_complexity.pth"
    ),
    "DEFAULT_INVERSE_MODEL_PATH": os.path.join(
        MINIGRID_DIR, 
        "checkpoints", 
        "pretrained_inverse_model_for_state_complexity.pth"
    ),
    "DEFAULT_BATCH_SIZE": 64,
    "DEFAULT_FORWARD_CARRIED_LOSS_WEIGHT": 10.0,
}

# ============================================================================
# Noise Robustness Experiment Configuration
# ============================================================================
NOISE_ROBUSTNESS = {
    "EXPERIMENTS": [
        {
            "name": "No Noise",
            "train": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-v0_train.npz",
            ),
            "test": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-v0_test.npz",
            ),
        },
        {
            "name": "Noise-1",
            "train": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-1-noise-v0_train.npz",
            ),
            "test": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-v0_test.npz",
            ),
        },
        {
            "name": "Noise-2",
            "train": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-2-noise-v0_train.npz",
            ),
            "test": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-v0_test.npz",
            ),
        },
        {
            "name": "Noise-3",
            "train": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-3-noise-v0_train.npz",
            ),
            "test": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-v0_test.npz",
            ),
        },
        {
            "name": "Noise-4",
            "train": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-4-noise-v0_train.npz",
            ),
            "test": os.path.join(
                MINIGRID_DIR,
                "data",
                "MiniGrid-Empty-Interact-6x6-o6-v0_test.npz",
            ),
        },
    ],
    "SAVE_MODEL": False,
    "VIDEO_STAGE1_CKPT": os.path.join(
        MINIGRID_DIR, 
        "checkpoints", 
        "pretrained_video_model.pth"
    ),
    "BATCH_SIZE": 64,
    "LR": 1e-3,
    "EPOCHS": 300,
    "INVERSE_MODEL_EPOCHS": 300,
    "FORWARD_CARRIED_LOSS_WEIGHT": 10.0,
    "SEED": 42,
}

# ============================================================================
# World Model Active Learning Experiment Configuration
# ============================================================================
WM_ACTIVE_LEARNING = {
    "DATA_PATH": os.path.join(
        MINIGRID_DIR, 
        "data", 
        "MiniGrid-Empty-Interact-6x6-o3-v0_MultiTaskPolicy_data_pool.npz"
    ),
    "BASE_DATA_PATH": os.path.join(
        MINIGRID_DIR, 
        "data", 
        "random_selected_data.npz" # random selected data from the data pool to train a base World Model and IDM
    ),
    "STAGE1_CKPT": os.path.join(
        MINIGRID_DIR, 
        "checkpoints", 
        "pretrained_base_world_model.pth" # initialized from the Pre-trained Video Model; trained from the random selected data
    ),
    "INVERSE_MODEL_PATH": os.path.join(
        MINIGRID_DIR, 
        "checkpoints", 
        "pretrained_inverse_model_for_active_learning.pth"
    ), # trained from the random selected data
    "VIDEO_STAGE1_CKPT": os.path.join(
        MINIGRID_DIR, 
        "checkpoints", 
        "pretrained_video_model.pth"
    ),
    "SAVE_DIR": os.path.join(MINIGRID_DIR, "outputs", "active_learning_balanced"),
    "LOG_FILE": "active_learning_results.csv",
    "TEST_SET_PATH": os.path.join(
        MINIGRID_DIR, 
        "data", 
        "MiniGrid-Empty-Interact-6x6-o3-v0_MultiTaskPolicy_policy_test_set_for_active_learning.npz"
    ),
    "BATCH_SIZE": 64,
    "LR": 1e-4,
    "EPOCHS_FIRST_ROUND": 300,
    "EPOCHS_PER_ROUND": 300,
    "INVERSE_MODEL_EPOCHS_PER_ROUND": 50,
    "NUM_ROUNDS": 5,  # Aggregate rounds 1–5; round 0 is only the initial evaluation.
    "ADD_COUNT_FIRST_ROUND": 100,
    "ADD_COUNT_PER_ROUND": 100,
    "FORWARD_CARRIED_LOSS_WEIGHT": 10.0,
    "NUMS_SAMPLES_PER_STATE": 1,
    "CONSISTENCY_MODE": "oracle",  # "oracle" or "model"
    "STRATEGIES": ["Random", "Hard-Oracle", "Uncertainty", "Progress", "RLIR", "WAV"],
    "PROGRESS_GAMMA": 0.1,
    "PROGRESS_RANDOM_MIX_RATIO": 0.3,
    "ORACLE_RANDOM_MIX_RATIO": 0.3,
    "UNCERTAINTY_RANDOM_MIX_RATIO": 0.3,
    "UNCERTAINTY_TEMPERATURE": 0.5,
    "UNCERTAINTY_USE_TOPK": True,
    "UNCERTAINTY_N_SAMPLES": 5,
    "RLIR_ACTION_LOSS": "cross_entropy",  # cross_entropy | one_minus_probability | margin
    "RLIR_SCORE_NORMALIZATION": "action_zscore",  # none | action_zscore | action_percentile
    "RLIR_RANDOM_MIX_RATIO": 0.3,
    "RLIR_TEMPERATURE": 0.5,  # used only when RLIR_USE_TOPK=False
    "RLIR_USE_TOPK": True,
    "RLIR_ROUND_CARRIED": True,
    # Match the balanced-test experiment: update RLIR's IDM on cumulative labels.
    "RLIR_UPDATE_IDM": True,
    "USE_BASE_DATA": True,
    "USE_RANDOM_BASE_MODEL": True,
    "TRAIN_FROM_SCRATCH": False,
    "SEED": 52,  # Default for a single imported run / AL_SEED override.
    "SEEDS": list(range(52, 57)),  # Five paired seeds shared by all acquisition strategies.
    "CPU_THREADS": 1,
}

# ============================================================================
# Multi-round AFS benchmark (exps/afs_active_learning.py)
# ============================================================================
# pool_p05.npz: 12,000 transitions, 5% from AFS-eligible states.
# confirmation_states.npz: 174 evaluation states, actions 3/5/6.
WM_AFS_ACTIVE_LEARNING = {
    **{key: value for key, value in WM_ACTIVE_LEARNING.items() if key != "TEST_SET_PATH"},
    "DATA_PATH": os.path.join(MINIGRID_DIR, "data", "pool_p05.npz"),
    "AFS_STATES_PATH": os.path.join(MINIGRID_DIR, "data", "confirmation_states.npz"),
    "AFS_ACTIONS": [3, 5, 6],
    "SAVE_DIR": os.path.join(MINIGRID_DIR, "outputs", "afs_active_learning"),
    "NUM_ROUNDS": 9,
    "ADD_COUNT_FIRST_ROUND": 30,
    "ADD_COUNT_PER_ROUND": 30,
    "EPOCHS_FIRST_ROUND": 300,
    "EPOCHS_PER_ROUND": 300,
    "INVERSE_MODEL_EPOCHS_PER_ROUND": 50,
    "RLIR_UPDATE_IDM": True,
    "STRATEGIES": ["Random", "Hard-Oracle", "Uncertainty", "Progress", "RLIR-updated-IDM", "WAV"],
    "SEED": 201,
    "SEEDS": list(range(201, 206)),
    "DEVICE": str(DEVICE),
}


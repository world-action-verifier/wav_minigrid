"""
Configuration file for training scripts.
All hyperparameters and paths are defined here.
"""

import os
import torch

# Common paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MINIGRID_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Common Training Configuration
# ============================================================================
BATCH_SIZE = 64
LR = 1e-4
SEED = 42
NUM_ACTIONS = 7

# ============================================================================
# Video Model Training Configuration (train_vp.py)
# ============================================================================
VIDEO_TRAINING = {
    "DATA_PATH": os.path.join(
        MINIGRID_DIR,
        "data",
        "MiniGrid-Empty-Interact-6x6-o3-v0_video_pretraing.npz"
    ),
    "FORWARD_CARRIED_LOSS_WEIGHT": 10.0,
    "PRIOR_WEIGHT": 5.0,
    "VQ_LOSS_WEIGHT": 0,
    "AUX_ACTION_WEIGHT": 0,
    "CONSISTENCY_WEIGHT": 2.0,
}

# ============================================================================
# World Model Fine-tuning Configuration (train_wm.py)
# ============================================================================
WM_FINETUNING = {
    "VIDEO_STAGE1_CKPT": os.path.join(
        MINIGRID_DIR,
        "checkpoints",
        "pretrained_video_model.pth"
    ),
    "BASE_DATA_PATH": os.path.join(
        MINIGRID_DIR,
        "data",
        "random_selected_data.npz"  # random selected data from the data pool to train a base World Model and IDM
    ),
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "EPOCHS_FIRST_ROUND": 300,
    "FORWARD_CARRIED_LOSS_WEIGHT": 10.0,
    "TRAIN_FROM_SCRATCH": False,
    "SEED": SEED,
}

# ============================================================================
# IDM Training Configuration (train_idm.py)
# ============================================================================
IDM_TRAINING = {
    "BASE_DATA_PATH": os.path.join(
        MINIGRID_DIR,
        "data",
        "random_selected_data.npz"  # random selected data from the data pool to train a base World Model and IDM
    ),
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "EPOCHS_FIRST_ROUND": 300,
    "NUM_ACTIONS": NUM_ACTIONS,
    "SEED": SEED,
}


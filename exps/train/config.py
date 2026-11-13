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
# Video Model Training
# ============================================================================

VIDEO_DATA_DIR = os.environ.get(
    "VIDEO_ACTIONFREE_DATA_DIR", os.path.join(MINIGRID_DIR, "data")
)
VIDEO_TRAINING = {
    "DATA_PATH": os.path.join(VIDEO_DATA_DIR, "MiniGrid-Empty-Interact-6x6-o3-v0_video_pretraing.npz"),
    "SEED": 601,
    "NUM_LATENT_MODES": 7,  # Anonymous codes, not actions.
    "EPOCHS": 300,
    "STEPS_PER_EPOCH": 128,
    "GROUP_BATCH_SIZE": 48,
    "PAIR_BATCH_SIZE": 75,
    "LR": 1e-3,
    "WARMUP_EPOCHS": 10,  # Learning-rate warm-up; continuous cosine to 0.05x.
    "PRIOR_WARMUP_START": 20,  # Prior + pair reconstruction: weight 0 until epoch20.
    "PRIOR_WARMUP_END": 50,  # Linear ramp to weight 1 at epoch50.
    "CHANGED_PIXEL_WEIGHT": 5.0,
    "FORWARD_CARRIED_LOSS_WEIGHT": 10.0,
    "CONSISTENCY_WEIGHT": 2.0,
    "PRIOR_TEMPERATURE": 0.5,
    "UNMATCHED_WEIGHT": 0.1,
    "PAIR_WEIGHT": 0.5,
    "GRAD_CLIP": 10.0,
    "SAVE_FREQ": 20,
    "SAVE_DIR": os.path.join(MINIGRID_DIR, "checkpoints", "video_model"),
}
# Fresh model, one Adam optimizer throughout. Original carried regression head.
# Dropout0; posterior encoder and downstream action adapters frozen/unused.
# Reconstruction directly trains latent codes; no true-action CE or VQ objective.

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
    "EPOCHS_FIRST_ROUND": 1000,
    "NUM_ACTIONS": NUM_ACTIONS,
    "SEED": SEED,
}

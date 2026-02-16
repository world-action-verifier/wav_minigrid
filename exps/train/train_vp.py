#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
import argparse
from tqdm import tqdm
import wandb
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from asim_minigrid.models.wm import WorldModel
from train_utils import (
    compute_loss_vp, 
    evaluate, 
    get_dataloaders,
    prepare_batch_inputs,
    load_model_checkpoint,
    create_warmup_cosine_scheduler,
)
from config import DEVICE, VIDEO_TRAINING

print(f"Using device: {DEVICE}")

DATA_PATH = VIDEO_TRAINING["DATA_PATH"]
FORWARD_CARRIED_LOSS_WEIGHT = VIDEO_TRAINING["FORWARD_CARRIED_LOSS_WEIGHT"]
PRIOR_WEIGHT = VIDEO_TRAINING["PRIOR_WEIGHT"]
VQ_LOSS_WEIGHT = VIDEO_TRAINING["VQ_LOSS_WEIGHT"]
AUX_ACTION_WEIGHT = VIDEO_TRAINING["AUX_ACTION_WEIGHT"]
CONSISTENCY_WEIGHT = VIDEO_TRAINING["CONSISTENCY_WEIGHT"]

def train_stage1(args):
    train_loader, test_loader, obs_shape = get_dataloaders(
        args.data_path, 
        batch_size=args.batch_size, 
        split_ratio=0.5,
        seed=42
    )
    os.makedirs(args.save_dir, exist_ok=True)
    
    model = WorldModel(observation_shape=obs_shape, num_actions=7).to(DEVICE)
    if getattr(args, "pretrained_path", None) is not None and os.path.isfile(args.pretrained_path):
        model = load_model_checkpoint(model, args.pretrained_path, DEVICE, strict=False)
        print("Pretrained weights loaded successfully")
    else:
        print("No pretrained weights provided")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = create_warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)

    log_path = os.path.join(args.save_dir, 'training_log_stage1.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_loss'])

    print(f"Starting Stage 1 Training (No Action Condition)...")
    
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project if hasattr(args, 'wandb_project') else "world-model-training",
            name=args.wandb_name if hasattr(args, 'wandb_name') else f"pretrained_video_model",
            config={
                'forward_carried_loss_weight': FORWARD_CARRIED_LOSS_WEIGHT,
                'prior_weight': PRIOR_WEIGHT,
                'vq_loss_weight': VQ_LOSS_WEIGHT,
                'aux_action_weight': AUX_ACTION_WEIGHT,
                'consistency_weight': CONSISTENCY_WEIGHT,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'epochs': args.epochs,
                'eval_freq': args.eval_freq,
            }
        )
    
    best_test_loss = float('inf')
    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    criterion_mse = nn.MSELoss()

    for epoch in tqdm(range(args.epochs), desc="Training"):
        model.train()
        train_loss_total = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs, _ = prepare_batch_inputs(batch, DEVICE)

            optimizer.zero_grad()
            next_inputs = {k: v[1] for k, v in inputs.items()}
            pred_out = model(inputs, next_obs_inputs=next_inputs, mode='posterior', gt_actions=batch['action'].to(DEVICE))
            
            loss, _ = compute_loss_vp(
                pred_out,
                inputs,
                criterion_ce,
                criterion_mse,
                aux_action_weight=AUX_ACTION_WEIGHT,
                consistency_weight=CONSISTENCY_WEIGHT,
                carried_weight=FORWARD_CARRIED_LOSS_WEIGHT,
                vq_loss_weight=VQ_LOSS_WEIGHT,
                prior_weight=PRIOR_WEIGHT,
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
        
        avg_train_loss = train_loss_total / len(train_loader)
        scheduler.step()
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_train_loss,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
            }, step=epoch + 1)
        
        if (epoch + 1) % args.eval_freq == 0:
            eval_metrics = evaluate(
                model,
                test_loader,
                device=DEVICE,
                forward_carried_loss_weight=FORWARD_CARRIED_LOSS_WEIGHT,
                use_random_base_model=False,
                is_round_0=False,
            )
            test_loss = eval_metrics['mse']
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}")
            
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, test_loss])
            
            if args.use_wandb:
                wandb.log({'test/loss': test_loss}, step=epoch + 1)
                
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_path = os.path.join(args.save_dir, f'pretrained_video_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")

    if args.use_wandb:
        wandb.finish()
    
    print("Stage 1 Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to .npz data file")
    parser.add_argument("--save_dir", type=str, default="checkpoints_stage1", help="Directory to save models/logs")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs for learning rate scheduling")
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--use_wandb", action='store_true', help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="world-model-training", help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    
    args = parser.parse_args()
    train_stage1(args)
#!/usr/bin/env python3
"""
Train the video model from state transitions without action labels.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wav_minigrid.models import WorldModel
try:
    from .config import DEVICE, VIDEO_TRAINING
    from .train_utils import (set_all_seeds, load_video_pairs, video_observations,
                              video_all_codes, video_matching_loss, video_pair_losses)
except ImportError:
    from config import DEVICE, VIDEO_TRAINING
    from train_utils import (set_all_seeds, load_video_pairs, video_observations,
                            video_all_codes, video_matching_loss, video_pair_losses)


def train_video(config, root, device, smoke=False):
    root.mkdir(parents=True, exist_ok=False)
    set_all_seeds(config["SEED"])
    model = WorldModel((6, 6, 3), config["NUM_LATENT_MODES"]).to(device)
    model.decoder_shared[2].p = 0.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(("finetune_", "vq_encoder_net")))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config["LR"])

    def lr_factor(epoch):
        warmup = config["WARMUP_EPOCHS"]
        if epoch < warmup:
            return .1 + .9 * epoch / warmup
        progress = min(1., (epoch - warmup) / max(1, config["EPOCHS"] - warmup))
        return .05 + .95 * .5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    sampler = torch.Generator().manual_seed(config["SEED"] + 100000)
    groups = load_video_pairs(config["DATA_PATH"], device, role="groups")
    pairs = load_video_pairs(config["DATA_PATH"], device, role="policy")
    record = dict(config=config, smoke=smoke, architecture="original_WorldModel",
                  supervision="transition-only; no action labels", training_mode="single_stage_warmup",
                  data_sha256=hashlib.sha256(Path(config["DATA_PATH"]).read_bytes()).hexdigest())
    (root / "config.json").write_text(json.dumps(record, indent=2) + "\n")
    gb, pb = config["GROUP_BATCH_SIZE"], config["PAIR_BATCH_SIZE"]

    for epoch in range(1, config["EPOCHS"] + 1):
        model.train()
        start, end = config["PRIOR_WARMUP_START"], config["PRIOR_WARMUP_END"]
        scale = min(1., max(0., (epoch - start) / (end - start)))
        statistics = []
        lr = optimizer.param_groups[0]["lr"]
        for _ in range(config["STEPS_PER_EPOCH"]):
            gi = torch.randint(len(groups[0]), (gb,), generator=sampler).to(device)
            pi = torch.randint(len(pairs["states"]), (pb,), generator=sampler).to(device)
            permutation = torch.rand(gb, model.num_actions, generator=sampler).argsort(1).to(device)
            states, carried, targets = [v[gi] for v in groups]
            targets = targets.gather(1, permutation[:, :, None].expand(-1, -1, 110))
            prediction = video_all_codes(model, video_observations(states, carried))
            matching = video_matching_loss(prediction, states, targets, config)
            batch = {k: v[pi] for k, v in pairs.items()}
            prior, reconstruction, exact = video_pair_losses(model, batch, config)
            loss = matching + scale * (prior + config["PAIR_WEIGHT"] * reconstruction)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["GRAD_CLIP"])
            optimizer.step()
            statistics.append([float(v.detach()) for v in (loss, matching, prior, reconstruction, exact)])
        scheduler.step()
        stats = np.mean(statistics, axis=0)
        row = dict(epoch=epoch, step=epoch * config["STEPS_PER_EPOCH"], loss=float(stats[0]),
                   matching=float(stats[1]), prior=float(stats[2]), pair_reconstruction=float(stats[3]),
                   exact_pair_fraction=float(stats[4]), prior_weight=scale, lr=lr)
        with (root / "training.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if epoch == 1 or epoch % config["SAVE_FREQ"] == 0 or epoch == config["EPOCHS"]:
            nr = np.random.get_state()
            checkpoint = root / f"epoch{epoch:03d}.pth"
            torch.save(dict(model_state_dict=model.state_dict(), **record, epoch=epoch,
                            total_steps=row["step"],
                            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                            sampler_state=sampler.get_state(), resume_supported=False,
                            rng_state=dict(python=random.getstate(), numpy=(nr[0], nr[1].tolist(), *nr[2:]),
                                           torch=torch.get_rng_state(),
                                           cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])), checkpoint)
    # Same 40 video tensors as the legacy format; initialize 8 action adapters separately.
    video_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("finetune_")}
    torch.save(dict(model_state_dict=video_state, **record, training_checkpoint=str(checkpoint)),
               root / "video_actionfree_original_arch.pth")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--device", default=str(DEVICE))
    parser.add_argument("--smoke", action="store_true", help="Two tiny epochs exercising both loss weights 0 and 1.")
    args = parser.parse_args()
    config = copy.deepcopy(VIDEO_TRAINING)
    if args.smoke:
        config.update(EPOCHS=2, STEPS_PER_EPOCH=2, GROUP_BATCH_SIZE=4, PAIR_BATCH_SIZE=4,
                      PRIOR_WARMUP_START=1, PRIOR_WARMUP_END=2)
    root = args.save_dir or Path(config["SAVE_DIR"] + ("_smoke" if args.smoke else ""))
    config["SAVE_DIR"] = str(root)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train_video(config, root, torch.device(args.device), args.smoke)
    print(f"Finished: {root}")


if __name__ == "__main__":
    main()

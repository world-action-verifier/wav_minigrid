"""
Action Following Score (AFS) evaluation.

Note: The AFS benchmark uses a dedicated active-learning setup that is separate
from the prediction-error benchmark reported in the paper. All methods are
compared using models obtained from the same active-learning round within the
AFS benchmark.

This file reproduces the AFS metric used in the paper.
"""
import argparse
import json
import os
import re
from itertools import combinations

import numpy as np
import torch

from wav_minigrid.config import DEVICE, WM_AFS_EVAL
from wav_minigrid.evaluate_generation import MiniGridPhysicsOracle
from wav_minigrid.models import WorldModel

ALL_ACTIONS = list(range(7))


# ---------------------------------------------------------------------------
# CLI (defaults from wav_minigrid.config.WM_AFS_EVAL)
# ---------------------------------------------------------------------------

def parse_args():
    cfg = WM_AFS_EVAL
    parser = argparse.ArgumentParser(
        description="Evaluate world model action following score (AFS)."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=cfg["CHECKPOINT_DIR"],
        help="Directory containing world model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        default=cfg["CHECKPOINT_PATTERN"],
        help="Regex used to filter checkpoint filenames.",
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        default=cfg["TEST_SET_PATH"],
        help="Path to test .npz file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=cfg["NUM_SAMPLES"],
        help="Number of initial states sampled from the test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg["SEED"],
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(DEVICE),
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=cfg["OUTPUT_JSON"],
        help="Output path for results JSON.",
    )
    parser.add_argument(
        "--action_mode",
        type=str,
        default=cfg["ACTION_MODE"],
        choices=["fixed_interact", "oracle_effective"],
        help=(
            "fixed_interact: evaluate on the fixed action list given by --actions. "
            "oracle_effective: evaluate on actions that actually change the oracle state."
        ),
    )
    parser.add_argument(
        "--actions",
        type=str,
        default=cfg["ACTIONS"],
        help="Comma-separated action ids to evaluate (used when action_mode=fixed_interact).",
    )
    parser.add_argument(
        "--required_executable_actions",
        type=str,
        default=cfg["REQUIRED_EXECUTABLE_ACTIONS"],
        help=(
            "When action_mode=fixed_interact, initial states are pre-filtered so that "
            "every action in this list actually changes the oracle state. "
            "Comma-separated ids."
        ),
    )
    parser.add_argument(
        "--action_following_tol",
        type=float,
        default=cfg["ACTION_FOLLOWING_TOL"],
        help=(
            "Per-channel tolerance for AFS mask: prediction matches GT if "
            "|pred - gt| <= tol. (0 means exact match on discrete predictions.)"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_action_list(text: str):
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_test_data(test_set_path: str):
    data = np.load(test_set_path)
    states = data["states"]
    while states.ndim > 4:
        states = states.squeeze(1)
    carried = data["carried"]  # [N, 2]
    return states.astype(np.int64), carried.astype(np.int64)


def sample_initial_states(states, carried, num_samples: int, seed: int):
    n = states.shape[0]
    k = min(n, num_samples)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    return states[indices], carried[indices], indices


def state_or_carried_changed(curr_state, curr_carried, next_state, next_carried) -> bool:
    grid_changed = np.any(curr_state != next_state)
    carried_changed = (curr_carried[0] != next_carried[0]) or (curr_carried[1] != next_carried[1])
    return bool(grid_changed or carried_changed)


def get_effective_actions(oracle, curr_state, curr_carried):
    """Return actions that actually change the oracle state or carried."""
    effective = []
    for action in ALL_ACTIONS:
        s_next, c_next_col, c_next_obj = oracle.step(
            curr_state.copy(), int(curr_carried[0]), int(curr_carried[1]), int(action),
        )
        c_next = np.array([c_next_col, c_next_obj], dtype=np.int64)
        if state_or_carried_changed(curr_state, curr_carried, s_next.astype(np.int64), c_next):
            effective.append(int(action))
    return effective


def filter_initial_states_by_executable_actions(states, carried, required_actions):
    """Keep only states where every action in required_actions is executable."""
    if not required_actions:
        return np.arange(states.shape[0], dtype=np.int64)
    oracle = MiniGridPhysicsOracle()
    keep = []
    for i in range(states.shape[0]):
        s0, c0 = states[i], carried[i]
        ok = True
        for action in required_actions:
            s_next, c_next_col, c_next_obj = oracle.step(
                s0.copy(), int(c0[0]), int(c0[1]), int(action),
            )
            c_next = np.array([c_next_col, c_next_obj], dtype=np.int64)
            if not state_or_carried_changed(s0, c0, s_next.astype(np.int64), c_next):
                ok = False
                break
        if ok:
            keep.append(i)
    return np.array(keep, dtype=np.int64)


def list_checkpoints(checkpoint_dir: str, pattern: str):
    regex = re.compile(pattern)
    matched = [
        os.path.join(checkpoint_dir, f)
        for f in sorted(os.listdir(checkpoint_dir))
        if regex.fullmatch(f)
    ]
    if not matched:
        raise FileNotFoundError(
            f"No checkpoint matched pattern `{pattern}` in `{checkpoint_dir}`."
        )
    return matched


def load_world_model(checkpoint_path: str, obs_shape, device):
    model = WorldModel(observation_shape=obs_shape, num_actions=7).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    name = os.path.basename(checkpoint_path)
    if incompatible.missing_keys:
        print(f"[WARN] {name}: missing keys {incompatible.missing_keys[:5]} ...")
    if incompatible.unexpected_keys:
        print(f"[WARN] {name}: unexpected keys {incompatible.unexpected_keys[:5]} ...")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_with_action(model, curr_state_np, curr_carried_np, action, device):
    frame_t = torch.from_numpy(curr_state_np).unsqueeze(0).to(device).float()
    c_col_t = torch.tensor(
        [[curr_carried_np[0]]], dtype=torch.float32, device=device
    ).unsqueeze(0)  # [T=1, B=1, 1]
    c_obj_t = torch.tensor(
        [[curr_carried_np[1]]], dtype=torch.float32, device=device
    ).unsqueeze(0)
    inputs = {
        "frame": frame_t.unsqueeze(0),  # [T=1, B=1, H, W, C]
        "carried_col": c_col_t,
        "carried_obj": c_obj_t,
    }
    gt_actions = torch.tensor([action], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(inputs, mode="predict_with_action", gt_actions=gt_actions)
    pred_obj = torch.argmax(pred["logits_obj"], dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    pred_col = torch.argmax(pred["logits_col"], dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    pred_state = torch.argmax(pred["logits_state"], dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    pred_grid = np.stack([pred_obj, pred_col, pred_state], axis=-1)
    pred_c_col = int(torch.round(pred["carried_col"]).squeeze().item())
    pred_c_obj = int(torch.round(pred["carried_obj"]).squeeze().item())
    return pred_grid, np.array([pred_c_col, pred_c_obj], dtype=np.int64)


# ---------------------------------------------------------------------------
# Action Following Score (AFS) computation
# ---------------------------------------------------------------------------

def _channel_match(pred_val, gt_val, tol: float) -> bool:
    """Scalar channel match: exact if tol <= 0, else L1 within tol."""
    if tol <= 0.0:
        return bool(pred_val == gt_val)
    return abs(float(pred_val) - float(gt_val)) <= tol + 1e-12


def pairwise_gt_channel_diversity_total(gt_states, gt_carried) -> int:
    """
    Denominator of AFS: sum over unordered action pairs of per-channel Hamming
    distance between oracle next-states (grid H×W×3 + carried length-2).
    Each differing channel counts 1.
    """
    k = len(gt_states)
    if k < 2:
        return 0
    total = 0
    for a, b in combinations(range(k), 2):
        total += int(np.sum(gt_states[a] != gt_states[b]))
        total += int(gt_carried[a][0] != gt_carried[b][0])
        total += int(gt_carried[a][1] != gt_carried[b][1])
    return total


def pairwise_wm_channel_diversity_masked_total(
    pred_states, pred_carried, gt_states, gt_carried, tol: float
) -> int:
    """
    Numerator of AFS: for each unordered pair (a, b), count grid+carried channels
    where WM predictions differ, but only on channels where pred_a matches gt_a
    AND pred_b matches gt_b (within tol). This ensures only correctly-predicted
    channels contribute to the following score.
    """
    k = len(pred_states)
    if k < 2:
        return 0
    total = 0
    for a, b in combinations(range(k), 2):
        sa, sb = pred_states[a], pred_states[b]
        ga, gb = gt_states[a], gt_states[b]
        ca, cb = pred_carried[a], pred_carried[b]
        gta, gtb = gt_carried[a], gt_carried[b]
        if tol <= 0.0:
            match_a = sa == ga
            match_b = sb == gb
            pred_diff = sa != sb
        else:
            sa_f, sb_f = sa.astype(np.float64), sb.astype(np.float64)
            ga_f, gb_f = ga.astype(np.float64), gb.astype(np.float64)
            match_a = np.abs(sa_f - ga_f) <= tol
            match_b = np.abs(sb_f - gb_f) <= tol
            pred_diff = np.abs(sa_f - sb_f) > tol
        total += int((match_a & match_b & pred_diff).sum())
        for idx in (0, 1):
            if not (_channel_match(ca[idx], gta[idx], tol) and _channel_match(cb[idx], gtb[idx], tol)):
                continue
            if tol <= 0.0:
                carried_diff = ca[idx] != cb[idx]
            else:
                carried_diff = abs(float(ca[idx]) - float(cb[idx])) > tol + 1e-12
            if carried_diff:
                total += 1
    return total


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_one_model(
    model, init_states, init_carried, actions, action_mode: str,
    action_following_tol: float = 0.0,
):
    """Compute action following score (AFS) aggregated over sampled initial states."""
    oracle = MiniGridPhysicsOracle()
    device = next(model.parameters()).device
    n = init_states.shape[0]
    skipped = 0

    afs_gt_channel_total = 0
    afs_wm_masked_channel_total = 0

    for i in range(n):
        s0, c0 = init_states[i], init_carried[i]

        if action_mode == "fixed_interact":
            eval_actions = actions
        else:
            eval_actions = get_effective_actions(oracle, s0, c0)

        if len(eval_actions) < 2:
            skipped += 1
            continue

        pred_states, pred_carried_list = [], []
        gt_states, gt_carried_list = [], []

        for action in eval_actions:
            s_pred, c_pred = predict_with_action(model, s0, c0, action, device)
            s_gt, c_gt_col, c_gt_obj = oracle.step(
                s0.copy(), int(c0[0]), int(c0[1]), action,
            )
            c_gt = np.array([c_gt_col, c_gt_obj], dtype=np.int64)
            pred_states.append(s_pred)
            pred_carried_list.append(c_pred)
            gt_states.append(s_gt.astype(np.int64))
            gt_carried_list.append(c_gt)

        gt_div = pairwise_gt_channel_diversity_total(gt_states, gt_carried_list)
        wm_masked = pairwise_wm_channel_diversity_masked_total(
            pred_states, pred_carried_list, gt_states, gt_carried_list,
            tol=action_following_tol,
        )
        afs_gt_channel_total += gt_div
        afs_wm_masked_channel_total += wm_masked

    return {
        "num_initial_states": n,
        "valid_states_used": n - skipped,
        "skipped_too_few_actions": skipped,
        "afs_gt_channel_total": int(afs_gt_channel_total),
        "afs_wm_masked_channel_total": int(afs_wm_masked_channel_total),
        "afs_score": (
            float(afs_wm_masked_channel_total) / float(afs_gt_channel_total)
            if afs_gt_channel_total > 0 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)

    actions = parse_action_list(args.actions)
    required_executable_actions = parse_action_list(args.required_executable_actions)

    ckpt_paths = list_checkpoints(args.checkpoint_dir, args.checkpoint_pattern)
    states, carried = load_test_data(args.test_set_path)
    candidate_indices = np.arange(states.shape[0], dtype=np.int64)

    if args.action_mode == "fixed_interact" and required_executable_actions:
        pool_before = states.shape[0]
        valid_local_indices = filter_initial_states_by_executable_actions(
            states, carried, required_executable_actions
        )
        if valid_local_indices.size == 0:
            raise ValueError(
                f"No initial states satisfy required executable actions: "
                f"{required_executable_actions}"
            )
        candidate_indices = candidate_indices[valid_local_indices]
        states = states[valid_local_indices]
        carried = carried[valid_local_indices]
        print(f"Filtered initial-state pool: {len(candidate_indices)} / {pool_before} states kept.")

    init_states, init_carried, sampled_local_indices = sample_initial_states(
        states, carried, args.num_samples, args.seed
    )
    sampled_indices = candidate_indices[sampled_local_indices]
    obs_shape = init_states.shape[1:]

    all_results = {
        "config": {
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_pattern": args.checkpoint_pattern,
            "test_set_path": args.test_set_path,
            "num_samples": int(init_states.shape[0]),
            "seed": args.seed,
            "device": str(device),
            "action_mode": args.action_mode,
            "actions": actions,
            "required_executable_actions": required_executable_actions,
            "action_following_tol": args.action_following_tol,
        },
        "sampled_indices": sampled_indices.tolist(),
        "results": {},
    }

    print("=" * 80)
    print("Evaluate WM Action Following Score (AFS)")
    print("=" * 80)
    print(f"Device         : {device}")
    print(f"Test set       : {args.test_set_path}")
    print(f"Initial states : {init_states.shape[0]}")
    print(f"Action mode    : {args.action_mode}")
    print(f"Actions        : {actions}")
    print(f"AFS tol        : {args.action_following_tol}")
    print("-" * 80)

    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.basename(ckpt_path)
        method_name = re.sub(r"^wm_", "", re.sub(r"\.pth$", "", ckpt_name))
        print(f"\n[Evaluating] {method_name}")

        model = load_world_model(ckpt_path, obs_shape, device)
        metrics = evaluate_one_model(
            model, init_states, init_carried, actions,
            action_mode=args.action_mode,
            action_following_tol=args.action_following_tol,
        )
        all_results["results"][method_name] = {"checkpoint": ckpt_path, **metrics}
        print(f"  AFS={metrics['afs_score']:.4f}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Saved: {args.output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()

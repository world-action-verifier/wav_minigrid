"""Helpers for AFS data loading, model inference, training and result summaries."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from wav_minigrid.al_experiment import write_csv
from wav_minigrid.models import WorldModel


def load_test_data(test_set_path: str):
    data = np.load(test_set_path)
    states = data["states"]
    while states.ndim > 4:
        states = states.squeeze(1)
    carried = data["carried"]  # [N, 2]
    return states.astype(np.int64), carried.astype(np.int64)


def load_world_model(checkpoint_path: str, obs_shape, device):
    model = WorldModel(observation_shape=obs_shape, num_actions=7).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    name = Path(checkpoint_path).name
    if incompatible.missing_keys:
        print(f"[WARN] {name}: missing keys {incompatible.missing_keys[:5]} ...")
    if incompatible.unexpected_keys:
        print(f"[WARN] {name}: unexpected keys {incompatible.unexpected_keys[:5]} ...")
    model.eval()
    return model


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


def train_checkpoints(config, run_dir, method, seed):
    """Run one method/seed with its own models and save inference checkpoints."""
    if method == 'RLIR-updated-IDM' and not config['RLIR_UPDATE_IDM']:
        raise ValueError('RLIR-updated-IDM requires RLIR_UPDATE_IDM=True.')
    path = Path(__file__).resolve().parents[2] / 'exps' / 'wm_active_learning.py'
    spec = importlib.util.spec_from_file_location('_afs_active_learning', path)
    al = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(al)
    # Apply run-specific settings to a fresh AL module instance.
    for key, value in config.items():
        if hasattr(al, key):
            setattr(al, key, value)
    al.DEVICE = torch.device(config['DEVICE'])
    al.SEED = seed
    al.STRATEGIES = ['RLIR' if method == 'RLIR-updated-IDM' else method]
    al.SAVE_DIR, al.SAVE_MODEL = str(run_dir), 'False'
    checkpoints = Path(run_dir) / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=False)
    saved_rounds = []

    def save_checkpoint(model, strategy, round_id, acquired):
        # Match the skipped evaluation DataLoader's CPU seed draw to preserve
        # the training RNG sequence.
        torch.empty((), dtype=torch.int64, device='cpu').random_()
        torch.save(dict(model_state_dict=model.state_dict(), method=method,
                        seed=seed, round=round_id, acquired=acquired),
                   checkpoints / f'round{round_id:02d}.pth')
        saved_rounds.append(round_id)

    al.run_active_learning(evaluate_prediction_loss=False, on_round_end=save_checkpoint)
    if saved_rounds != list(range(config['NUM_ROUNDS'] + 1)):
        raise ValueError(f'Incomplete checkpoints: {run_dir}')


def acquired_count(config, round_id):
    return (config['ADD_COUNT_FIRST_ROUND'] + (round_id - 1) * config['ADD_COUNT_PER_ROUND']
            if round_id else 0)


def load_round_records(config, run_dir, method):
    """Check round and acquisition counts; prediction-loss columns are unnecessary."""
    path = Path(run_dir) / config['LOG_FILE']
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if [int(row['Round']) for row in rows] != list(range(config['NUM_ROUNDS'] + 1)):
        raise ValueError(f'Incomplete or duplicated rounds: {path}')
    internal = 'RLIR' if method == 'RLIR-updated-IDM' else method
    for round_id, row in enumerate(rows):
        if row['Strategy'] != internal or int(row['Labeled_Size']) != acquired_count(config, round_id):
            raise ValueError(f'Invalid round record: {path}: {row}')
    return rows


def summarize_afs_results(output_dir, config):
    """Average rounds 1 through NUM_ROUNDS per seed, then compute mean and sample SD."""
    root = Path(output_dir)
    rounds, seeds = config['NUM_ROUNDS'], config['SEEDS']
    metrics = ['strict_pair_afs']
    raw, per_seed, summary, per_round = [], [], [], []

    def stats(values):
        return dict(n_seeds=len(values), mean=statistics.mean(values),
                    sd=statistics.stdev(values) if len(values) > 1 else float('nan'))

    for method in config['STRATEGIES']:
        values = {}
        for seed in seeds:
            run_dir = root / 'runs' / f'{method}_seed{seed}'
            rows = json.loads((run_dir / 'metrics.json').read_text())
            load_round_records(config, run_dir, method)
            if [row['round'] for row in rows] != list(range(rounds + 1)):
                raise ValueError(f'Incomplete or duplicated AFS rounds: {run_dir}')
            for round_id, row in enumerate(rows):
                if row['acquired'] != acquired_count(config, round_id):
                    raise ValueError(f'AFS acquisition count mismatch: {run_dir}')
                for metric in metrics:
                    score = row[metric]
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f'Invalid {metric}: {run_dir}, round {round_id}')
                    values[metric, seed, round_id] = score
                raw.append(dict(method=method, seed=seed, round=round_id,
                                acquired=row['acquired'],
                                **{metric: row[metric] for metric in metrics}))
        for metric in metrics:
            averages = [statistics.mean(values[metric, s, r] for r in range(1, rounds + 1))
                        for s in seeds]
            per_seed.extend(dict(metric=metric, method=method, window=f'rounds1-{rounds}',
                                 seed=seed, value=value) for seed, value in zip(seeds, averages))
            summary.append(dict(metric=metric, method=method, window=f'rounds1-{rounds}',
                                **stats(averages)))
            per_round.extend(dict(metric=metric, method=method, round=r,
                                  **stats([values[metric, s, r] for s in seeds]))
                             for r in range(rounds + 1))
    # Publish only complete comparisons; missing seeds/rounds are errors.
    for name, rows in [('results_per_round', raw), ('seed_aggregates', per_seed),
                       ('summary', summary), ('per_round_summary', per_round)]:
        write_csv(root / f'{name}.csv', rows)
    print(f'AFS mean ± sample SD across {len(seeds)} seeds (round 0 excluded):')
    for row in summary:
        print(f"  {row['metric']:16s} {row['method']:17s} {row['window']:12s} "
              f"{row['mean']:.6f} ± {row['sd']:.6f}")
    return summary

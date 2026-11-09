"""Train active-learning models and evaluate strict-pair AFS.

Usage:
    python exps/afs_active_learning.py

Configure paths via `WM_AFS_ACTIVE_LEARNING` in `wav_minigrid.config`.
Requires `pool_p05.npz` and `confirmation_states.npz` in `data/`.
"""


import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from wav_minigrid.al_experiment import experiment_parser, run_experiments
from wav_minigrid.afs_experiment import (
    acquired_count, load_round_records, load_test_data, load_world_model,
    predict_with_action, summarize_afs_results, train_checkpoints,
)
from wav_minigrid.config import WM_AFS_ACTIVE_LEARNING
from wav_minigrid.evaluate_generation import MiniGridPhysicsOracle


def calculate_afs(predictions, truth):
    """Return per-state strict-pair AFS; arrays are [state, action, channel]."""
    predictions, truth = np.asarray(predictions), np.asarray(truth)
    if predictions.shape != truth.shape or truth.ndim != 3 or truth.shape[1] < 2:
        raise ValueError('Expected matching [state, action>=2, channel] arrays.')
    successes, valid_pairs = (np.zeros(len(truth)) for _ in range(2))
    for a, b in combinations(range(truth.shape[1]), 2):
        different = truth[:, a] != truth[:, b]
        correct = (predictions[:, a] == truth[:, a]) & (predictions[:, b] == truth[:, b])
        valid = different.any(axis=1)
        successes += np.all(~different | correct, axis=1) & valid
        valid_pairs += valid
    if not len(truth) or np.any(valid_pairs == 0):
        raise ValueError('Each AFS state must have at least one distinct oracle action pair.')
    return dict(strict_pair_afs=successes / valid_pairs)


def evaluate_checkpoints(config, run_dir, method, seed):
    """Score every checkpoint after training, predicting each state/action separately."""
    run_dir = Path(run_dir)
    rounds = load_round_records(config, run_dir, method)
    states, carried = load_test_data(config['AFS_STATES_PATH'])
    actions = config['AFS_ACTIONS']
    if len(actions) < 2 or len(set(actions)) != len(actions) or any(a not in range(7) for a in actions):
        raise ValueError('AFS_ACTIONS must contain at least two unique actions in 0..6.')
    oracle = MiniGridPhysicsOracle()
    truth = []
    for state, carry in zip(states, carried):
        outcomes = []
        for action in actions:
            grid, col, obj = oracle.step(state.copy(), int(carry[0]), int(carry[1]), action)
            outcomes.append(np.concatenate([grid.reshape(-1), [col, obj]]))
        truth.append(outcomes)
    truth = np.asarray(truth)
    device = torch.device(config['DEVICE'])
    results = []
    for round_id in range(len(rounds)):
        checkpoint = run_dir / 'checkpoints' / f'round{round_id:02d}.pth'
        model = load_world_model(str(checkpoint), states.shape[1:], device)
        predictions = np.empty_like(truth)
        for i, (state, carry) in enumerate(zip(states, carried)):
            for j, action in enumerate(actions):
                grid, next_carry = predict_with_action(model, state, carry, action, device)
                predictions[i, j] = np.concatenate([grid.reshape(-1), next_carry])
        scores = calculate_afs(predictions, truth)
        results.append(dict(round=round_id, acquired=acquired_count(config, round_id),
                            num_states=len(states),
                            **{name: float(value.mean()) for name, value in scores.items()},
                            **{f'per_state_{name}': value.tolist() for name, value in scores.items()}))
        print(f"{method} seed {seed}, round {round_id}: "
              f"strict_pair_afs={results[-1]['strict_pair_afs']:.6f}",
              flush=True)
        del model
    (run_dir / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')


def main():
    if os.environ.get('AL_WORKER') == '1':
        config = json.loads(Path(os.environ['AL_CONFIG_PATH']).read_text())
        torch.set_num_threads(config['CPU_THREADS'])
        torch.set_num_interop_threads(config['CPU_THREADS'])
        run_dir = Path(os.environ['AL_OUTPUT_DIR'])
        method, seed = os.environ['AL_STRATEGIES'], int(os.environ['AL_SEED'])
        train_checkpoints(config, run_dir, method, seed)
        evaluate_checkpoints(config, run_dir, method, seed)
        return

    config = dict(WM_AFS_ACTIVE_LEARNING)
    parser = experiment_parser(config, __doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument('--data-dir', type=Path, help='Directory containing pool_p05.npz and confirmation_states.npz.')
    parser.add_argument('--rounds', type=int, default=config['NUM_ROUNDS'])
    parser.add_argument('--device', default=config['DEVICE'], help='Inference/training device, e.g. cuda or cpu.')
    parser.add_argument('--evaluate-only', action='store_true', help='Re-score saved checkpoints using their saved config.')
    args = parser.parse_args()
    if args.evaluate_only and args.summarize_only:
        parser.error('Choose only one of --evaluate-only and --summarize-only.')
    if args.evaluate_only:
        config = json.loads((args.output_dir / 'experiment_config.json').read_text())
        config['DEVICE'] = args.device
        torch.set_num_threads(config['CPU_THREADS'])
        for method in config['STRATEGIES']:
            for seed in config['SEEDS']:
                evaluate_checkpoints(config, args.output_dir / 'runs' / f'{method}_seed{seed}', method, seed)
        summarize_afs_results(args.output_dir, config)
        return
    if not args.summarize_only:
        config.update(NUM_ROUNDS=args.rounds, DEVICE=args.device)
        if args.data_dir:
            for key in ['DATA_PATH', 'AFS_STATES_PATH']:
                config[key] = str(args.data_dir.resolve() / Path(config[key]).name)
        for key in ['DATA_PATH', 'AFS_STATES_PATH', 'BASE_DATA_PATH',
                    'STAGE1_CKPT', 'INVERSE_MODEL_PATH', 'VIDEO_STAGE1_CKPT']:
            if not Path(config[key]).is_file():
                parser.error(f'Missing {key}: {config[key]}. Set the path in config or use --data-dir.')
    run_experiments(__file__, config, args=args, summarize=summarize_afs_results)


if __name__ == '__main__':
    main()

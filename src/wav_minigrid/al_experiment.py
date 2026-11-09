"""Multi-seed/GPU experiment launcher and seed-level result summaries."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(output_dir, config):
    """Require every requested seed/round; average within seed before computing SD."""
    root = Path(output_dir)
    seeds, methods, rounds = config['SEEDS'], config['STRATEGIES'], config['NUM_ROUNDS']
    raw, per_seed, per_round, aggregate = [], [], [], []

    def stats(values):
        # A single-seed debugging run has no estimable across-seed SD.
        sd = statistics.stdev(values) if len(values) > 1 else float('nan')
        return dict(n=len(values), mean=statistics.mean(values), sd=sd)

    for method in methods:
        losses = {}
        for seed in seeds:
            path = root / 'runs' / f'{method}_seed{seed}' / config['LOG_FILE']
            with path.open() as f:
                rows = list(csv.DictReader(f))
            if [int(row['Round']) for row in rows] != list(range(rounds + 1)):
                raise ValueError(f'Incomplete or duplicated rounds: {path}')
            for r, row in enumerate(rows):
                value = float(row['Test_MSE'])
                labeled = 0 if r == 0 else config['ADD_COUNT_FIRST_ROUND'] + (r-1)*config['ADD_COUNT_PER_ROUND']
                if row['Strategy'] != method or int(row['Labeled_Size']) != labeled or not math.isfinite(value):
                    raise ValueError(f'Invalid result in {path}: {row}')
                losses[seed, r] = value
                raw.append(dict(method=method, seed=seed, round=r,
                                labeled_size=labeled, test_prediction_loss=value))
            per_seed.append(dict(method=method, seed=seed, mean_rounds_1_to_N=
                                 statistics.mean(losses[seed, r] for r in range(1, rounds+1))))
        aggregate.append(dict(method=method, rounds=f'1-{rounds}',
                              **stats([row['mean_rounds_1_to_N'] for row in per_seed if row['method'] == method])))
        per_round.extend(dict(method=method, round=r, **stats([losses[s, r] for s in seeds]))
                         for r in range(rounds+1))

    # Write only after all requested runs pass validation; never average partial runs.
    for name, rows in [('all_results_long', raw), ('per_seed_aggregate', per_seed),
                       ('all_round_aggregate', aggregate), ('per_round_summary', per_round)]:
        write_csv(root / f'{name}.csv', rows)
    print(f'All-round mean ± sample SD ({len(seeds)} seeds, rounds 1–{rounds}):')
    for row in aggregate:
        print(f"  {row['method']:12s} {row['mean']:.6f} ± {row['sd']:.6f}")
    return aggregate


def experiment_parser(config, description=__doc__):
    """Common CLI for prediction-error and AFS active-learning experiments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[int(os.environ['AL_SEED'])] if 'AL_SEED' in os.environ else config['SEEDS'])
    parser.add_argument('--strategies', nargs='+', choices=config['STRATEGIES'],
                        default=os.environ.get('AL_STRATEGIES', ','.join(config['STRATEGIES'])).split(','))
    parser.add_argument('--gpus', nargs='+', help='CUDA device IDs; one worker per ID (default: sequential).')
    parser.add_argument('--output-dir', type=Path, default=Path(config['SAVE_DIR']))
    parser.add_argument('--summarize-only', action='store_true', help='Use the saved config and CSVs; no training.')
    return parser


def run_experiments(script_path, config, *, args=None, summarize=summarize_results):
    """One subprocess per method/seed; optionally one concurrent worker per GPU."""
    parser = experiment_parser(config)
    args = parser.parse_args() if args is None else args
    root = args.output_dir.resolve()
    config_path = root / 'experiment_config.json'
    if args.summarize_only:
        summarize(root, json.loads(config_path.read_text()))
        return
    for name, values in [('seeds', args.seeds), ('strategies', args.strategies), ('gpus', args.gpus or [None])]:
        if len(set(values)) != len(values):
            parser.error(f'Duplicate {name} are not allowed.')
    if any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('Seeds must be in [0, 2**32).')
    if any(m not in config['STRATEGIES'] for m in args.strategies):
        parser.error('Unknown strategy in AL_STRATEGIES.')
    for key in ['NUM_ROUNDS', 'ADD_COUNT_FIRST_ROUND', 'ADD_COUNT_PER_ROUND', 'CPU_THREADS']:
        if config[key] < 1:
            parser.error(f'{key} must be positive.')
    # Fail rather than overwrite an earlier or partially completed experiment.
    root.mkdir(parents=True, exist_ok=False)
    (root / 'runs').mkdir()
    (root / 'logs').mkdir()
    config = dict(config, SEEDS=args.seeds, STRATEGIES=args.strategies, SAVE_DIR=str(root))
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    jobs = [(method, seed) for method in args.strategies for seed in args.seeds]
    devices = args.gpus or [None]

    def run_group(gpu, tasks):
        for method, seed in tasks:
            name = f'{method}_seed{seed}'
            run_dir = root / 'runs' / name
            run_dir.mkdir()
            env = os.environ.copy()
            env.update(AL_WORKER='1', AL_SEED=str(seed), AL_STRATEGIES=method,
                       AL_OUTPUT_DIR=str(run_dir), AL_CONFIG_PATH=str(config_path), PYTHONHASHSEED=str(seed),
                       OMP_NUM_THREADS=str(config['CPU_THREADS']), MKL_NUM_THREADS=str(config['CPU_THREADS']),
                       OPENBLAS_NUM_THREADS=str(config['CPU_THREADS']), PYTHONDONTWRITEBYTECODE='1')
            if gpu is not None:
                env['CUDA_VISIBLE_DEVICES'] = gpu
            log_path = root / 'logs' / f'{name}.log'
            print(f'Running {name} on CUDA_VISIBLE_DEVICES={env.get("CUDA_VISIBLE_DEVICES", "default")}', flush=True)
            with log_path.open('w') as log:
                subprocess.run([sys.executable, '-B', '-u', str(Path(script_path).resolve())],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(run_group, gpu, jobs[i::len(devices)]) for i, gpu in enumerate(devices)]
        for future in futures:
            future.result()  # Any failed run prevents publishing aggregate statistics.
    summarize(root, config)

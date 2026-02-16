"""
Test World Model and Inverse Model performance across different state complexities.
"""

import os
import sys
import argparse
import json
import re

import torch
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from asim_minigrid.dataset import MiniGridDynamicsDataset
from asim_minigrid.models import WorldModel, SparseIDM
from asim_minigrid.evaluate_generation import MiniGridPhysicsOracle
from asim_minigrid.utils import test_world_model, test_inverse_model
from asim_minigrid.config import STATE_COMPLEXITY_GAP, DEVICE

# Configuration
DEFAULT_TEST_SETS = STATE_COMPLEXITY_GAP["DEFAULT_TEST_SETS"]
DEFAULT_WORLD_MODEL_PATH = STATE_COMPLEXITY_GAP["DEFAULT_WORLD_MODEL_PATH"]
DEFAULT_INVERSE_MODEL_PATH = STATE_COMPLEXITY_GAP["DEFAULT_INVERSE_MODEL_PATH"]
DEFAULT_BATCH_SIZE = STATE_COMPLEXITY_GAP["DEFAULT_BATCH_SIZE"]
DEFAULT_FORWARD_CARRIED_LOSS_WEIGHT = STATE_COMPLEXITY_GAP["DEFAULT_FORWARD_CARRIED_LOSS_WEIGHT"]


def parse_complexity_from_path(path: str) -> int:
    """
    Parse state complexity from path (e.g., o6, o8, o10).
    Returns parsed number, or -1 if parsing fails.
    """
    match = re.search(r"-o(\d+)-", os.path.basename(path))
    if match:
        return int(match.group(1))
    return -1


def load_world_model(model_path, obs_shape, num_actions):
    """
    Load trained world model.
    
    Args:
        model_path: Path to model weights
        obs_shape: Observation shape
        num_actions: Number of actions
    
    Returns:
        Loaded model
    """
    print(f"Loading model from: {model_path}")

    model = WorldModel(obs_shape, num_actions).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print("Model loaded successfully: WorldModel")
    return model


def load_inverse_model(model_path, num_actions):
    """
    Load trained inverse model.
    
    Args:
        model_path: Path to model weights
        num_actions: Number of actions
    
    Returns:
        Loaded model
    """
    print(f"Loading Inverse Model from: {model_path}")

    model = SparseIDM(num_actions=num_actions).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print("Inverse Model loaded successfully: SparseIDM")
    return model


def evaluate_on_dataset(
    test_set_path: str,
    world_model,
    inverse_model,
    batch_size: int,
    forward_carried_loss_weight: float,
):
    """Evaluate both models on a single test set."""
    test_dataset = MiniGridDynamicsDataset(test_set_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    world_model.eval()
    world_results = test_world_model(
        world_model,
        test_loader,
        forward_carried_loss_weight=forward_carried_loss_weight,
        device=DEVICE,
    )

    inverse_model.eval()
    oracle = MiniGridPhysicsOracle()
    inverse_results = test_inverse_model(
        inverse_model,
        oracle,
        test_loader,
        device=DEVICE,
    )

    return test_dataset, world_results, inverse_results


def main():
    parser = argparse.ArgumentParser(description="Test model performance across different state complexities")
    parser.add_argument(
        "--test_sets",
        type=str,
        nargs="+",
        default=DEFAULT_TEST_SETS,
        help="List of test set paths",
    )
    parser.add_argument(
        "--world_model_path",
        type=str,
        default=DEFAULT_WORLD_MODEL_PATH,
        help="Path to World Model weights",
    )
    parser.add_argument(
        "--inverse_model_path",
        type=str,
        default=DEFAULT_INVERSE_MODEL_PATH,
        help="Path to Inverse Model weights",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--forward_carried_loss_weight",
        type=float,
        default=DEFAULT_FORWARD_CARRIED_LOSS_WEIGHT,
        help="Carried loss weight",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="state_complexity_test_results.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("State Complexity Test Script")
    print("=" * 80)
    print(f"Number of test sets: {len(args.test_sets)}")
    print(f"World Model path: {args.world_model_path}")
    print(f"Inverse Model path: {args.inverse_model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    results_list = []
    loaded_obs_shape = None
    world_model = None
    inverse_model = None
    num_actions = 7

    for test_set_path in args.test_sets:
        print("\n" + "=" * 80)
        print(f"Test set: {test_set_path}")
        print("=" * 80)

        test_dataset = MiniGridDynamicsDataset(test_set_path)
        obs_shape = test_dataset.states.shape[1:]

        if world_model is None or obs_shape != loaded_obs_shape:
            if world_model is None:
                print("Loading World Model...")
            else:
                print("Observation shape changed, reloading World Model...")
            world_model = load_world_model(
                args.world_model_path,
                obs_shape,
                num_actions,
            )

            print("Loading Inverse Model...")
            inverse_model = load_inverse_model(
                args.inverse_model_path,
                num_actions,
            )
            loaded_obs_shape = obs_shape

        _, world_results, inverse_results = evaluate_on_dataset(
            test_set_path,
            world_model,
            inverse_model,
            args.batch_size,
            args.forward_carried_loss_weight,
        )

        complexity = parse_complexity_from_path(test_set_path)

        results_list.append(
            {
                "test_set_path": test_set_path,
                "state_complexity": complexity,
                "world_model": {
                    "dyn_accuracy": world_results.get("dyn_acc", 0.0),
                },
                "inverse_model": {
                    "dyn_accuracy": inverse_results.get("dyn_accuracy", 0.0),
                },
            }
        )

    print("\n" + "=" * 80)
    print("Test Results Summary (Dynamic Accuracy)")
    print("=" * 80)
    print(
        f"{'Complexity':<12} {'WM Dyn Acc':<15} {'IM Dyn Acc':<15}"
    )
    print("-" * 50)
    for item in results_list:
        comp = item["state_complexity"]
        wm = item["world_model"]
        im = item["inverse_model"]
        print(
            f"{comp:<12} {wm['dyn_accuracy']:<15.4f} {im['dyn_accuracy']:<15.4f}"
        )

    with open(args.output, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Simple runner script for FL evaluation.

This script provides an easy way to run the evaluation after FL training
has completed and generated the required array files.

Usage:
    python run_evaluation.py [--dataset PSM] [--seq_length 100] [--thresh_num 1000]
"""

import subprocess
import sys
import os
from pathlib import Path


def check_array_files(dataset="PSM", data_num=0):
    """Check if the required array files exist."""
    time_path = f"./time_arrays/{dataset}_{data_num}_time_evaluation_array.pkl"
    freq_path = f"./freq_arrays/{dataset}_{data_num}_freq_evaluation_array.pkl"

    time_exists = os.path.exists(time_path)
    freq_exists = os.path.exists(freq_path)

    if not time_exists or not freq_exists:
        print("❌ Required array files not found:")
        if not time_exists:
            print(f"   Missing: {time_path}")
        if not freq_exists:
            print(f"   Missing: {freq_path}")
        print("\n💡 Please run FL training first:")
        print("   python run_simulation.py")
        return False

    print("✅ Array files found:")
    print(f"   - {time_path}")
    print(f"   - {freq_path}")
    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run FL evaluation')
    parser.add_argument('--dataset', type=str, default='PSM',
                        help='Dataset name (default: PSM)')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='Sequence length (default: 100)')
    parser.add_argument('--nest_length', type=int, default=25,
                        help='Nested sequence length (default: 25)')
    parser.add_argument('--thresh_num', type=int, default=1000,
                        help='Number of thresholds (default: 1000)')
    parser.add_argument('--data_num', type=int, default=0,
                        help='Data number (default: 0)')
    parser.add_argument('--step', type=int, default=1,
                        help='Step size (default: 1)')

    args = parser.parse_args()

    print("🔍 FL Evaluation Runner")
    print("=" * 50)

    # Check if array files exist
    if not check_array_files(args.dataset, args.data_num):
        return 1

    # Prepare command
    cmd = [
        sys.executable,
        "evaluation_fl.py",
        "--dataset", args.dataset,
        "--seq_length", str(args.seq_length),
        "--nest_length", str(args.nest_length),
        "--thresh_num", str(args.thresh_num),
        "--data_num", str(args.data_num),
        "--step", str(args.step)
    ]

    print(f"\n🚀 Running evaluation with settings:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Sequence Length: {args.seq_length}")
    print(f"   Nest Length: {args.nest_length}")
    print(f"   Thresholds: {args.thresh_num}")
    print(f"   Data Number: {args.data_num}")
    print("=" * 50)

    # Run evaluation
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running evaluation: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
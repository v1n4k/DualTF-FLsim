#!/usr/bin/env python3
"""Test script to validate the FL workflow.

This script performs a quick validation of the complete workflow:
1. Check if all required files exist
2. Validate imports
3. Test array generation logic
4. Test evaluation compatibility

This is for testing purposes only - not meant to replace actual FL training.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Make core symbols available globally for subsequent tests
# Make core symbols available globally for subsequent tests
from dualflsim.task import FederatedDualTF, get_weights, set_weights  # noqa: E402

# Constants to keep tests fast
TEST_SEQ_LEN = 16
TEST_NEST_LEN = 8


def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")

    try:
        # FL-specific imports
        from dualflsim.task import FederatedDualTF, get_weights, set_weights
        from utils.array_generator import generate_evaluation_arrays
        print("  ✅ FL task imports successful")

        # Data loader imports
        from utils.data_loader import load_PSM
        print("  ✅ Data loader imports successful")

        # Model imports
        from model.TimeTransformer import AnomalyTransformer
        from model.FrequencyTransformer import FrequencyTransformer
        print("  ✅ Model imports successful")

        return True

    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_model_creation():
    """Test if we can create the federated model."""
    print("🧪 Testing model creation...")

    try:
        seq_len = TEST_SEQ_LEN
        nest_len = TEST_NEST_LEN
        time_model_args = {'win_size': seq_len, 'enc_in': 25, 'c_out': 25, 'e_layers': 3}
        freq_model_args = {
            'win_size': (seq_len - nest_len + 1) * (nest_len // 2),
            'enc_in': 25,
            'c_out': 25,
            'e_layers': 3,
            'n_heads': 2,
        }

        model = FederatedDualTF(time_model_args, freq_model_args)
        weights = get_weights(model)
        print(f"  ✅ Model created successfully with {len(weights)} parameters")

        # Test weight setting
        set_weights(model, weights)
        print("  ✅ Weight setting successful")

        return True

    except Exception as e:
        print(f"  ❌ Model creation error: {e}")
        return False


def test_data_loading():
    """Test if we can load data."""
    print("🧪 Testing data loading...")

    try:
        from dualflsim.task import load_centralized_test_data

        # Test centralized data loading
        testloader_time, testloader_freq = load_centralized_test_data(
            time_batch_size=16,
            freq_batch_size=4,
            seq_length=TEST_SEQ_LEN,
            nest_length=TEST_NEST_LEN,
        )

        time_size = len(testloader_time.dataset) if testloader_time else 0
        freq_size = len(testloader_freq.dataset) if testloader_freq else 0

        print(f"  ✅ Data loaded - Time: {time_size}, Freq: {freq_size} samples")

        return testloader_time, testloader_freq

    except Exception as e:
        print(f"  ❌ Data loading error: {e}")
        return None, None


def test_array_generation_logic():
    """Test the array generation functions."""
    print("🧪 Testing array generation logic...")

    try:
        # Test with dummy data
        model_created = test_model_creation()
        if not model_created:
            return False

        testloader_time, testloader_freq = test_data_loading()
        if testloader_time is None or testloader_freq is None:
            return False

        # Create model
        seq_len = TEST_SEQ_LEN
        nest_len = TEST_NEST_LEN
        time_model_args = {'win_size': seq_len, 'enc_in': 25, 'c_out': 25, 'e_layers': 3}
        freq_model_args = {
            'win_size': (seq_len - nest_len + 1) * (nest_len // 2),
            'enc_in': 25,
            'c_out': 25,
            'e_layers': 3,
            'n_heads': 2,
        }

        from dualflsim.task import FederatedDualTF
        model = FederatedDualTF(time_model_args, freq_model_args)

        # Use CPU for testing
        device = torch.device("cpu")
        model.to(device)

        # Test array generation functions individually
        from utils.array_generator import generate_time_evaluation_array, generate_freq_evaluation_array

        print("  🔄 Testing time array generation...")
        time_df = generate_time_evaluation_array(
            model=model,
            testloader_time=testloader_time,
            device=device,
            seq_length=seq_len
        )
        print(f"  ✅ Time array generated: {time_df.shape}")

        print("  🔄 Testing freq array generation...")
        freq_df = generate_freq_evaluation_array(
            model=model,
            testloader_freq=testloader_freq,
            device=device,
            seq_length=seq_len
        )
        print(f"  ✅ Freq array generated: {freq_df.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Array generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test if all required files exist."""
    print("🧪 Testing file structure...")

    required_files = [
        "run_simulation.py",
        "evaluation_fl.py",
        "run_evaluation.py",
        "utils/array_generator.py",
        "dualflsim/task.py",
        "dualflsim/client_app.py",
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("🚀 FL Workflow Validation")
    print("=" * 50)

    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Array Generation Logic", test_array_generation_logic),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            if test_name == "Data Loading":
                # Special handling for data loading test
                result = test_func()
                success = result[0] is not None and result[1] is not None
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1

    print(f"\nTests passed: {passed}/{len(results)}")

    if passed == len(results):
        print("🎉 All tests passed! The FL workflow should work correctly.")
        print("\n📝 Next steps:")
        print("1. Run FL training: python run_simulation.py")
        print("2. Run evaluation: python run_evaluation.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    exit(main())
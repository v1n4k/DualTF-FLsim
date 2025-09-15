# DualTF-FLSim: Federated Learning with Array Generation

This document explains the enhanced DualTF-FLSim implementation that integrates array generation capabilities for comprehensive evaluation, matching the original DualTF workflow.

## 🚀 New Features

### ✨ Optimized GPU Usage
- **Training Phase**: Uses all 4 GPUs for maximum parallelism (33% improvement over previous setup)
- **Post-Training Phase**: Uses 1 GPU for comprehensive array generation and evaluation

### 🔧 Array Generation
- Generates `time_evaluation_array.pkl` and `freq_evaluation_array.pkl` files after FL training
- Compatible with original DualTF evaluation methodology
- Same array structure and format as non-FL implementation

### 📊 Comprehensive Evaluation
- FL-compatible evaluation script with same metrics as original
- Multiple evaluation modes: Point Adjusted, Point-Wise, Released Point-Wise
- Seamless workflow from FL training to comprehensive evaluation

## 📁 File Structure

```
DualTF-FLSim/
├── dualflsim/
│   ├── run_simulation.py          # Main FL training script (enhanced)
│   ├── evaluation_fl.py           # FL-compatible evaluation script
│   ├── run_evaluation.py          # Easy evaluation runner
│   ├── test_workflow.py           # Validation script
│   └── utils/
│       └── array_generator.py     # Array generation module
├── time_arrays/                   # Generated time domain arrays
├── freq_arrays/                   # Generated frequency domain arrays
└── README_ARRAYS.md              # This file
```

## 🏃‍♂️ Quick Start

### 1. Validate Setup
```bash
cd dualflsim
python test_workflow.py
```

### 2. Run FL Training with Array Generation
```bash
python run_simulation.py
```
This will:
- Use all 4 GPUs for FL training
- Generate evaluation arrays after training completes
- Save arrays to `./time_arrays/` and `./freq_arrays/`

### 3. Run Comprehensive Evaluation
```bash
python run_evaluation.py
```
Or with custom parameters:
```bash
python run_evaluation.py --thresh_num 1000 --seq_length 100
```

## 🔧 Technical Details

### Resource Management

#### During Training
- **Ray Configuration**: `num_gpus=4` (all GPUs)
- **Client Resources**: `num_gpus=1` per client
- **Max Concurrent Clients**: 4 (full GPU utilization)

#### After Training
- **Array Generation**: Uses any available GPU
- **Memory Management**: Automatic cleanup and cache clearing

### Array Generation Process

#### Time Domain Array
- **Shape**: `(7, sequence_length)`
- **Indices**: `['Normal', 'Anomaly', '#Seq', 'Pred(%)', 'Pred', 'GT', 'Avg(RE)']`
- **Process**: Runs inference on centralized test data using aggregated FL model

#### Frequency Domain Array
- **Shape**: `(5, sequence_length)`
- **Indices**: `['#SubSeq', '#GrandSeq', 'Avg(exp(RE))', 'Pred', 'GT']`
- **Process**: Similar to time domain but with frequency-specific transformations

### Evaluation Compatibility

The generated arrays are fully compatible with the original evaluation methodology:

1. **Load Arrays**: Same pickle format as original DualTF
2. **Score Fusion**: Combines time and frequency scores using RobustScaler + normalization
3. **Threshold Sweeping**: Identical threshold generation and evaluation logic
4. **Metrics**: Same precision, recall, F1, AUC calculations

## 📊 Performance Benefits

### Training Performance
- **33% More Parallelism**: 4 concurrent clients vs 3 previously
- **Faster Convergence**: More diverse client updates per round
- **Better Resource Utilization**: All GPUs actively used during training

### Evaluation Performance
- **GPU-Accelerated Inference**: Fast array generation using dedicated GPU
- **Comprehensive Metrics**: Same evaluation depth as original DualTF
- **Automated Workflow**: Seamless transition from training to evaluation

## 🔍 Workflow Comparison

### Original DualTF Workflow
```
1. python main.py          → Saves time_evaluation_array.pkl
2. python main_freq.py     → Saves freq_evaluation_array.pkl
3. python evaluation.py    → Loads arrays and evaluates
```

### New FL Workflow
```
1. python run_simulation.py → FL training + saves both arrays
2. python run_evaluation.py → Loads arrays and evaluates (same logic)
```

## 🛠️ Configuration Options

### FL Training (`run_simulation.py`)
- **Total Rounds**: `total_rounds = 30`
- **Sequence Length**: `seq_len = 100`
- **Client Selection**: `min_fit_clients = 6`
- **Dataset**: Currently configured for PSM dataset

### Array Generation
- **Dataset**: `dataset = "PSM"`
- **Data Number**: `data_num = 0`
- **Nest Length**: `nest_length = 25`

### Evaluation (`evaluation_fl.py`)
- **Thresholds**: `--thresh_num 1000`
- **Sequence Length**: `--seq_length 100`
- **Evaluation Modes**: Point Adjusted, Point-Wise, Released Point-Wise

## 🧪 Testing & Validation

### Validation Script
```bash
python test_workflow.py
```
Tests:
- File structure and imports
- Model creation and weight handling
- Data loading functionality
- Array generation logic
- End-to-end compatibility

### Manual Testing
1. **Check Arrays Exist**:
   ```bash
   ls -la time_arrays/ freq_arrays/
   ```

2. **Verify Array Structure**:
   ```python
   import pandas as pd
   time_df = pd.read_pickle('./time_arrays/PSM_0_time_evaluation_array.pkl')
   print(time_df.shape)  # Should be (7, N)
   print(time_df.index)  # Should show correct labels
   ```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure you're running from the `dualflsim/` directory
   - Check that all required packages are installed

2. **GPU Memory Issues**:
   - Reduce batch sizes if encountering OOM errors
   - Ensure proper cleanup between training and array generation

3. **Array File Not Found**:
   - Run FL training first: `python run_simulation.py`
   - Check that training completed successfully

4. **Evaluation Errors**:
   - Verify array files exist and have correct structure
   - Ensure sequence length matches training configuration

### Debug Mode
Add debug prints to see detailed execution:
```python
# In array_generator.py
print(f"[DEBUG] Processing batch {i}/{total_batches}")

# In evaluation_fl.py
print(f"[DEBUG] Final scores shape: {final_as.shape}")
```

## 🎯 Next Steps

### Possible Extensions
1. **Multi-Dataset Support**: Extend beyond PSM dataset
2. **Hyperparameter Optimization**: Integrate with FL training
3. **Real-time Evaluation**: Stream evaluation during FL rounds
4. **Advanced Aggregation**: Explore alternative aggregation strategies

### Performance Optimizations
1. **Mixed Precision**: Further reduce memory usage
2. **Gradient Compression**: Reduce communication overhead
3. **Asynchronous Evaluation**: Parallel array generation
4. **Distributed Evaluation**: Multi-GPU evaluation

## 🤝 Contributing

When modifying the array generation logic:
1. Ensure compatibility with original evaluation.py
2. Maintain array structure and naming conventions
3. Test with validation script before committing
4. Update documentation for any parameter changes

## 📝 License

Same as original DualTF project.
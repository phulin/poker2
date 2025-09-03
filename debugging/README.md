# Debugging Scripts

This directory contains various debugging and diagnostic scripts for the AlphaHoldem training system.

## Scripts Overview

### 🔍 **diagnose_training.py**
Comprehensive diagnostic script that checks:
- Model health (parameters, NaN/Inf values)
- Forward pass functionality
- Trajectory collection
- Training step execution
- Parameter updates

**Usage:**
```bash
cd /path/to/poker2
python debugging/diagnose_training.py
```

### 📊 **check_advantages.py**
Analyzes the distribution of advantages and returns during training to identify potential issues:
- Advantage bias and variance
- Return distributions
- Warning flags for problematic values

**Usage:**
```bash
cd /path/to/poker2
python debugging/check_advantages.py
```

### ⚙️ **test_conservative.py**
Tests training with conservative hyperparameters:
- Lower learning rate (1e-4)
- Larger batch size (32)
- Less aggressive gradient clipping (0.5)
- Tracks loss and reward trends

**Usage:**
```bash
cd /path/to/poker2
python debugging/test_conservative.py
```

### 🔄 **compare_ppo.py**
Compares Trinal-Clip PPO vs Standard PPO loss functions:
- Side-by-side loss comparison
- Ratio analysis
- Identifies which loss function works better

**Usage:**
```bash
cd /path/to/poker2
python debugging/compare_ppo.py
```

### 🎛️ **test_settings.py**
Tests multiple hyperparameter configurations:
- Conservative, Aggressive, and Balanced settings
- Compares different learning rates, batch sizes, and value coefficients
- Helps find optimal training parameters

**Usage:**
```bash
cd /path/to/poker2
python debugging/test_settings.py
```

### 📈 **visualize_trajectories.py**
Explains and visualizes how `trajectories_per_step` affects training:
- Shows training flow
- Demonstrates buffer management
- Explains model update thresholds

**Usage:**
```bash
cd /path/to/poker2
python debugging/visualize_trajectories.py
```

### 🔬 **detailed_training_analysis.py**
Performs 10 regular training steps followed by one extremely detailed analysis:
- Complete GAE computation breakdown
- Detailed loss function analysis
- Gradient computation and clipping
- Parameter update tracking
- Comprehensive training diagnostics

**Usage:**
```bash
cd /path/to/poker2
python debugging/detailed_training_analysis.py
```

## Quick Debugging Workflow

1. **Start with diagnostics:**
   ```bash
   cd /path/to/poker2
   python debugging/diagnose_training.py
   ```

2. **Check advantage distribution:**
   ```bash
   cd /path/to/poker2
   python debugging/check_advantages.py
   ```

3. **Test conservative settings:**
   ```bash
   cd /path/to/poker2
   python debugging/test_conservative.py
   ```

4. **Compare loss functions:**
   ```bash
   cd /path/to/poker2
   python debugging/compare_ppo.py
   ```

5. **Find optimal settings:**
   ```bash
   cd /path/to/poker2
   python debugging/test_settings.py
   ```

6. **Understand training flow:**
   ```bash
   cd /path/to/poker2
   python debugging/visualize_trajectories.py
   ```

7. **Detailed training analysis:**
   ```bash
   cd /path/to/poker2
   python debugging/detailed_training_analysis.py
   ```

## Common Issues and Solutions

### Loss Not Decreasing
- Run `diagnose_training.py` to check model health
- Use `test_conservative.py` for stable settings
- Try `compare_ppo.py` to see if Trinal-Clip is the issue

### High Advantage Bias
- Run `check_advantages.py` to identify the problem
- Consider adjusting value function initialization

### Training Instability
- Use `test_settings.py` to find better hyperparameters
- Increase `trajectories_per_step` for more stable updates

### Understanding Parameters
- Use `visualize_trajectories.py` to understand training flow
- Check how different settings affect model updates

# AlphaHoldem Training on vast.ai - Complete Setup Guide

This guide will help you set up AlphaHoldem training on vast.ai cloud GPUs.

## Prerequisites

1. **vast.ai Account**: Sign up at [vast.ai](https://vast.ai/)
2. **Payment Method**: Add credits to your account (new users get $2 free)
3. **SSH Key**: Set up SSH keys in your vast.ai account settings

## Step 1: Prepare Your Project

Your project is already set up with:
- ✅ CUDA support added to training scripts
- ✅ Docker configuration created
- ✅ Training script for vast.ai created

## Step 2: Upload Your Project to GitHub

1. Push your project to GitHub (if not already done):
```bash
git add .
git commit -m "Add vast.ai support"
git push origin main
```

## Step 3: Configure vast.ai Instance

### Option A: Using Docker Image (Recommended)

1. **Go to vast.ai Console** → "EDIT IMAGE & CONFIGURATION"
2. **Docker Image Templates** → "Create New Template"
3. **Configure the template**:
   - **Image Path/Tag**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`
   - **Launch Mode**: "Run a jupyter python notebook (easiest)"
   - **Check**: "Use Jupyter Lab interface"
   - **Check**: "Expose SSH port"
   - **Check**: "Expose TensorBoard port" (optional)
4. **Click**: "SELECT & SAVE"

### Option B: Using Custom Dockerfile

1. **Create a Docker image** with your project:
```bash
# Build the image locally (optional, for testing)
docker build -t alphaholdem-vast .

# Or use the Dockerfile directly on vast.ai
```

## Step 4: Rent a GPU Instance

1. **Go to "Create"** section
2. **Filter GPUs** by:
   - **GPU Type**: RTX 4090, RTX 4080, or RTX 3090 (recommended)
   - **Price**: $0.20-$0.50/hour typically
   - **Reliability**: 95%+ uptime
3. **Click "RENT"** on your chosen GPU
4. **Wait for instance** to start (usually 1-2 minutes)

## Step 5: Set Up the Training Environment

### Access via Jupyter Lab:
1. **Click the Jupyter link** in your instance
2. **Open a terminal** in Jupyter Lab
3. **Clone your project**:
```bash
git clone https://github.com/YOUR_USERNAME/poker2.git
cd poker2
```

### Access via SSH:
```bash
ssh -p PORT root@IP_ADDRESS
```

### Install Dependencies:
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install the package with all extras (includes wandb, tensorboard, etc.)
pip install -e .[all]
```

## Step 6: Configure Training Parameters

### Hydra Configuration System

The project now uses Hydra for configuration management, making it easy to switch between different training setups:

#### Available Configurations:
- **`config`**: Default configuration (balanced)
- **`config_high_perf`**: High-performance configuration for vast.ai GPUs
- **`config_fast`**: Fast configuration for testing and development

#### Configuration Structure:
```
conf/
├── config.yaml              # Default configuration (balanced)
├── config_high_perf.yaml    # High-performance preset for vast.ai
└── config_fast.yaml         # Fast testing preset
```

Each configuration file contains all settings in a single, easy-to-read format:
- **Training parameters**: Learning rate, batch size, epochs, etc.
- **Model configuration**: Architecture, channels, hidden layers
- **Environment settings**: Game rules, betting bins, encoders
- **RL parameters**: K-Best pool, evaluation intervals
- **Device and logging**: GPU settings, Wandb configuration

### Environment Variables (Optional):
```bash
export STEPS=2000                    # Number of training steps
export K_BEST_POOL_SIZE=10          # Size of opponent pool
export MIN_ELO_DIFF=50.0            # Minimum ELO difference
export CHECKPOINT_INTERVAL=50       # Save checkpoints every N steps
export EVAL_INTERVAL=100            # Evaluate every N steps
export BATCH_SIZE=1024              # Batch size
export NUM_ENVS=512                 # Number of parallel environments
export USE_WANDB=true               # Enable Weights & Biases logging
export WANDB_PROJECT="poker-kbest-vast"  # Wandb project name
```

### Wandb Setup (Optional but Recommended):
```bash
# Login to wandb (if using)
wandb login
```

## Step 7: Start Training

### Method 1: Using the Training Script
```bash
./train_vast.sh
```

### Method 2: Using Hydra Configurations
```bash
# Use high-performance configuration (recommended for vast.ai)
python alphaholdem/cli/train_kbest.py --config-name=config_high_perf

# Use default configuration
python alphaholdem/cli/train_kbest.py --config-name=config

# Use fast configuration for testing
python alphaholdem/cli/train_kbest.py --config-name=config_fast
```

### Method 3: Override Configuration Parameters
```bash
python alphaholdem/cli/train_kbest.py \
    --config-name=config_high_perf \
    device=cuda \
    num_steps=2000 \
    k_best_pool_size=10 \
    min_elo_diff=50.0 \
    checkpoint_interval=50 \
    eval_interval=100 \
    checkpoint_dir=/workspace/checkpoints \
    use_tensor_env=true \
    num_envs=512 \
    wandb_project="poker-kbest-vast" \
    wandb_name="vast-$(date +%Y%m%d-%H%M%S)" \
    wandb_tags="[vast-ai,poker,kbest,ppo]"
```

### Method 4: Resume from Checkpoint
```bash
python alphaholdem/cli/train_kbest.py \
    --config-name=config_high_perf \
    device=cuda \
    resume_from=/workspace/checkpoints/checkpoint_step_1000.pt \
    use_tensor_env=true \
    num_envs=512
```

## Step 8: Monitor Training

### Check GPU Usage:
```bash
nvidia-smi
```

### Monitor Training Progress:
- **Wandb**: Check your wandb dashboard
- **Checkpoints**: Saved in `/workspace/checkpoints/`
- **Logs**: Training output in terminal

### View Checkpoints:
```bash
ls -la /workspace/checkpoints/
```

## Step 9: Download Results

### Download Checkpoints:
```bash
# From your local machine
scp -P PORT root@IP_ADDRESS:/workspace/checkpoints/* ./checkpoints/
```

### Download Logs:
```bash
# From your local machine
scp -P PORT root@IP_ADDRESS:/workspace/logs/* ./logs/
```

## Step 10: Terminate Instance

**Important**: Always terminate your instance when done to avoid charges!

1. **Go to "Instances"** section
2. **Click the trash can icon** to terminate
3. **Confirm termination**

## Recommended GPU Configurations

### Budget Option ($0.20-0.30/hour):
- **RTX 3080** or **RTX 4070**
- **Batch size**: 512-1024
- **Num envs**: 256-512

### Performance Option ($0.40-0.60/hour):
- **RTX 4090** or **RTX 4080**
- **Batch size**: 1024-2048
- **Num envs**: 512-1024

### High-End Option ($0.80-1.20/hour):
- **RTX 6000** or **A100**
- **Batch size**: 2048+
- **Num envs**: 1024+

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce `--batch-size` or `--num-envs`
   - Use gradient accumulation

2. **Instance Won't Start**:
   - Check GPU availability
   - Try different GPU types
   - Restart the instance

3. **Training Too Slow**:
   - Increase `--num-envs` for tensorized environment
   - Use `--use-tensor-env` flag
   - Check GPU utilization with `nvidia-smi`

4. **Wandb Connection Issues**:
   - Check internet connection
   - Verify wandb login
   - Use `--no-use-wandb` to disable

### Performance Tips:

1. **Use Tensorized Environment**: Always use `--use-tensor-env`
2. **Optimize Batch Size**: Start with 1024, adjust based on GPU memory
3. **Monitor GPU Usage**: Keep GPU utilization >80%
4. **Save Checkpoints**: Use reasonable checkpoint intervals
5. **Use Wandb**: Track experiments and compare runs

## Cost Estimation

### Typical Training Run (2000 steps):
- **RTX 4090**: ~2-3 hours = $0.80-1.80
- **RTX 3080**: ~4-6 hours = $0.80-1.80
- **RTX 4070**: ~3-4 hours = $0.60-1.20

### Factors Affecting Cost:
- GPU type and price
- Training duration
- Instance reliability
- Network usage

## Hydra Configuration Benefits

### Easy Configuration Management:
- **Single File Configs**: Each configuration is in one easy-to-read file
- **Preset Configurations**: Switch between different setups with `--config-name`
- **Parameter Overrides**: Override any parameter from command line
- **Automatic Logging**: Hydra automatically logs all configuration changes

### Example Usage Patterns:
```bash
# Test with fast configuration
python alphaholdem/cli/train_kbest.py --config-name=config_fast

# Use high-performance config but override steps
python alphaholdem/cli/train_kbest.py --config-name=config_high_perf num_steps=1000

# Override multiple parameters
python alphaholdem/cli/train_kbest.py \
    --config-name=config_high_perf \
    train.batch_size=2048 \
    train.learning_rate=1e-5 \
    num_envs=1024

# Override model parameters
python alphaholdem/cli/train_kbest.py \
    --config-name=config \
    model.kwargs.fusion_hidden="[2048, 2048]" \
    train.batch_size=1536
```

### Configuration Validation:
- Hydra validates all configuration parameters
- Type checking ensures correct parameter types
- Missing required parameters are caught early

## Next Steps

1. **Start with fast configuration** (`--config-name=config_fast`) to test setup
2. **Monitor costs** and adjust parameters using Hydra overrides
3. **Scale up** to high-performance configuration for longer runs
4. **Compare results** with local training using different configs
5. **Optimize hyperparameters** by creating custom configurations
6. **Use Hydra's multirun** for hyperparameter sweeps (advanced)

## Support

- **vast.ai Documentation**: [vast.ai/docs](https://vast.ai/docs)
- **PyTorch CUDA**: [pytorch.org/cuda](https://pytorch.org/cuda)
- **Project Issues**: Check your GitHub repository

Happy training! 🚀

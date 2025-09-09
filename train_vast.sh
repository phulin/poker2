# AlphaHoldem Training Script for vast.ai
#!/bin/bash

# Set up environment
export PYTHONPATH=/workspace
export CUDA_VISIBLE_DEVICES=0

# Install dependencies (if not already installed)
if [ ! -d "venv" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    pip install -e .[all]
fi

# Create checkpoint directory
mkdir -p /workspace/checkpoints

# Print GPU info
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Print PyTorch CUDA info
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Training parameters (modify as needed)
STEPS=${STEPS:-2000}
K_BEST_POOL_SIZE=${K_BEST_POOL_SIZE:-10}
MIN_ELO_DIFF=${MIN_ELO_DIFF:-50.0}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
BATCH_SIZE=${BATCH_SIZE:-1024}
NUM_ENVS=${NUM_ENVS:-512}
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-"poker-kbest-vast"}
WANDB_NAME=${WANDB_NAME:-"vast-$(date +%Y%m%d-%H%M%S)"}

echo "=== Training Configuration ==="
echo "Steps: $STEPS"
echo "K-Best Pool Size: $K_BEST_POOL_SIZE"
echo "Min ELO Diff: $MIN_ELO_DIFF"
echo "Checkpoint Interval: $CHECKPOINT_INTERVAL"
echo "Eval Interval: $EVAL_INTERVAL"
echo "Batch Size: $BATCH_SIZE"
echo "Num Envs: $NUM_ENVS"
echo "Wandb Project: $WANDB_PROJECT"
echo "Wandb Name: $WANDB_NAME"
echo ""

# Start training
echo "=== Starting Training ==="
python alphaholdem/cli/train_kbest.py \
    --device cuda \
    --steps $STEPS \
    --k-best-pool-size $K_BEST_POOL_SIZE \
    --min-elo-diff $MIN_ELO_DIFF \
    --checkpoint-interval $CHECKPOINT_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --checkpoint-dir /workspace/checkpoints \
    --use-tensor-env \
    --num-envs $NUM_ENVS \
    --wandb-project $WANDB_PROJECT \
    --wandb-name $WANDB_NAME \
    --wandb-tags vast-ai poker kbest ppo

echo "=== Training Completed ==="

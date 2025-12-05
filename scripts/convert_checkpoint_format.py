#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to TensorFlow.js format.

This script:
1. Loads a PyTorch checkpoint
2. Exports the model to ONNX format
3. Converts ONNX to TensorFlow SavedModel
4. Converts SavedModel to TFJS format
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.onnx
from omegaconf import OmegaConf

# Import from the main package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphaholdem.core.structured_config import Config, NonlinearityType


def export_to_onnx(
    checkpoint_path: str,
    onnx_path: str,
    device: str = "cpu",
    opset_version: int = 15,
) -> None:
    """
    Export PyTorch model from checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint
        onnx_path: Path to save the ONNX model
        device: Device to use for export
        opset_version: ONNX opset version to use
    """
    print(f"\n=== Step 1: Exporting PyTorch model to ONNX ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    device_obj = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location=device_obj
    )

    # Get config from checkpoint
    if "full_config" in checkpoint:
        config = checkpoint["full_config"]
        config.device = device
        config.use_wandb = False
        config.strict_model_loading = True
    elif "config" in checkpoint:
        print("Loading config dict from checkpoint...")
        config_dict = checkpoint["config"]
        config = OmegaConf.create(config_dict)
        config.device = device
        config.use_wandb = False
        config.strict_model_loading = True
    else:
        raise ValueError(
            "Checkpoint must contain 'full_config' or 'config' for proper conversion"
        )

    # Create the model directly
    print("Creating model from checkpoint...")

    # Determine which key has the model state
    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError("Checkpoint must contain 'model' or 'model_state_dict'")

    step = checkpoint.get("step", 0)

    # Import the model class
    from alphaholdem.models.mlp.better_trm import BetterTRM

    # Create model with config params
    model_config = config.model
    num_actions = len(config.env.bet_bins) + 3

    # Infer actual number of layers from checkpoint structure
    trunk_layers = set()
    for k in model_state_dict.keys():
        if k.startswith("trunk.") and ".inner." in k:
            layer_num = k.split(".")[1]
            trunk_layers.add(int(layer_num))
    actual_num_hidden_layers = len(trunk_layers)
    if actual_num_hidden_layers > 0:
        print(f"Detected {actual_num_hidden_layers} hidden layers in checkpoint")

    model = BetterTRM(
        num_actions=num_actions,
        hidden_dim=model_config.get("hidden_dim", 512),
        range_hidden_dim=model_config.get("range_hidden_dim", 256),
        ffn_dim=model_config.get("ffn_dim", 1024),
        num_hidden_layers=(
            actual_num_hidden_layers
            if actual_num_hidden_layers > 0
            else model_config.get("num_hidden_layers", 3)
        ),
        num_policy_layers=model_config.get("num_policy_layers", 1),
        num_value_layers=model_config.get("num_value_layers", 1),
        num_recursions=model_config.get("num_recursions", 6),
        num_iterations=model_config.get("num_iterations", 3),
        shared_trunk=model_config.get("shared_trunk", True),
        enforce_zero_sum=model_config.get("enforce_zero_sum", True),
        nonlinearity=NonlinearityType(model_config.get("nonlinearity", "gelu")),
    )

    # Load the state dict
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device_obj)
    model.eval()

    print(f"✅ Loaded checkpoint at step {step}")

    # Create dummy input for BetterTRM
    print("Creating dummy input for ONNX export...")

    # Import required types
    from alphaholdem.models.mlp.mlp_features import MLPFeatures
    from alphaholdem.models.mlp.better_features import context_length
    from alphaholdem.env.card_utils import NUM_HANDS

    batch_size = 1
    num_players = 2

    # Create dummy MLPFeatures
    dummy_features = MLPFeatures(
        street=torch.zeros(batch_size, dtype=torch.long, device=device_obj),
        to_act=torch.zeros(batch_size, dtype=torch.long, device=device_obj),
        board=torch.zeros(
            batch_size, 5, dtype=torch.long, device=device_obj
        ),  # 5 cards (encoded)
        context=torch.zeros(
            batch_size,
            context_length(num_players),
            dtype=torch.float32,
            device=device_obj,
        ),
        beliefs=torch.ones(
            batch_size, num_players * NUM_HANDS, dtype=torch.float32, device=device_obj
        )
        / NUM_HANDS,
    )

    # Export to ONNX
    print(f"Exporting to ONNX (opset version {opset_version})...")

    # Create a wrapper class that accepts individual tensors instead of MLPFeatures
    class BetterTRMWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, street, to_act, board, context, beliefs):
            features = MLPFeatures(
                street=street,
                to_act=to_act,
                board=board,
                context=context,
                beliefs=beliefs,
            )
            output = self.model(features, latent=None)
            return output.policy_logits, output.value

    wrapper = BetterTRMWrapper(model)
    wrapper.eval()

    # Define input/output names
    input_names = ["street", "to_act", "board", "context", "beliefs"]
    output_names = ["policy_logits", "value"]

    # Dynamic axes for variable batch size
    dynamic_axes = {
        "street": {0: "batch_size"},
        "to_act": {0: "batch_size"},
        "board": {0: "batch_size"},
        "context": {0: "batch_size"},
        "beliefs": {0: "batch_size"},
        "policy_logits": {0: "batch_size"},
        "value": {0: "batch_size"},
    }

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (
                    dummy_features.street,
                    dummy_features.to_act,
                    dummy_features.board,
                    dummy_features.context,
                    dummy_features.beliefs,
                ),
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
            )
        print(f"✅ Successfully exported to ONNX: {onnx_path}")
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")
        import traceback

        traceback.print_exc()
        raise


def onnx_to_savedmodel(onnx_path: str, savedmodel_path: str, venv_python: str) -> None:
    """
    Convert ONNX model to TensorFlow SavedModel format.

    Args:
        onnx_path: Path to the ONNX model
        savedmodel_path: Path to save the TensorFlow SavedModel
        venv_python: Path to the Python interpreter in the conversion venv
    """
    print(f"\n=== Step 2: Converting ONNX to TensorFlow SavedModel ===")
    print(f"Input: {onnx_path}")
    print(f"Output: {savedmodel_path}")

    # Use onnx-tf to convert
    cmd = [
        venv_python,
        "-c",
        f"""
import onnx
from onnx_tf.backend import prepare

# Load ONNX model
onnx_model = onnx.load('{onnx_path}')

# Convert to TensorFlow
tf_rep = prepare(onnx_model)

# Export as SavedModel
tf_rep.export_graph('{savedmodel_path}')
print('✅ Successfully converted to TensorFlow SavedModel')
""",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ONNX to SavedModel conversion:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def savedmodel_to_tfjs(
    savedmodel_path: str, tfjs_path: str, converter_path: str
) -> None:
    """
    Convert TensorFlow SavedModel to TFJS format.

    Args:
        savedmodel_path: Path to the TensorFlow SavedModel
        tfjs_path: Path to save the TFJS model
        converter_path: Path to the tensorflowjs_converter executable
    """
    print(f"\n=== Step 3: Converting SavedModel to TFJS ===")
    print(f"Input: {savedmodel_path}")
    print(f"Output: {tfjs_path}")

    # Create output directory
    os.makedirs(tfjs_path, exist_ok=True)

    # Use tensorflowjs_converter
    cmd = [
        converter_path,
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
        savedmodel_path,
        tfjs_path,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"✅ Successfully converted to TFJS: {tfjs_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during SavedModel to TFJS conversion:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def convert_checkpoint_to_tfjs(
    checkpoint_path: str,
    output_dir: str,
    device: str = "cpu",
    keep_intermediate: bool = False,
) -> None:
    """
    Convert PyTorch checkpoint to TFJS format.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint
        output_dir: Directory to save the output files
        device: Device to use for conversion
        keep_intermediate: Whether to keep intermediate ONNX and SavedModel files
    """
    print(f"Converting checkpoint to TFJS format")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    # Get paths to conversion tools
    script_dir = Path(__file__).parent
    venv_dir = script_dir / "convert_to_tfjs_env" / "venv"
    venv_python = venv_dir / "bin" / "python"
    converter_path = venv_dir / "bin" / "tensorflowjs_converter"

    if not venv_python.exists():
        raise FileNotFoundError(
            f"Conversion venv not found at {venv_dir}. "
            f"Please create it and install: torch, onnx, onnx-tf, tensorflowjs"
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define paths
    onnx_path = os.path.join(output_dir, "model.onnx")
    savedmodel_path = os.path.join(output_dir, "savedmodel")
    tfjs_path = os.path.join(output_dir, "tfjs_model")

    try:
        # Step 1: Export to ONNX
        export_to_onnx(checkpoint_path, onnx_path, device)

        # Step 2: Convert ONNX to SavedModel
        onnx_to_savedmodel(onnx_path, savedmodel_path, str(venv_python))

        # Step 3: Convert SavedModel to TFJS
        savedmodel_to_tfjs(savedmodel_path, tfjs_path, str(converter_path))

        print(f"\n🎉 Conversion complete!")
        print(f"TFJS model saved to: {tfjs_path}")

        # Clean up intermediate files if requested
        if not keep_intermediate:
            print("\nCleaning up intermediate files...")
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
                print(f"Removed: {onnx_path}")
            if os.path.exists(savedmodel_path):
                import shutil

                shutil.rmtree(savedmodel_path)
                print(f"Removed: {savedmodel_path}")

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint to TensorFlow.js format"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the PyTorch checkpoint file",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the converted model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for conversion (default: cpu)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate ONNX and SavedModel files",
    )

    args = parser.parse_args()

    try:
        convert_checkpoint_to_tfjs(
            args.checkpoint,
            args.output_dir,
            args.device,
            args.keep_intermediate,
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

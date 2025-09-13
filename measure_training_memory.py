#!/usr/bin/env python3
"""
Script to measure MPS memory usage during actual training steps.
"""

import time
from typing import Dict, List, Tuple

import psutil
import torch
import torch.nn as nn
import torch.optim as optim

from alphaholdem.models.cnn.siamese_convnet import SiameseConvNetV1
from alphaholdem.models.cnn_embedding_data import CNNEmbeddingData
from alphaholdem.models.transformer.embedding_data import StructuredEmbeddingData
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1


def get_mps_memory_usage() -> float:
    """Get current MPS memory usage in MB."""
    if torch.backends.mps.is_available():
        memory_allocated = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB
        return memory_allocated
    else:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # MB


def clear_mps_cache():
    """Clear MPS cache to get accurate memory measurements."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def create_transformer_data(
    batch_size: int, seq_len: int, device: torch.device
) -> StructuredEmbeddingData:
    """Create dummy structured embedding data for transformer."""
    return StructuredEmbeddingData(
        token_ids=torch.randint(
            0, 100, (batch_size, seq_len), dtype=torch.int8, device=device
        ),
        card_ranks=torch.randint(
            0, 13, (batch_size, seq_len), dtype=torch.uint8, device=device
        ),
        card_suits=torch.randint(
            0, 4, (batch_size, seq_len), dtype=torch.uint8, device=device
        ),
        card_streets=torch.randint(
            0, 4, (batch_size, seq_len), dtype=torch.uint8, device=device
        ),
        action_actors=torch.randint(
            0, 2, (batch_size, seq_len), dtype=torch.uint8, device=device
        ),
        action_streets=torch.randint(
            0, 4, (batch_size, seq_len), dtype=torch.uint8, device=device
        ),
        action_legal_masks=torch.ones(
            batch_size, seq_len, 8, dtype=torch.bool, device=device
        ),
        context_features=torch.randint(
            0, 10, (batch_size, seq_len, 10), dtype=torch.long, device=device
        ),
    )


def create_cnn_data(batch_size: int, device: torch.device) -> CNNEmbeddingData:
    """Create dummy CNN embedding data."""
    cards = torch.zeros(batch_size, 6, 4, 13, dtype=torch.bool, device=device)
    actions = torch.zeros(batch_size, 24, 4, 8, dtype=torch.bool, device=device)
    return CNNEmbeddingData(cards=cards, actions=actions)


def create_dummy_targets(batch_size: int, num_actions: int, device: torch.device):
    """Create dummy targets for training."""
    return {
        "policy_targets": torch.randint(0, num_actions, (batch_size,), device=device),
        "value_targets": torch.randn(batch_size, device=device),
        "legal_masks": torch.ones(
            batch_size, num_actions, dtype=torch.bool, device=device
        ),
    }


def measure_training_memory(
    model: nn.Module,
    data_func,
    batch_sizes: List[int],
    device: torch.device,
    model_name: str,
    num_actions: int = 8,
) -> Dict[int, Dict[str, float]]:
    """Measure memory usage during training for different batch sizes."""
    print(f"\n📊 Measuring {model_name} training memory usage...")

    memory_usage = {}

    for batch_size in batch_sizes:
        print(f"  Batch size: {batch_size:3d}", end=" ")

        # Clear cache before measurement
        clear_mps_cache()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Get baseline memory
        baseline_memory = get_mps_memory_usage()

        try:
            # Create optimizer
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            # Create data and targets
            data = data_func(batch_size, device)
            targets = create_dummy_targets(batch_size, num_actions, device)

            # Forward pass with autocast
            memory_after_forward = get_mps_memory_usage()

            # Use autocast for mixed precision (both CUDA and MPS support it)
            use_autocast = (
                torch.cuda.is_available() or torch.backends.mps.is_available()
            )

            if torch.cuda.is_available():
                scaler = torch.cuda.amp.GradScaler()
                autocast_context = torch.cuda.amp.autocast()
            elif torch.backends.mps.is_available():
                scaler = None  # MPS doesn't need scaler
                autocast_context = torch.autocast("mps", dtype=torch.float16)
            else:
                scaler = None
                autocast_context = None

            if use_autocast:
                with autocast_context:
                    outputs = model(data)

                    # Compute loss
                    policy_logits = outputs["policy_logits"]
                    values = outputs["value"]

                    # Simple policy loss (cross entropy)
                    policy_loss = nn.CrossEntropyLoss()(
                        policy_logits, targets["policy_targets"]
                    )

                    # Simple value loss (MSE)
                    value_loss = nn.MSELoss()(values, targets["value_targets"])

                    total_loss = policy_loss + value_loss
            else:
                # No autocast for MPS
                outputs = model(data)

                # Compute loss
                policy_logits = outputs["policy_logits"]
                values = outputs["value"]

                # Simple policy loss (cross entropy)
                policy_loss = nn.CrossEntropyLoss()(
                    policy_logits, targets["policy_targets"]
                )

                # Simple value loss (MSE)
                value_loss = nn.MSELoss()(values, targets["value_targets"])

                total_loss = policy_loss + value_loss

            memory_after_loss = get_mps_memory_usage()

            # Backward pass with scaler (only for CUDA)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            memory_after_backward = get_mps_memory_usage()
            memory_after_step = get_mps_memory_usage()

            # Calculate memory usage at each stage
            forward_memory = memory_after_forward - baseline_memory
            loss_memory = memory_after_loss - baseline_memory
            backward_memory = memory_after_backward - baseline_memory
            step_memory = memory_after_step - baseline_memory

            memory_usage[batch_size] = {
                "forward": forward_memory,
                "loss": loss_memory,
                "backward": backward_memory,
                "step": step_memory,
                "total": step_memory,
            }

            print(
                f"→ Forward: {forward_memory:5.2f}MB, Backward: {backward_memory:5.2f}MB, Total: {step_memory:5.2f}MB"
            )

            # Clean up
            del data, targets, outputs, optimizer, total_loss, policy_loss, value_loss
            clear_mps_cache()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"→ FAILED: {e}")
            memory_usage[batch_size] = {
                "forward": float("inf"),
                "loss": float("inf"),
                "backward": float("inf"),
                "step": float("inf"),
                "total": float("inf"),
            }

    return memory_usage


def print_training_memory_summary(results: Dict[str, Dict[int, Dict[str, float]]]):
    """Print a formatted training memory usage summary."""
    print("\n" + "=" * 80)
    print("📋 TRAINING MEMORY USAGE SUMMARY")
    print("=" * 80)

    for model_name, memory_data in results.items():
        print(f"\n{model_name}:")
        print("-" * 60)

        valid_data = {
            k: v for k, v in memory_data.items() if v["total"] != float("inf")
        }
        if not valid_data:
            print("  No successful measurements")
            continue

        # Print header
        print(
            f"{'Batch':>6} {'Forward':>8} {'Loss':>8} {'Backward':>8} {'Total':>8} {'Peak':>8}"
        )
        print("-" * 60)

        # Print data
        for batch_size in sorted(valid_data.keys()):
            data = valid_data[batch_size]
            forward = data["forward"]
            loss = data["loss"]
            backward = data["backward"]
            total = data["total"]
            peak = max(forward, loss, backward, total)

            print(
                f"{batch_size:6d} {forward:8.2f} {loss:8.2f} {backward:8.2f} {total:8.2f} {peak:8.2f}"
            )

        # Calculate scaling
        batch_sizes = sorted(valid_data.keys())
        if len(batch_sizes) >= 2:
            first_batch, first_memory = (
                batch_sizes[0],
                valid_data[batch_sizes[0]]["total"],
            )
            last_batch, last_memory = (
                batch_sizes[-1],
                valid_data[batch_sizes[-1]]["total"],
            )

            if first_memory > 0:
                scaling_factor = (last_memory / first_memory) / (
                    last_batch / first_batch
                )
                print(f"\n  Scaling factor: {scaling_factor:.2f}x (linear = 1.0x)")

    print("\n" + "=" * 80)


def main():
    """Main function to run training memory measurements."""
    print("🚀 MPS Training Memory Usage Measurement Script")
    print("=" * 60)

    # Check for autocast support
    if torch.cuda.is_available():
        print("✅ CUDA available - Using CUDA autocast for mixed precision")
    elif torch.backends.mps.is_available():
        print("✅ MPS available - Using MPS autocast for mixed precision")
    else:
        print("⚠️  No GPU available - No autocast support")

    print("✅ Gradient checkpointing enabled for memory efficiency")

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("❌ MPS is not available on this system!")
        print("   Falling back to CPU memory measurements...")
        device = torch.device("cpu")
    else:
        print("✅ MPS is available!")
        device = torch.device("mps")

    # Define batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Initialize results dictionary
    results = {}

    # Test Transformer Model
    print("\n🔧 Setting up Transformer model...")
    try:
        transformer_model = PokerTransformerV1(
            d_model=256,
            n_layers=4,
            n_heads=4,
            vocab_size=80,
            num_actions=8,
            use_auxiliary_loss=False,  # Disable to reduce memory
            use_gradient_checkpointing=True,  # Enable gradient checkpointing
        ).to(device)

        # Set to training mode for gradient checkpointing
        transformer_model.train()

        def transformer_data_func(batch_size, device):
            return create_transformer_data(batch_size, seq_len=42, device=device)

        transformer_memory = measure_training_memory(
            transformer_model, transformer_data_func, batch_sizes, device, "Transformer"
        )
        results["Transformer"] = transformer_memory

        # Clean up
        del transformer_model
        clear_mps_cache()

    except Exception as e:
        print(f"❌ Transformer model failed: {e}")
        results["Transformer"] = {}

    # Test CNN Model
    print("\n🔧 Setting up CNN model...")
    try:
        cnn_model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=256,
            actions_hidden=256,
            fusion_hidden=512,
            num_actions=8,
            use_gradient_checkpointing=False,  # Disable to reduce memory
        ).to(device)

        def cnn_data_func(batch_size, device):
            return create_cnn_data(batch_size, device)

        cnn_memory = measure_training_memory(
            cnn_model, cnn_data_func, batch_sizes, device, "CNN"
        )
        results["CNN"] = cnn_memory

        # Clean up
        del cnn_model
        clear_mps_cache()

    except Exception as e:
        print(f"❌ CNN model failed: {e}")
        results["CNN"] = {}

    # Print summary
    print_training_memory_summary(results)

    print("\n✅ Training memory measurement complete!")
    print("\n💡 Training Memory Insights:")
    print("   - Forward pass: Model computation + activations")
    print("   - Loss computation: Additional intermediate tensors")
    print("   - Backward pass: Gradients + intermediate computations")
    print("   - Optimizer step: Parameter updates")
    print("   - Peak memory typically occurs during backward pass")


if __name__ == "__main__":
    main()

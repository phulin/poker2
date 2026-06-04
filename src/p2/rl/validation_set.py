from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.rebel_solved_dataset import RebelSolvedDataset


def dataset_manifest(dataset_path: str) -> dict[str, Any]:
    return json.loads((Path(dataset_path) / "manifest.json").read_text())


class RebelValueValidationSetEvaluator:
    """Evaluate a ReBeL trainer's value head against a solved value dataset."""

    def __init__(
        self,
        *,
        trainer: RebelCFRTrainer,
        cfg: Config,
        dataset_path: str,
        batch_size: int,
        max_examples: int | None = None,
    ) -> None:
        self.trainer = trainer
        self.cfg = cfg
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_examples = max_examples

        manifest = dataset_manifest(dataset_path)
        self.dataset = RebelSolvedDataset(
            dataset_path,
            num_players=trainer.num_players,
            num_actions=trainer.num_actions,
            context_length=int(manifest["context_length"]),
            model_name=(
                cfg.model.name.value
                if hasattr(cfg.model.name, "value")
                else str(cfg.model.name)
            ),
            action_schedule={
                "bet_bins": list(cfg.env.bet_bins),
                "bet_bins_by_depth": cfg.search.bet_bins_by_depth,
                "allin_by_depth": cfg.search.allin_by_depth,
            },
            street_support=[0, 1, 2, 3],
        )
        self.total_examples = self.dataset.stream_len("value")

    @torch.no_grad()
    def evaluate(self) -> dict[str, float | int | str]:
        total_examples = self.total_examples
        if self.max_examples is not None:
            total_examples = min(total_examples, self.max_examples)
        if total_examples <= 0:
            raise ValueError(f"Validation dataset has no value examples: {self.dataset_path}")

        weighted_loss_sum = 0.0
        weighted_loss_elements = 0
        weight_normalized_denom = 0.0
        batch_loss_sum = 0.0
        num_batches = 0

        for start in range(0, total_examples, self.batch_size):
            count = min(self.batch_size, total_examples - start)
            batch = self.dataset.get_batch(
                "value",
                start,
                count,
                device=self.trainer.device,
                float_dtype=self.trainer.float_dtype,
            )
            with self.trainer._model_autocast():
                output = self.trainer.inference_model.repeat(
                    batch.features,
                    count=self.cfg.model.num_supervisions,
                    include_policy=False,
                )
            loss_dict = self.trainer.loss_fn.forward_value(output, batch)
            value_loss_all = loss_dict["value_loss_all"]
            value_weights = loss_dict["value_weights"]

            weighted_loss_sum += float(value_loss_all.sum().detach().cpu().item())
            weighted_loss_elements += int(value_loss_all.numel())
            weight_normalized_denom += float(value_weights.sum().detach().cpu().item())
            batch_loss_sum += float(loss_dict["value_loss"].detach().cpu().item())
            num_batches += 1

        value_loss = weighted_loss_sum / max(weight_normalized_denom, 1e-12)
        return {
            "validation_value_loss": value_loss,
            "validation_element_mean_weighted_square_error": (
                weighted_loss_sum / max(weighted_loss_elements, 1)
            ),
            "validation_batch_loss_mean": batch_loss_sum / max(num_batches, 1),
            "validation_examples": total_examples,
            "validation_num_batches": num_batches,
            "validation_dataset": self.dataset_path,
        }


__all__ = [
    "RebelValueValidationSetEvaluator",
    "dataset_manifest",
]

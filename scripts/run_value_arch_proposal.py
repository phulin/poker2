#!/usr/bin/env python3
"""Run one value-only S_river architecture proposal on pregenerated data."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_COMPILE_CACHE_ROOT = Path.cwd() / "outputs" / "torch_compile_cache"
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", str(_DEFAULT_COMPILE_CACHE_ROOT / "inductor")
)
os.environ.setdefault("TRITON_CACHE_DIR", str(_DEFAULT_COMPILE_CACHE_ROOT / "triton"))
os.environ.setdefault("CUDA_CACHE_PATH", str(_DEFAULT_COMPILE_CACHE_ROOT / "cuda"))

import torch

from p2.config.rebel_load import load_rebel_config_file
from p2.core.structured_config import PregeneratedDatasetConfig
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_loop import run_training_loop
from p2.search.rebel_solved_dataset import rebel_batch_from_tensors
from p2.rl.validation_set import RebelValueValidationSetEvaluator


DEFAULT_DATASET_MANIFEST = Path(
    "outputs/rebel_postflop/river_value_100steps_102400_300it_20260630/manifest.json"
)
DEFAULT_VALIDATION_DATASET = (
    "outputs/rebel_postflop/river_val_8192_10k_sapdcfr_nowarm_ctx41_20260630"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/value_arch_proposals_100step_20260630")


PROPOSALS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "residual": {"value_per_hand_residual": True},
    "board_embed": {"board_conditioned_hand_embedding_dim": 8},
    "cross_range": {"cross_range_rank": 64},
    "card_token": {"card_token_value_head_dim": 128},
    "range_stats": {"context_range_stats": True},
    "multi_token": {"postflop_multi_token_trunk": True},
    "second_moment": {"belief_second_moment": True},
    "bet_strat": {"value_strength_bucket_count": 4},
    "bet_strat_film": {
        "value_strength_bucket_count": 4,
        "value_strength_bucket_film": True,
    },
    "bet_strat_relative": {
        "value_strength_bucket_count": 4,
        "value_strength_bucket_relative": True,
    },
    "bet_strat_k8": {"value_strength_bucket_count": 8},
    "bet_strat_film_relative": {
        "value_strength_bucket_count": 4,
        "value_strength_bucket_film": True,
        "value_strength_bucket_relative": True,
    },
    "river_showdown_baseline": {
        "value_exact_river_features": True,
        "value_showdown_baseline": True,
    },
    "river_exact_features": {"value_exact_river_features": True},
    "river_range_equity_baseline_c050": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.50,
    },
    "river_range_equity_baseline_c065": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
    },
    "river_range_equity_baseline_c075": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.75,
    },
    "river_range_equity_baseline_c100": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 1.00,
    },
    "river_range_equity_action_c065": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.05},
    },
    "river_range_equity_blockers_c050": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.50,
        "value_river_range_equity_blockers": True,
    },
    "river_range_equity_blockers_c065": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_river_range_equity_blockers": True,
    },
    "river_range_equity_blockers_c065_r96": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
    },
    "river_range_equity_blockers_posneg_r96": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
    },
    "river_range_equity_blockers_posneg_r96_lr1p5": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "train": {
            "learning_rate": 0.06,
            "learning_rate_final": 0.006,
            "adamw_learning_rate": 0.06,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p7": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "train": {
            "learning_rate": 0.028,
            "learning_rate_final": 0.0028,
            "adamw_learning_rate": 0.028,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p04_out0p00": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_output_init_scale": 0.0,
        "train": {
            "learning_rate": 0.04,
            "learning_rate_final": 0.004,
            "adamw_learning_rate": 0.04,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p04_out0p03": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_output_init_scale": 0.03,
        "train": {
            "learning_rate": 0.04,
            "learning_rate_final": 0.004,
            "adamw_learning_rate": 0.04,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p06_out0p00": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_output_init_scale": 0.0,
        "train": {
            "learning_rate": 0.06,
            "learning_rate_final": 0.006,
            "adamw_learning_rate": 0.06,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p06_out0p03": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_output_init_scale": 0.03,
        "train": {
            "learning_rate": 0.06,
            "learning_rate_final": 0.006,
            "adamw_learning_rate": 0.06,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p08": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "train": {
            "learning_rate": 0.08,
            "learning_rate_final": 0.008,
            "adamw_learning_rate": 0.08,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p08_out0p00": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_output_init_scale": 0.0,
        "train": {
            "learning_rate": 0.08,
            "learning_rate_final": 0.008,
            "adamw_learning_rate": 0.08,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p08_out0p03": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_output_init_scale": 0.03,
        "train": {
            "learning_rate": 0.08,
            "learning_rate_final": 0.008,
            "adamw_learning_rate": 0.08,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p08_out0p10": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_output_init_scale": 0.1,
        "train": {
            "learning_rate": 0.08,
            "learning_rate_final": 0.008,
            "adamw_learning_rate": 0.08,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p08_out0p30": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_output_init_scale": 0.3,
        "train": {
            "learning_rate": 0.08,
            "learning_rate_final": 0.008,
            "adamw_learning_rate": 0.08,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p08": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "train": {
            "learning_rate": 0.08,
            "learning_rate_final": 0.008,
            "adamw_learning_rate": 0.08,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p06": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "train": {
            "learning_rate": 0.06,
            "learning_rate_final": 0.006,
            "adamw_learning_rate": 0.06,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p06_out0p00": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "value_output_init_scale": 0.0,
        "train": {
            "learning_rate": 0.06,
            "learning_rate_final": 0.006,
            "adamw_learning_rate": 0.06,
        },
    },
    "river_range_equity_film_r8_posneg_r96_lr0p06_out0p00": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "value_river_range_equity_film_rank": 8,
        "value_river_range_equity_film_hidden_dim": 16,
        "value_output_init_scale": 0.0,
        "train": {
            "learning_rate": 0.06,
            "learning_rate_final": 0.006,
            "adamw_learning_rate": 0.06,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p06_out0p03": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "value_output_init_scale": 0.03,
        "train": {
            "learning_rate": 0.06,
            "learning_rate_final": 0.006,
            "adamw_learning_rate": 0.06,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p06_out0p10": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "value_output_init_scale": 0.1,
        "train": {
            "learning_rate": 0.06,
            "learning_rate_final": 0.006,
            "adamw_learning_rate": 0.06,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p06_out0p30": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "value_output_init_scale": 0.3,
        "train": {
            "learning_rate": 0.06,
            "learning_rate_final": 0.006,
            "adamw_learning_rate": 0.06,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p04": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "train": {
            "learning_rate": 0.04,
            "learning_rate_final": 0.004,
            "adamw_learning_rate": 0.04,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p04_out0p00": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "value_output_init_scale": 0.0,
        "train": {
            "learning_rate": 0.04,
            "learning_rate_final": 0.004,
            "adamw_learning_rate": 0.04,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p04_out0p03": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "value_output_init_scale": 0.03,
        "train": {
            "learning_rate": 0.04,
            "learning_rate_final": 0.004,
            "adamw_learning_rate": 0.04,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p02": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "train": {
            "learning_rate": 0.02,
            "learning_rate_final": 0.002,
            "adamw_learning_rate": 0.02,
        },
    },
    "river_range_equity_inputs_posneg_r96_lr0p10": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_river_range_equity_feature_head": True,
        "value_river_range_equity_trunk_context": True,
        "train": {
            "learning_rate": 0.10,
            "learning_rate_final": 0.010,
            "adamw_learning_rate": 0.10,
        },
    },
    "river_range_equity_blockers_posneg_r96_lr0p10": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_pos_scale": 0.8543022528460094,
        "value_river_range_equity_neg_scale": 0.4753640305061305,
        "value_river_range_equity_intercept": -0.010797645393242563,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "train": {
            "learning_rate": 0.10,
            "learning_rate_final": 0.010,
            "adamw_learning_rate": 0.10,
        },
    },
    "river_range_equity_blockers_c065_r96_potp05": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_river_range_equity_pot_power": 0.5,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
    },
    "river_range_equity_blockers_c080_r96_potp05": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.80,
        "value_river_range_equity_pot_power": 0.5,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
    },
    "river_range_equity_blockers_c100_r96_potp05": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 1.00,
        "value_river_range_equity_pot_power": 0.5,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
    },
    "river_range_equity_blockers_c065_r96_potp075": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_river_range_equity_pot_power": 0.75,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
    },
    "river_range_equity_blockers_c080_r96_potp075": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.80,
        "value_river_range_equity_pot_power": 0.75,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
    },
    "river_range_equity_blockers_c100_r96_potp075": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 1.00,
        "value_river_range_equity_pot_power": 0.75,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
    },
    "river_range_equity_blockers_exact_c065": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 1326,
    },
    "river_range_equity_blockers_exact_c100": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 1.00,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 1326,
    },
    "river_range_equity_blockers_c065_r64": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 64,
    },
    "river_range_equity_blockers_c075": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.75,
        "value_river_range_equity_blockers": True,
    },
    "river_range_equity_blockers_action_c065": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_river_range_equity_blockers": True,
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.05},
    },
    "river_range_equity_blockers_action_c065_r96": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 96,
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.05},
    },
    "river_range_equity_blockers_action_c065_r64": {
        "value_river_range_equity_baseline": True,
        "value_river_range_equity_baseline_scale": 0.65,
        "value_river_range_equity_blockers": True,
        "value_river_range_equity_rank_bins": 64,
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.05},
    },
    "river_multiresolution": {
        "train": {"value_multiresolution_coef": 0.25},
    },
    "river_strength_bucket_aux": {
        "value_strength_bucket_count": 8,
        "value_strength_bucket_relative": True,
        "value_strength_bucket_board_only": True,
        "train": {"value_bucket_aux_coef": 0.25},
    },
    "river_ordering": {
        "value_exact_river_features": True,
        "train": {"value_ordering_coef": 0.05},
    },
    "river_bucket_calibration": {
        "value_strength_bucket_count": 8,
        "value_strength_bucket_relative": True,
        "value_strength_bucket_board_only": True,
        "train": {
            "value_bucket_usage_coef": 0.01,
            "value_bucket_entropy_coef": 0.001,
        },
    },
    "river_bucket_calibration_fast": {
        "value_strength_bucket_count": 8,
        "value_strength_bucket_relative": True,
        "value_strength_bucket_board_only": True,
        "value_strength_bucket_blockers": False,
        "train": {
            "value_bucket_usage_coef": 0.01,
            "value_bucket_entropy_coef": 0.001,
        },
    },
    "river_bucket_calibration_projected": {
        "value_strength_bucket_count": 8,
        "value_strength_bucket_relative": True,
        "value_strength_bucket_board_only": True,
        "value_strength_bucket_blockers": False,
        "value_strength_bucket_coarse_residual": True,
        "train": {
            "value_bucket_usage_coef": 0.01,
            "value_bucket_entropy_coef": 0.001,
        },
    },
    "river_bucket_coarse_head": {
        "value_strength_bucket_count": 8,
        "value_strength_bucket_relative": True,
        "value_strength_bucket_board_only": True,
        "value_bucket_coarse_dim": 64,
    },
    "river_bucket_coarse_head_fast": {
        "value_strength_bucket_count": 8,
        "value_strength_bucket_relative": True,
        "value_strength_bucket_board_only": True,
        "value_strength_bucket_blockers": False,
        "value_bucket_coarse_dim": 64,
    },
    "river_bucket_coarse_head_projected": {
        "value_strength_bucket_count": 8,
        "value_strength_bucket_relative": True,
        "value_strength_bucket_board_only": True,
        "value_strength_bucket_blockers": False,
        "value_strength_bucket_coarse_residual": True,
        "value_bucket_coarse_dim": 64,
    },
    "river_latent_bucket_k8": {
        "value_latent_bucket_count": 8,
        "value_latent_bucket_dim": 32,
    },
    "river_latent_bucket_k16": {
        "value_latent_bucket_count": 16,
        "value_latent_bucket_dim": 32,
    },
    "river_latent_bucket_aux_k8": {
        "value_latent_bucket_count": 8,
        "value_latent_bucket_dim": 32,
        "train": {"value_bucket_aux_coef": 0.25},
    },
    "river_latent_bucket_calibration_k8": {
        "value_latent_bucket_count": 8,
        "value_latent_bucket_dim": 32,
        "train": {
            "value_bucket_usage_coef": 0.01,
            "value_bucket_entropy_coef": 0.001,
        },
    },
    "river_latent_bucket_action_k8": {
        "value_latent_bucket_count": 8,
        "value_latent_bucket_dim": 32,
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.05},
    },
    "river_action_summary": {
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.05},
    },
    "river_action_summary_c001": {
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.01},
    },
    "river_action_summary_c0025": {
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.025},
    },
    "river_action_summary_c010": {
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.10},
    },
    "river_action_summary_c020": {
        "value_action_summary_head": True,
        "train": {"value_action_summary_coef": 0.20},
    },
    "river_aux_combo": {
        "value_exact_river_features": True,
        "value_strength_bucket_count": 8,
        "value_strength_bucket_relative": True,
        "value_strength_bucket_board_only": True,
        "value_bucket_coarse_dim": 64,
        "value_action_summary_head": True,
        "train": {
            "value_multiresolution_coef": 0.25,
            "value_bucket_aux_coef": 0.25,
            "value_ordering_coef": 0.05,
            "value_bucket_usage_coef": 0.01,
            "value_bucket_entropy_coef": 0.001,
            "value_action_summary_coef": 0.05,
        },
    },
    "river_aux_combo_projected": {
        "value_exact_river_features": True,
        "value_strength_bucket_count": 8,
        "value_strength_bucket_relative": True,
        "value_strength_bucket_board_only": True,
        "value_strength_bucket_blockers": False,
        "value_strength_bucket_coarse_residual": True,
        "value_bucket_coarse_dim": 64,
        "value_action_summary_head": True,
        "train": {
            "value_multiresolution_coef": 0.25,
            "value_bucket_aux_coef": 0.25,
            "value_ordering_coef": 0.05,
            "value_bucket_usage_coef": 0.01,
            "value_bucket_entropy_coef": 0.001,
            "value_action_summary_coef": 0.05,
        },
    },
    "flops_value_lr512": {"value_head_rank": 512},
    "flops_value_lr192": {"value_head_rank": 192},
    "flops_value_lr256": {"value_head_rank": 256},
    "flops_value_lr256_bet_film": {
        "value_head_rank": 256,
        "value_strength_bucket_count": 4,
        "value_strength_bucket_film": True,
    },
    "flops_hand_basis_r256": {"value_hand_basis_rank": 256},
    "flops_hand_basis_r384": {"value_hand_basis_rank": 384},
    "flops_hand_basis_r512": {"value_hand_basis_rank": 512},
    "flops_belief_low256": {"belief_low_rank_dim": 256},
    "flops_belief_in128": {"belief_low_rank_dim": 128},
    "flops_belief_in128_board": {
        "belief_low_rank_dim": 128,
        "belief_low_rank_board_conditioned": True,
    },
    "flops_belief_in128_second": {
        "belief_low_rank_dim": 128,
        "belief_second_moment": True,
    },
    "flops_belief_in128_board_second": {
        "belief_low_rank_dim": 128,
        "belief_low_rank_board_conditioned": True,
        "belief_second_moment": True,
    },
    "flops_belief_in96": {"belief_low_rank_dim": 96},
    "flops_belief_in96_board": {
        "belief_low_rank_dim": 96,
        "belief_low_rank_board_conditioned": True,
    },
    "flops_belief_in96_second": {
        "belief_low_rank_dim": 96,
        "belief_second_moment": True,
    },
    "flops_belief_in96_second_skip_encoder": {
        "belief_low_rank_dim": 96,
        "belief_second_moment": True,
        "belief_skip_matching_encoder": True,
    },
    "flops_belief_in96_board_second": {
        "belief_low_rank_dim": 96,
        "belief_low_rank_board_conditioned": True,
        "belief_second_moment": True,
    },
    "flops_value_lr256_residual": {
        "value_head_rank": 256,
        "value_per_hand_residual": True,
    },
    "flops_board_basis_r256": {
        "value_hand_basis_rank": 256,
        "board_conditioned_hand_embedding_dim": 8,
    },
    "flops_thin_value_tower": {"num_value_layers": 1},
    "flops_hidden512": {"hidden_dim": 512},
    "flops_hidden512_ffn1024": {"hidden_dim": 512, "ffn_dim": 1024},
    "flops_hidden512_ffn1024_value2": {
        "hidden_dim": 512,
        "ffn_dim": 1024,
        "num_value_layers": 2,
    },
    "flops_hidden512_ffn1024_value3": {
        "hidden_dim": 512,
        "ffn_dim": 1024,
        "num_value_layers": 3,
    },
    "flops_hidden512_ffn1024_value4": {
        "hidden_dim": 512,
        "ffn_dim": 1024,
        "num_value_layers": 4,
    },
    "flops_hidden512_ffn1024_value5": {
        "hidden_dim": 512,
        "ffn_dim": 1024,
        "num_value_layers": 5,
    },
    "flops_hidden512_ffn1024_value6": {
        "hidden_dim": 512,
        "ffn_dim": 1024,
        "num_value_layers": 6,
    },
    "flops_hidden512_ffn1024_value6_board16": {
        "hidden_dim": 512,
        "ffn_dim": 1024,
        "num_value_layers": 6,
        "board_interaction_dim": 16,
    },
    "flops_hidden512_ffn1024_value6_board128": {
        "hidden_dim": 512,
        "ffn_dim": 1024,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
    },
    "flops_hidden384_ffn768_value6_board96_belief128": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 96,
        "belief_low_rank_dim": 128,
    },
    "flops_hidden384_ffn768_value6_board128": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
    },
    "flops_hidden384_ffn768_value6_board128_belief96": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
    },
    "flops_hidden384_ffn640_value6_board128_belief128": {
        "hidden_dim": 384,
        "ffn_dim": 640,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 128,
    },
    "flops_hidden384_ffn896_value6_board128_belief128": {
        "hidden_dim": 384,
        "ffn_dim": 896,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 128,
    },
    "flops_hidden384_ffn768_value6_board128_belief128_handbasis384": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 128,
        "value_hand_basis_rank": 384,
    },
    "flops_hidden384_ffn768_value6_board128_belief128_handbasis512": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 128,
        "value_hand_basis_rank": 512,
    },
    "flops_hidden384_ffn768_value6_board64gated_belief128": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 64,
        "board_interaction_gated": True,
        "belief_low_rank_dim": 128,
    },
    "flops_hidden384_ffn768_value6_board128gated_belief128": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "board_interaction_gated": True,
        "belief_low_rank_dim": 128,
    },
    "flops_hidden384_ffn768_value6_board128_belief192": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 192,
    },
    "flops_hidden384_ffn768_value6_board128_belief192_skip": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 192,
        "belief_skip_matching_encoder": True,
    },
    "flops_hidden384_ffn768_value6_board192skip_belief128": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 192,
        "board_interaction_skip_out": True,
        "belief_low_rank_dim": 128,
    },
    "flops_hidden384_ffn768_value6_board192skip_belief192_skip": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 192,
        "board_interaction_skip_out": True,
        "belief_low_rank_dim": 192,
        "belief_skip_matching_encoder": True,
    },
    "flops_hidden384_ffn768_value6_board128_belief64_boardcond": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 64,
        "belief_low_rank_board_conditioned": True,
    },
    "flops_hidden384_ffn768_value6_board128_belief96_boardcond": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_low_rank_board_conditioned": True,
    },
    "flops_hidden384_ffn768_value6_board128_belief128_boardcond": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 128,
        "belief_low_rank_board_conditioned": True,
    },
    "flops_hidden384_ffn576_value6_board128_belief96_boardcond": {
        "hidden_dim": 384,
        "ffn_dim": 576,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_low_rank_board_conditioned": True,
    },
    "flops_hidden320_ffn640_value6_board128_belief96_boardcond": {
        "hidden_dim": 320,
        "ffn_dim": 640,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_low_rank_board_conditioned": True,
    },
    "flops_hidden384_ffn768_value6_board128_belief96_boardcond_linear": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_low_rank_board_conditioned": True,
        "belief_linear_encoder": True,
    },
    "flops_hidden384_ffn768_value6_board128_belief96_boardfilm": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_board_film": True,
    },
    "flops_hidden384_ffn768_value6_board128_belief96_boardbilin32": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_board_bilinear_rank": 32,
    },
    "flops_hidden384_ffn768_value6_board128_belief96_boardbilin64": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_board_bilinear_rank": 64,
    },
    "flops_hidden384_ffn768_value6_board128_belief96_boardmass": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_board_mass_features": True,
    },
    "flops_hidden384_ffn768_value6_board128_belief96_boardfilm_mass": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_board_film": True,
        "belief_board_mass_features": True,
    },
    "flops_hidden384_ffn768_range0_value6_board128": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "range_hidden_dim": 0,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
    },
    "flops_hidden384_ffn768_range0_value6_board128_belief96": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "range_hidden_dim": 0,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
    },
    "flops_hidden384_ffn576_value6_board128_belief96": {
        "hidden_dim": 384,
        "ffn_dim": 576,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
    },
    "flops_hidden384_ffn512_value6_board128_belief96": {
        "hidden_dim": 384,
        "ffn_dim": 512,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
    },
    "flops_hidden352_ffn704_value6_board128_belief96": {
        "hidden_dim": 352,
        "ffn_dim": 704,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
    },
    "flops_hidden384_ffn768_value6_board128_belief64": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 64,
    },
    "flops_hidden384_ffn768_value6_board128_belief128": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 128,
    },
    "flops_hidden384_ffn768_value6_board128_belief96_linear": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "belief_linear_encoder": True,
    },
    "flops_hidden384_ffn768_value6_board128_linear": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_linear_encoder": True,
    },
    "flops_hidden384_ffn768_value6_board128_belief96_handcond8": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
        "board_conditioned_hand_embedding_dim": 8,
    },
    "flops_hidden384_ffn768_value6_handcond8": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 6,
        "board_conditioned_hand_embedding_dim": 8,
    },
    "flops_hidden384_ffn768_value5_board128": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 5,
        "board_interaction_dim": 128,
    },
    "flops_hidden384_ffn768_value5_board128_belief96": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 5,
        "board_interaction_dim": 128,
        "belief_low_rank_dim": 96,
    },
    "flops_hidden384_ffn768_value4_board128": {
        "hidden_dim": 384,
        "ffn_dim": 768,
        "num_value_layers": 4,
        "board_interaction_dim": 128,
    },
    "flops_hidden512_ffn1024_value8": {
        "hidden_dim": 512,
        "ffn_dim": 1024,
        "num_value_layers": 8,
    },
    "flops_hidden320_ffn640": {"hidden_dim": 320, "ffn_dim": 640},
    "flops_hidden320_ffn640_value6_board128": {
        "hidden_dim": 320,
        "ffn_dim": 640,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
    },
    "flops_hidden320_ffn640_value5_board128": {
        "hidden_dim": 320,
        "ffn_dim": 640,
        "num_value_layers": 5,
        "board_interaction_dim": 128,
    },
    "flops_hidden320_ffn640_value4_board128": {
        "hidden_dim": 320,
        "ffn_dim": 640,
        "num_value_layers": 4,
        "board_interaction_dim": 128,
    },
    "flops_hidden256_ffn512_value6_board128": {
        "hidden_dim": 256,
        "ffn_dim": 512,
        "num_value_layers": 6,
        "board_interaction_dim": 128,
    },
    "flops_hidden256_ffn512": {"hidden_dim": 256, "ffn_dim": 512},
    "flops_hidden256_ffn512_value4": {
        "hidden_dim": 256,
        "ffn_dim": 512,
        "num_value_layers": 4,
    },
    "flops_value_layers5": {"num_value_layers": 5},
}


def _dataset_dir(path: Path) -> Path:
    return path.parent if path.name == "manifest.json" else path


def _load_manifest(dataset_dir: Path) -> dict[str, Any]:
    return json.loads((dataset_dir / "manifest.json").read_text())


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _copy_payload_to_full_tensors(
    full_tensors: dict[str, torch.Tensor],
    payload: dict[str, torch.Tensor],
    *,
    start: int,
    end: int,
    device: torch.device,
) -> int:
    copied_bytes = 0
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"expected tensor payload for {key!r}, got {type(value)}")
        if key not in full_tensors:
            raise KeyError(f"unexpected tensor key {key!r} in solved-data shard")
        if int(value.shape[0]) != end - start:
            raise ValueError(
                f"shard key {key!r} has leading dim {value.shape[0]}, "
                f"expected {end - start}"
            )
        full_tensors[key][start:end].copy_(
            value.to(device=device, non_blocking=True),
            non_blocking=True,
        )
        copied_bytes += _tensor_bytes(value)
    return copied_bytes


@dataclass(frozen=True)
class GpuValueEpoch:
    batch: RebelBatch
    batch_size: int
    shuffle_seed: int
    examples: int
    tensor_bytes: int
    load_time_s: float

    def step_batch(self, step: int) -> RebelBatch:
        start = step * self.batch_size
        end = start + self.batch_size
        if end > self.examples:
            raise IndexError(
                f"GPU value epoch exhausted at step {step}: requested [{start}, {end}), "
                f"but epoch has {self.examples} examples"
            )
        return self.batch[start:end]

    def slice_batch(self, start: int, count: int) -> RebelBatch:
        end = start + count
        if start < 0 or count <= 0 or end > self.examples:
            raise IndexError(
                f"GPU value epoch slice [{start}, {end}) is outside "
                f"{self.examples} examples"
            )
        return self.batch[start:end]


def _load_gpu_value_epoch(
    *,
    dataset_dir: Path,
    manifest: dict[str, Any],
    device: torch.device,
    batch_size: int,
    steps: int,
    shuffle_seed: int,
    shuffle: bool,
) -> GpuValueEpoch:
    if device.type != "cuda":
        raise ValueError("GPU value epoch loading requires cfg.device to be a CUDA device")
    value_shards = list(manifest.get("shards", {}).get("value", []))
    if not value_shards:
        raise ValueError(f"dataset {dataset_dir} has no value shards")
    examples = int(manifest["value_examples"])
    requested_examples = int(batch_size) * int(steps)
    if requested_examples > examples:
        raise ValueError(
            f"requested {requested_examples} examples "
            f"({steps} steps x batch {batch_size}), but dataset only has {examples}"
        )

    load_started = time.time()
    first_shard = value_shards[0]
    first_payload = torch.load(
        dataset_dir / first_shard["file"],
        map_location="cpu",
        weights_only=True,
    )
    full_tensors: dict[str, torch.Tensor] = {}
    for key, value in first_payload.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"expected tensor payload for {key!r}, got {type(value)}")
        full_tensors[key] = torch.empty(
            (examples, *value.shape[1:]),
            dtype=value.dtype,
            device=device,
        )

    copied_bytes = _copy_payload_to_full_tensors(
        full_tensors,
        first_payload,
        start=int(first_shard["start"]),
        end=int(first_shard["end"]),
        device=device,
    )
    del first_payload

    for shard in value_shards[1:]:
        payload = torch.load(
            dataset_dir / shard["file"],
            map_location="cpu",
            weights_only=True,
        )
        copied_bytes += _copy_payload_to_full_tensors(
            full_tensors,
            payload,
            start=int(shard["start"]),
            end=int(shard["end"]),
            device=device,
        )
        del payload

    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(shuffle_seed))
        permutation = torch.randperm(examples, generator=generator).to(device=device)
        full_tensors = {
            key: value.index_select(0, permutation)
            for key, value in full_tensors.items()
        }
        del permutation
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    batch = rebel_batch_from_tensors(
        full_tensors,
        device=device,
        float_dtype=torch.float32,
        hand_dim=int(manifest.get("hands", 1326)),
    )
    return GpuValueEpoch(
        batch=batch,
        batch_size=int(batch_size),
        shuffle_seed=int(shuffle_seed),
        examples=examples,
        tensor_bytes=copied_bytes,
        load_time_s=time.time() - load_started,
    )


def _training_step_timing(step_times: list[float]) -> dict[str, float | int | None] | None:
    if not step_times:
        return None
    return {
        "count": len(step_times),
        "mean_s": statistics.fmean(step_times),
        "median_s": statistics.median(step_times),
        "mean_excluding_first_s": (
            statistics.fmean(step_times[1:]) if len(step_times) > 1 else None
        ),
        "min_s": min(step_times),
        "max_s": max(step_times),
    }


def _benchmark_no_grad_value_inference(
    *,
    trainer: RebelCFRTrainer,
    gpu_epoch: GpuValueEpoch,
    batch_size: int = 4096,
    warmup_batches: int = 3,
    timed_batches: int = 20,
    value_head: str = "post",
) -> dict[str, Any]:
    if timed_batches <= 0:
        return {
            "kind": "no_grad_value_forward",
            "skipped": True,
            "reason": "timed_batches <= 0",
            "batch_size": int(batch_size),
            "warmup_batches": int(warmup_batches),
            "timed_batches": int(timed_batches),
            "value_head": str(value_head),
            "count": 0,
        }
    required = batch_size * (warmup_batches + timed_batches)
    if required > gpu_epoch.examples:
        raise ValueError(
            f"inference benchmark needs {required} examples, "
            f"but GPU epoch only has {gpu_epoch.examples}"
        )

    model = trainer.model
    was_training = model.training
    model.eval()
    device = trainer.device

    timed_event_s: list[float] = []
    timed_wall_s: list[float] = []
    try:
        with torch.inference_mode(), trainer.model_autocast():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for batch_idx in range(warmup_batches):
                start = batch_idx * batch_size
                batch = gpu_epoch.slice_batch(start, batch_size)
                model(
                    batch.features,
                    include_policy=False,
                    apply_zero_sum=False,
                    value_head=value_head,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            for timed_idx in range(timed_batches):
                batch_idx = warmup_batches + timed_idx
                start = batch_idx * batch_size
                batch = gpu_epoch.slice_batch(start, batch_size)
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(device)
                    wall_started = time.perf_counter()
                    start_event.record()
                    model(
                        batch.features,
                        include_policy=False,
                        apply_zero_sum=False,
                        value_head=value_head,
                    )
                    end_event.record()
                    end_event.synchronize()
                    timed_wall_s.append(time.perf_counter() - wall_started)
                    timed_event_s.append(start_event.elapsed_time(end_event) / 1000.0)
                else:
                    wall_started = time.perf_counter()
                    model(
                        batch.features,
                        include_policy=False,
                        apply_zero_sum=False,
                        value_head=value_head,
                    )
                    timed_wall_s.append(time.perf_counter() - wall_started)
    finally:
        model.train(was_training)

    runtime_s = timed_event_s if timed_event_s else timed_wall_s
    return {
        "kind": "no_grad_value_forward",
        "batch_size": int(batch_size),
        "warmup_batches": int(warmup_batches),
        "timed_batches": int(timed_batches),
        "value_head": str(value_head),
        "count": len(runtime_s),
        "mean_s": statistics.fmean(runtime_s),
        "median_s": statistics.median(runtime_s),
        "min_s": min(runtime_s),
        "max_s": max(runtime_s),
        "runtime_s": runtime_s,
        "cuda_event_runtime_s": timed_event_s,
        "wall_runtime_s": timed_wall_s,
        "mean_excluding_first_s": statistics.fmean(runtime_s),
    }


def _apply_overnight_overrides(cfg) -> None:
    cfg.num_steps = 100
    cfg.use_wandb = False
    cfg.data.belief_mode = "mixed"
    cfg.data.belief_profile = "actions_12_end"
    cfg.train.replay_buffer_batches = 32
    cfg.trueskill.enabled = False

    cfg.search.depth = 5
    cfg.search.bet_bins_by_depth = [
        [0.25, 0.5, 0.75, 1.0, 1.5],
        [0.5, 1.0],
        [1.0],
        [1.0],
        [],
    ]
    cfg.search.allin_by_depth = [True, True, True, True, False]
    cfg.search.cfr_type = "sapcfr"
    cfg.search.iterations = 300
    cfg.search.iterations_final = 300
    cfg.search.sapcfr_alpha = 2.0
    cfg.search.predictive_cfr_delay = 40
    cfg.search.predictive_cfr_dcfr_hybrid = True
    cfg.search.dcfr_plus_delay = 80
    cfg.search.warm_start_iterations = 15
    cfg.search.warm_start_type = "model_br"
    cfg.search.warm_start_multiplier = 2
    cfg.search.sparse = True
    cfg.search.sparse_fused = True
    cfg.search.cfr_plus = False
    cfg.search.cfr_avg = False

    cfg.validation_set.enabled = True
    cfg.validation_set.dataset = DEFAULT_VALIDATION_DATASET
    cfg.validation_set.interval = 50
    cfg.validation_set.batch_size = 1024
    cfg.validation_set.max_examples = None


def _resolve_value_batch_size(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> int:
    if args.value_batch_size is not None:
        return int(args.value_batch_size)
    value_examples = int(manifest["value_examples"])
    if value_examples % int(args.steps) == 0:
        return value_examples // int(args.steps)
    cfg = load_rebel_config_file("conf/config_rebel_curriculum_river.yaml")
    _apply_overnight_overrides(cfg)
    return max(
        1,
        int(round(cfg.train.batch_size / cfg.train.value_reuse_goal)),
    )


def _build_config(
    args: argparse.Namespace,
    *,
    proposal: str,
    manifest: dict[str, Any],
    value_batch_size: int,
):
    cfg = load_rebel_config_file("conf/config_rebel_curriculum_river.yaml")
    _apply_overnight_overrides(cfg)
    cfg.num_steps = int(args.steps)
    cfg.checkpoint_interval = int(args.validation_interval)
    cfg.validation_set.interval = int(args.validation_interval)
    cfg.seed = int(args.seed)
    cfg.model.compile = str(getattr(args, "compile_mode", "reduce-overhead"))

    dataset_dir = _dataset_dir(args.dataset)
    cfg.data.mode = "pregenerated"
    cfg.data.live_root_source = "random_river"
    cfg.data.pregenerated.value_batch_size = value_batch_size
    cfg.data.pregenerated.policy_batch_size = 0
    cfg.data.pregenerated.shuffle = bool(args.shuffle)
    cfg.data.pregenerated.direct_sample = False
    cfg.data.pregenerated.datasets = [
        PregeneratedDatasetConfig(
            path=str(dataset_dir),
            value_weight=1.0,
            policy_weight=0.0,
        )
    ]
    cfg.curriculum.stages = []
    cfg.curriculum.substeps = {}

    for key, value in PROPOSALS[proposal].items():
        if key == "train":
            for train_key, train_value in value.items():
                setattr(cfg.train, train_key, train_value)
        else:
            setattr(cfg.model, key, value)

    run_dir = args.output_root / proposal
    cfg.checkpoint_dir = str(run_dir / "checkpoints")
    cfg.wandb_name = f"value_arch_{proposal}_{args.steps}step"
    return cfg, manifest, value_batch_size, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("proposals", nargs="+", choices=sorted(PROPOSALS))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--value-batch-size", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--timing-warmup-batches", type=int, default=3)
    parser.add_argument("--timing-batches", type=int, default=20)
    parser.add_argument("--timing-value-head", default="post")
    return parser.parse_args()


def _run_one_proposal(
    *,
    args: argparse.Namespace,
    proposal: str,
    manifest: dict[str, Any],
    value_batch_size: int,
    gpu_epoch: GpuValueEpoch,
) -> dict[str, Any]:
    cfg, manifest, value_batch_size, run_dir = _build_config(
        args,
        proposal=proposal,
        manifest=manifest,
        value_batch_size=value_batch_size,
    )
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "curriculum_substep": "river",
        "curriculum_kind": "train",
        "curriculum_net": "S_river",
        "value_arch_proposal": proposal,
        "value_arch_proposal_settings": PROPOSALS[proposal],
        "value_arch_proposal_queue": list(args.proposals),
        "effective_model_settings": {
            "hidden_dim": int(cfg.model.hidden_dim),
            "ffn_dim": int(cfg.model.ffn_dim),
            "range_hidden_dim": int(cfg.model.range_hidden_dim),
            "num_hidden_layers": int(cfg.model.num_hidden_layers),
            "num_value_layers": int(cfg.model.num_value_layers),
            "board_interaction_dim": int(cfg.model.board_interaction_dim),
            "board_interaction_skip_out": bool(cfg.model.board_interaction_skip_out),
            "board_interaction_gated": bool(cfg.model.board_interaction_gated),
            "value_head_rank": int(cfg.model.value_head_rank),
            "value_hand_basis_rank": int(cfg.model.value_hand_basis_rank),
            "belief_low_rank_dim": int(cfg.model.belief_low_rank_dim),
            "belief_low_rank_board_conditioned": bool(
                cfg.model.belief_low_rank_board_conditioned
            ),
            "belief_second_moment": bool(cfg.model.belief_second_moment),
            "belief_skip_matching_encoder": bool(
                cfg.model.belief_skip_matching_encoder
            ),
            "belief_linear_encoder": bool(cfg.model.belief_linear_encoder),
            "belief_board_film": bool(cfg.model.belief_board_film),
            "belief_board_bilinear_rank": int(cfg.model.belief_board_bilinear_rank),
            "belief_board_mass_features": bool(cfg.model.belief_board_mass_features),
            "value_strength_bucket_count": int(cfg.model.value_strength_bucket_count),
            "value_strength_bucket_relative": bool(
                cfg.model.value_strength_bucket_relative
            ),
            "value_strength_bucket_board_only": bool(
                cfg.model.value_strength_bucket_board_only
            ),
            "value_strength_bucket_blockers": bool(
                cfg.model.value_strength_bucket_blockers
            ),
            "value_strength_bucket_coarse_residual": bool(
                cfg.model.value_strength_bucket_coarse_residual
            ),
            "value_bucket_coarse_dim": int(cfg.model.value_bucket_coarse_dim),
            "value_latent_bucket_count": int(cfg.model.value_latent_bucket_count),
            "value_latent_bucket_dim": int(cfg.model.value_latent_bucket_dim),
            "value_exact_river_features": bool(cfg.model.value_exact_river_features),
            "value_showdown_baseline": bool(cfg.model.value_showdown_baseline),
            "value_river_range_equity_baseline": bool(
                cfg.model.value_river_range_equity_baseline
            ),
            "value_river_range_equity_baseline_scale": float(
                cfg.model.value_river_range_equity_baseline_scale
            ),
            "value_river_range_equity_pot_power": float(
                cfg.model.value_river_range_equity_pot_power
            ),
            "value_river_range_equity_pos_scale": float(
                cfg.model.value_river_range_equity_pos_scale
            ),
            "value_river_range_equity_neg_scale": float(
                cfg.model.value_river_range_equity_neg_scale
            ),
            "value_river_range_equity_intercept": float(
                cfg.model.value_river_range_equity_intercept
            ),
            "value_river_range_equity_blockers": bool(
                cfg.model.value_river_range_equity_blockers
            ),
            "value_river_range_equity_rank_bins": int(
                cfg.model.value_river_range_equity_rank_bins
            ),
            "value_river_range_equity_feature_head": bool(
                cfg.model.value_river_range_equity_feature_head
            ),
            "value_river_range_equity_trunk_context": bool(
                cfg.model.value_river_range_equity_trunk_context
            ),
            "value_river_range_equity_film_rank": int(
                cfg.model.value_river_range_equity_film_rank
            ),
            "value_river_range_equity_film_hidden_dim": int(
                cfg.model.value_river_range_equity_film_hidden_dim
            ),
            "value_output_init_scale": float(cfg.model.value_output_init_scale),
            "value_action_summary_head": bool(cfg.model.value_action_summary_head),
            "value_multiresolution_coef": float(
                cfg.train.value_multiresolution_coef
            ),
            "value_bucket_aux_coef": float(cfg.train.value_bucket_aux_coef),
            "value_ordering_coef": float(cfg.train.value_ordering_coef),
            "value_bucket_usage_coef": float(cfg.train.value_bucket_usage_coef),
            "value_bucket_entropy_coef": float(cfg.train.value_bucket_entropy_coef),
            "value_action_summary_coef": float(cfg.train.value_action_summary_coef),
            "compile": str(cfg.model.compile),
        },
        "pregenerated_dataset": str(_dataset_dir(args.dataset)),
        "pregenerated_manifest": manifest,
        "value_batch_size": value_batch_size,
        "seed": int(args.seed),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    trainer = RebelCFRTrainer(cfg, torch.device(cfg.device))
    metadata["gpu_value_epoch"] = {
        "enabled": True,
        "examples": gpu_epoch.examples,
        "batch_size": gpu_epoch.batch_size,
        "shuffle": bool(args.shuffle),
        "shuffle_seed": gpu_epoch.shuffle_seed,
        "tensor_bytes": gpu_epoch.tensor_bytes,
        "tensor_gib": gpu_epoch.tensor_bytes / (1024**3),
        "load_time_s": gpu_epoch.load_time_s,
        "replay_buffer_write": False,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")
    step_times: list[float] = []

    def step_body(step: int) -> dict[str, Any]:
        step_started = time.time()
        value_batch = gpu_epoch.step_batch(step)
        metrics = trainer.train_value_batch(value_batch, step)
        step_time_s = time.time() - step_started
        metrics["step_time_s"] = step_time_s
        step_times.append(step_time_s)
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")
        return metrics

    started = time.time()
    last_step = run_training_loop(
        trainer,
        cfg,
        run=None,
        start_step=0,
        stop_step=cfg.num_steps,
        stage_tag=f"value_arch_{proposal}",
        step_body=step_body,
        checkpoint_metadata=metadata,
        value_only=True,
        print_preflop_analyzer=False,
        log_interval=int(args.log_interval),
    )
    validation_evaluator = RebelValueValidationSetEvaluator(
        trainer=trainer,
        cfg=cfg,
        dataset_path=cfg.validation_set.dataset,
        batch_size=cfg.validation_set.batch_size,
        max_examples=cfg.validation_set.max_examples,
    )
    final_validation = validation_evaluator.evaluate()
    (run_dir / "final_validation.json").write_text(
        json.dumps(final_validation, indent=2, sort_keys=True) + "\n"
    )
    training_step_timing = _training_step_timing(step_times)
    inference_timing = _benchmark_no_grad_value_inference(
        trainer=trainer,
        gpu_epoch=gpu_epoch,
        batch_size=4096,
        warmup_batches=int(getattr(args, "timing_warmup_batches", 3)),
        timed_batches=int(getattr(args, "timing_batches", 20)),
        value_head=str(getattr(args, "timing_value_head", "post")),
    )
    summary = {
        "proposal": proposal,
        "last_step": last_step,
        "elapsed_s": time.time() - started,
        "step_timing": inference_timing,
        "training_step_timing": training_step_timing,
        "metrics_path": str(metrics_path),
        "checkpoint_dir": cfg.checkpoint_dir,
        "value_batch_size": value_batch_size,
        "final_validation": final_validation,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    dataset_dir = _dataset_dir(args.dataset)
    manifest = _load_manifest(dataset_dir)
    value_batch_size = _resolve_value_batch_size(args, manifest)
    base_cfg = load_rebel_config_file("conf/config_rebel_curriculum_river.yaml")
    _apply_overnight_overrides(base_cfg)
    base_cfg.seed = int(args.seed)
    base_cfg.model.compile = str(getattr(args, "compile_mode", "reduce-overhead"))
    gpu_epoch = _load_gpu_value_epoch(
        dataset_dir=dataset_dir,
        manifest=manifest,
        device=torch.device(base_cfg.device),
        batch_size=value_batch_size,
        steps=int(args.steps),
        shuffle_seed=int(args.seed),
        shuffle=bool(args.shuffle),
    )

    summaries = []
    for proposal in args.proposals:
        summaries.append(
            _run_one_proposal(
                args=args,
                proposal=proposal,
                manifest=manifest,
                value_batch_size=value_batch_size,
                gpu_epoch=gpu_epoch,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(summaries) > 1:
        queue_summary = {
            "proposals": list(args.proposals),
            "dataset": str(dataset_dir),
            "steps": int(args.steps),
            "value_batch_size": int(value_batch_size),
            "compile_mode": str(getattr(args, "compile_mode", "reduce-overhead")),
            "timing_warmup_batches": int(
                getattr(args, "timing_warmup_batches", 3)
            ),
            "timing_batches": int(getattr(args, "timing_batches", 20)),
            "timing_value_head": str(getattr(args, "timing_value_head", "post")),
            "gpu_value_epoch": {
                "examples": gpu_epoch.examples,
                "batch_size": gpu_epoch.batch_size,
                "shuffle": bool(args.shuffle),
                "shuffle_seed": gpu_epoch.shuffle_seed,
                "tensor_bytes": gpu_epoch.tensor_bytes,
                "tensor_gib": gpu_epoch.tensor_bytes / (1024**3),
                "load_time_s": gpu_epoch.load_time_s,
            },
            "summaries": summaries,
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "queue_summary.json").write_text(
            json.dumps(queue_summary, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

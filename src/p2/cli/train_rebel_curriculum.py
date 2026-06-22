#!/usr/bin/env python3
"""Hydra entry point for staged postflop ReBeL curriculum training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from p2.config.rebel_load import load_rebel_config
from p2.stages.curriculum import train_rebel_curriculum


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="config_rebel_cfr"
)
def main(dict_config: DictConfig) -> None:
    config = load_rebel_config(dict_config)
    train_rebel_curriculum(config)


if __name__ == "__main__":
    main()

"""Feed-forward poker models inspired by ReBeL."""

from p2.models.mlp.better_ffn import (
    BetterFFN,
    BetterPolicyFFN,
    BetterSplitFFN,
    BetterStreetValueFFN,
)
from p2.models.mlp.rebel_ffn import RebelFFN

__all__ = [
    "BetterFFN",
    "BetterPolicyFFN",
    "BetterSplitFFN",
    "BetterStreetValueFFN",
    "RebelFFN",
]

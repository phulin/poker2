from __future__ import annotations

from .compare_multiway_showdown_tiers import (
    TierResult,
    tier1_hero_removal,
    tier1_hero_removal_by_hand,
    tier2_first_order_opp_collision,
    tier2_first_order_opp_collision_by_hand,
    tier3_second_order_opp_collision,
    tier3_second_order_opp_collision_by_hand,
    tier4_third_degree_card_collision,
    tier4_third_degree_card_collision_by_hand,
)

__all__ = [
    "TierResult",
    "tier1_hero_removal",
    "tier1_hero_removal_by_hand",
    "tier2_first_order_opp_collision",
    "tier2_first_order_opp_collision_by_hand",
    "tier3_second_order_opp_collision",
    "tier3_second_order_opp_collision_by_hand",
    "tier4_third_degree_card_collision",
    "tier4_third_degree_card_collision_by_hand",
]

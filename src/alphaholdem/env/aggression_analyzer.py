"""
Aggression analyzer for computing bet amounts by hand equity groups.

This analyzer processes batches to compute average bet amounts across the 1326 hands,
grouped into 5 equity-based buckets.
"""

from functools import lru_cache

import torch

from alphaholdem.env.card_utils import hand_combos_tensor
from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS
from alphaholdem.rl.rebel_replay import RebelBatch


# Rank mapping: A=12, K=11, Q=10, J=9, T=8, 9=7, 8=6, 7=5, 6=4, 5=3, 4=2, 3=1, 2=0
RANK_TO_IDX = {
    "A": 12,
    "K": 11,
    "Q": 10,
    "J": 9,
    "T": 8,
    "9": 7,
    "8": 6,
    "7": 5,
    "6": 4,
    "5": 3,
    "4": 2,
    "3": 1,
    "2": 0,
}
IDX_TO_RANK = {v: k for k, v in RANK_TO_IDX.items()}


def parse_hand_name(hand_name: str) -> tuple[int, int]:
    """Parse a poker hand name to get card indices.

    Args:
        hand_name: Like 'AA', 'AKs', 'KQo'
                   - Pairs: 'AA' means pair of aces (any suits)
                   - Suited: 'AKs' means AK suited
                   - Offsuit: 'AKo' means AK offsuit

    Returns:
        Tuple of (card1, card2) card indices
    """
    if len(hand_name) == 2 and hand_name[0] == hand_name[1]:
        # Pair - use first two suits (0, 1)
        rank = RANK_TO_IDX[hand_name[0]]
        return rank, rank + 13
    elif len(hand_name) >= 3:
        rank1 = RANK_TO_IDX[hand_name[0]]
        rank2 = RANK_TO_IDX[hand_name[1]]
        is_suited = hand_name[2] == "s"

        # Suited hands use same suit, offsuit use different suits
        if is_suited:
            return rank1, rank2
        else:
            return rank1, rank2 + 13
    else:
        raise ValueError(f"Invalid hand name: {hand_name}")


NUM_GROUPS = 5
HAND_EQUITY_ORDERING = (
    "AA,KK,QQ,JJ,AKs,AQs,TT,AKo,AJs,KQs,99,ATs,AQo,KJs,88,QJs,KTs,AJo,A9s,QTs,"
    "77,KQo,JTs,A8s,K9s,ATo,A7s,A5s,66,KJo,A4s,Q9s,T9s,J9s,A6s,QJo,55,A3s,KTo,"
    "K8s,A2s,K7s,T8s,98s,QTo,Q8s,87s,44,A9o,JTo,J8s,76s,K6s,97s,K5s,K4s,T7s,"
    "Q7s,33,A8o,K9o,J7s,86s,65s,K3s,K2s,Q9o,Q6s,J9o,T9o,54s,22,Q5s,T8o,96s,75s,"
    "64s,A7o,Q4s,J8o,T7o,98o,97o,K8o,K7o,Q8o,Q3s,J6s,J5s,J4s,T6o,T6s,86o,85o,"
    "85s,76o,75o,74s,63s,53s,A6o,A5o,A4o,K6o,Q7o,Q2s,J7o,J6o,T5o,T5s,T4o,T3o,"
    "T2o,96o,95o,95s,94o,93o,92o,87o,84o,83o,82o,74o,73o,72o,65o,64o,63o,62o,"
    "53o,52o,42o,A3o,K5o,K4o,Q6o,Q5o,Q4o,Q3o,Q2o,J5o,J4o,J3o,J3s,J2o,T4s,T3s,"
    "84s,54o,43o,43s,K3o,K2o,J2s,T2s,93s,92s,82s,73s,62s,52s,42s,32s,A2o,94s,"
    "83s,72s,32o"
).split(",")


@lru_cache(maxsize=1)
def build_hand_to_group_mapping() -> torch.Tensor:
    """Build a [1326] tensor mapping each combo index to its equity group (0-4).

    Returns:
        Tensor of shape [1326] with group assignments
    """
    combos = hand_combos_tensor()

    # Build a reverse lookup: hand name -> list of combo indices (pre-compute once)
    hand_name_to_combos: dict[str, list[int]] = {}
    for idx in range(NUM_HANDS):
        c1, c2 = combos[idx]
        r1, r2 = c1 % 13, c2 % 13
        is_suited = c1 // 13 == c2 // 13

        # Create a canonical hand name from this combo
        if r1 == r2:
            hand_name = (
                f"{IDX_TO_RANK[int(r1)]}{IDX_TO_RANK[int(r1)]}"  # Pair like "AA"
            )
        else:
            rank_char_1 = IDX_TO_RANK[int(max(r1, r2))]
            rank_char_2 = IDX_TO_RANK[int(min(r1, r2))]
            suffix = "s" if is_suited else "o"
            hand_name = f"{rank_char_1}{rank_char_2}{suffix}"  # Like "AKs" or "AKo"

        if hand_name not in hand_name_to_combos:
            hand_name_to_combos[hand_name] = []
        hand_name_to_combos[hand_name].append(idx)

    # All hand combos, ranked descending by hand strength
    combos_ranked_list = [
        hand_name_to_combos[hand_name] for hand_name in HAND_EQUITY_ORDERING
    ]
    combos_ranked_tensor = torch.tensor(sum(combos_ranked_list, []), dtype=torch.long)

    return torch.chunk(combos_ranked_tensor, NUM_GROUPS)


class AggressionAnalyzer:
    """Singleton class for analyzing aggression metrics by hand equity groups."""

    _instance = None
    _group_mapping = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if AggressionAnalyzer._group_mapping is None:
            AggressionAnalyzer._group_mapping = build_hand_to_group_mapping()

    @classmethod
    def instance(cls) -> "AggressionAnalyzer":
        if cls._instance is None:
            cls._instance = AggressionAnalyzer()
        return cls._instance

    def analyze_batch(self, batch: RebelBatch) -> dict[str, torch.Tensor]:
        """Analyze a batch and return average bet amounts by hand equity group.

        Args:
            batch: A RebelBatch with statistics including 'bet_amounts'
                   The batch should have policy_targets [N, NUM_HANDS, num_actions]

        Returns:
            Dictionary with keys:
              - 'group_avg_bets': [5] tensor of average bet amounts per group
              - 'group_counts': [5] tensor of number of hand-state pairs per group
              - 'group_states': [5] tensor of number of unique states per group
        """
        if "bet_amounts" not in batch.statistics:
            return {}

        bet_amounts = batch.statistics["bet_amounts"]  # [N, num_bins]

        # Get the group chunks (tuple of 5 tensors)
        chunk_tuples = AggressionAnalyzer._group_mapping

        # Build a mapping from combo_idx to group_idx
        group_mapping = torch.zeros(
            NUM_HANDS, dtype=torch.long, device=bet_amounts.device
        )
        for group_idx, chunk in enumerate(chunk_tuples):
            chunk = chunk.to(bet_amounts.device)
            group_mapping[chunk] = group_idx

        # Get policy targets if available - shape [N, NUM_HANDS, num_actions]
        if batch.policy_targets is not None:
            policy_targets = batch.policy_targets  # [N, NUM_HANDS, num_actions]

            # For each state and each hand, compute the expected bet amount
            # by averaging over actions weighted by policy
            num_states, num_hands, num_actions = policy_targets.shape

            # Compute expected bet amount per hand using the policy
            # policy_targets: [N, NUM_HANDS, num_actions]
            # bet_amounts: [N, num_actions] (assuming num_actions == num_bins)
            # We want: for each state n and hand h, sum over a: policy[n,h,a] * bet_amounts[n,a]
            bet_expanded = bet_amounts.unsqueeze(1).expand(
                num_states, num_hands, -1
            )  # [N, NUM_HANDS, num_actions]
            expected_bet_per_hand = (policy_targets * bet_expanded).sum(
                dim=-1
            )  # [N, NUM_HANDS]

            # Group by hand equity groups
            # Flatten to [N * NUM_HANDS]
            total_bet_flat = expected_bet_per_hand.flatten()

            # Get group for each hand: [NUM_HANDS]
            hand_groups = group_mapping

            # Expand to match the flattened batch
            batch_size = num_states
            group_indices = (
                hand_groups.unsqueeze(0).expand(batch_size, -1).flatten()
            )  # [N * 1326]

            # Compute statistics per group
            num_groups = 5
            group_bet_sums = torch.zeros(num_groups, device=bet_amounts.device)
            group_counts = torch.zeros(
                num_groups, device=bet_amounts.device, dtype=torch.long
            )

            for group_idx in range(num_groups):
                mask = group_indices == group_idx
                group_bet_sums[group_idx] = torch.where(mask, total_bet_flat, 0).sum()
                group_counts[group_idx] = mask.sum()

            # Average bet amount per group
            group_avg_bets = torch.where(
                group_counts > 0,
                group_bet_sums / group_counts.float(),
                torch.zeros(num_groups, device=bet_amounts.device),
            )

            return {
                "group_avg_bets": group_avg_bets,
                "group_counts": group_counts,
                "group_states": torch.tensor(
                    [len(chunk) for chunk in chunk_tuples], device=bet_amounts.device
                ),
                "overall_avg": total_bet_flat.mean().item(),
                "overall_std": total_bet_flat.std().item(),
            }

        # Fallback: if no policy targets, just return overall statistics
        total_bet_amount = bet_amounts.sum(dim=1)
        return {
            "group_avg_bets": torch.zeros(5, device=bet_amounts.device),
            "group_counts": torch.zeros(5, device=bet_amounts.device, dtype=torch.long),
            "overall_avg": total_bet_amount.mean().item(),
            "overall_std": total_bet_amount.std().item(),
        }


aggression_analyzer = AggressionAnalyzer()

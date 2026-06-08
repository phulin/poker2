"""
Aggression analyzer for computing bet amounts by hand equity groups.

This analyzer processes batches to compute average bet amounts across the 1326 hands,
grouped into 5 equity-based buckets.
"""

from functools import lru_cache

import torch

from p2.env.card_utils import (
    HAND_EQUITY_ORDERING,
    IDX_TO_RANK,
    NUM_HANDS,
    PREFLOP_HANDS,
    RANK_TO_IDX,
    hand_combos_tensor,
)
from p2.rl.rebel_batch import RebelBatch

NUM_GROUPS = 5


@lru_cache(maxsize=1)
def build_hand_to_group_mapping(device: torch.device | None = None) -> torch.Tensor:
    """Build a [1326] tensor mapping each combo index to its equity group (0-4).

    Returns:
        Tensor of shape [1326] with group assignments
    """
    combos = hand_combos_tensor(device=device)

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
    combos_ranked_tensor = torch.tensor(
        sum(combos_ranked_list, []), dtype=torch.long, device=device
    )

    return torch.chunk(combos_ranked_tensor, NUM_GROUPS)


@lru_cache(maxsize=1)
def build_preflop_class_to_group_mapping(
    device: torch.device | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build 169-class hand-strength groups using the same ordering."""

    class_ids: list[int] = []
    for hand_name in HAND_EQUITY_ORDERING:
        hi = RANK_TO_IDX[hand_name[0]]
        lo = RANK_TO_IDX[hand_name[1]]
        if len(hand_name) == 2:
            class_id = hi * 13 + hi
        elif hand_name[2] == "s":
            class_id = hi * 13 + lo
        elif hand_name[2] == "o":
            class_id = lo * 13 + hi
        else:
            raise ValueError(f"Unsupported preflop hand name {hand_name!r}")
        class_ids.append(class_id)

    ranked = torch.tensor(class_ids, dtype=torch.long, device=device)
    return torch.chunk(ranked, NUM_GROUPS)


class AggressionAnalyzer:
    """Singleton class for analyzing aggression metrics by hand equity groups."""

    def __init__(self, device: torch.device | None = None):
        self._group_mapping = build_hand_to_group_mapping(device=device)
        self._group_states = torch.tensor(
            [len(chunk) for chunk in self._group_mapping],
            dtype=torch.long,
            device=device,
        )
        self._preflop_group_mapping = build_preflop_class_to_group_mapping(
            device=device
        )
        self._preflop_group_states = torch.tensor(
            [len(chunk) for chunk in self._preflop_group_mapping],
            dtype=torch.long,
            device=device,
        )

    def analyze_batch(
        self, batch: RebelBatch, max_batch_size: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Analyze a batch and return average bet amounts by hand equity group.

        Args:
            batch: A RebelBatch with statistics including 'bet_amounts'
                   The batch should have policy_targets [N, NUM_HANDS, num_actions]
            max_batch_size: Maximum batch size to process at once. If None, processes
                           the entire batch without chunking.

        Returns:
            Dictionary with keys:
              - 'group_avg_bets': [5] tensor of average bet amounts per group
              - 'group_counts': [5] tensor of number of hand-state pairs per group
              - 'group_states': [5] tensor of number of unique states per group
        """
        if max_batch_size is None or len(batch) <= max_batch_size:
            result = self._analyze_single_batch(batch)
            if not result:
                return result
            # Convert internal format to public format
            if "overall_count" in result and result["overall_count"] > 0:
                overall_avg = result["overall_sum"] / result["overall_count"]
                overall_var = (result["overall_sum_sq"] / result["overall_count"]) - (
                    overall_avg**2
                )
                overall_std = overall_var.clamp_min(0.0).sqrt()
            else:
                overall_avg = 0.0
                overall_std = 0.0
            return {
                "group_avg_bets": result["group_avg_bets"],
                "group_counts": result["group_counts"],
                "group_states": result["group_states"],
                "overall_avg": overall_avg,
                "overall_std": overall_std,
            }

        # Chunk the batch and process each chunk
        results = []
        for i in range(0, len(batch), max_batch_size):
            chunk = batch[i : i + max_batch_size]
            chunk_result = self._analyze_single_batch(chunk)
            if chunk_result:  # Only add non-empty results
                results.append(chunk_result)

        if not results:
            return {}

        # Aggregate results across chunks
        # For group_avg_bets, we need to compute weighted average
        total_group_counts = torch.zeros_like(results[0]["group_counts"])
        total_group_bet_sums = torch.zeros_like(results[0]["group_avg_bets"])

        # Track overall statistics: sum, sum of squares, and count
        overall_sum = torch.zeros((), device=total_group_bet_sums.device)
        overall_sum_sq = torch.zeros((), device=total_group_bet_sums.device)
        overall_count = 0

        for result in results:
            group_counts = result["group_counts"]
            group_avg_bets = result["group_avg_bets"]
            total_group_counts += group_counts

            # Reconstruct bet sums from averages and counts
            group_bet_sums = group_avg_bets * group_counts.float()
            total_group_bet_sums += group_bet_sums

            # Aggregate overall statistics
            overall_sum += result["overall_sum"]
            overall_sum_sq += result["overall_sum_sq"]
            overall_count += result["overall_count"]

        # Compute weighted average bet amounts per group
        group_avg_bets = torch.where(
            total_group_counts > 0,
            total_group_bet_sums / total_group_counts.float(),
            0.0,
        )

        # Compute overall mean and std from aggregated statistics
        if overall_count > 0:
            overall_avg = overall_sum / overall_count
            # Variance = E[X^2] - (E[X])^2
            overall_var = (overall_sum_sq / overall_count) - (overall_avg**2)
            overall_std = overall_var.clamp_min(0.0).sqrt()
        else:
            overall_avg = 0.0
            overall_std = 0.0

        return {
            "group_avg_bets": group_avg_bets,
            "group_counts": total_group_counts,
            "group_states": results[0]["group_states"],  # Same for all chunks
            "overall_avg": overall_avg,
            "overall_std": overall_std,
        }

    def _analyze_single_batch(self, batch: RebelBatch) -> dict[str, torch.Tensor]:
        """Analyze a single batch chunk and return average bet amounts by hand equity group.

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

        # Get policy targets if available - shape [N, NUM_HANDS, num_actions]
        if batch.policy_targets is not None:
            policy_targets = batch.policy_targets  # [N, NUM_HANDS, num_actions]

            # For each state and each hand, compute the expected bet amount
            # by averaging over actions weighted by policy
            num_states, num_hands, num_actions = policy_targets.shape
            del num_actions
            if num_hands == NUM_HANDS:
                chunk_tuples = tuple(
                    chunk.to(bet_amounts.device) for chunk in self._group_mapping
                )
                group_states = self._group_states.to(bet_amounts.device)
            elif num_hands == PREFLOP_HANDS:
                chunk_tuples = tuple(
                    chunk.to(bet_amounts.device)
                    for chunk in self._preflop_group_mapping
                )
                group_states = self._preflop_group_states.to(bet_amounts.device)
            else:
                raise ValueError(f"Unsupported policy target hand dim {num_hands}")

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

            group_bet_sums = torch.stack(
                [
                    expected_bet_per_hand.index_select(1, chunk).sum()
                    for chunk in chunk_tuples
                ]
            )
            group_counts = group_states * num_states

            # Average bet amount per group
            group_avg_bets = torch.where(
                group_counts > 0, group_bet_sums / group_counts.float(), 0.0
            )

            # Compute overall statistics for aggregation
            overall_sum = expected_bet_per_hand.sum()
            overall_sum_sq = (expected_bet_per_hand**2).sum()
            overall_count = expected_bet_per_hand.numel()

            return {
                "group_avg_bets": group_avg_bets,
                "group_counts": group_counts,
                "group_states": group_states,
                "overall_sum": overall_sum,
                "overall_sum_sq": overall_sum_sq,
                "overall_count": overall_count,
            }

        # Fallback: if no policy targets, just return overall statistics
        group_states = self._group_states.to(bet_amounts.device)
        total_bet_amount = bet_amounts.sum(dim=1)
        overall_sum = total_bet_amount.sum()
        overall_sum_sq = (total_bet_amount**2).sum()
        overall_count = len(total_bet_amount)

        return {
            "group_avg_bets": torch.zeros(5, device=bet_amounts.device),
            "group_counts": torch.zeros(5, device=bet_amounts.device, dtype=torch.long),
            "group_states": group_states,
            "overall_sum": overall_sum,
            "overall_sum_sq": overall_sum_sq,
            "overall_count": overall_count,
        }

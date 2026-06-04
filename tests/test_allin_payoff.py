from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_write_allin_table_values_triton_matches_eager_for_both_players() -> None:
    pytest.importorskip("triton")

    from p2.env.card_utils import (
        NUM_HANDS,
        combo_compatible_tensor,
        combo_suit_permutation_tensor,
    )
    from p2.search.allin_payoff import (
        FLOP_I8_SCALE,
        I16_SCALE,
        allin_values_from_payoff_batch,
        write_allin_belief_card_stats_split_triton_,
        write_allin_grouped_table_values_card_denom_dot_values_triton_,
        write_allin_table_values_card_denom_dot_triton_,
        write_allin_table_values_card_denom_dot_values_triton_,
        write_allin_table_values_card_denom_triton_,
        write_allin_table_values_triton_,
    )

    device = torch.device("cuda")
    torch.manual_seed(0xA11)
    num_nodes = 7
    num_allin = 4
    node_indices = torch.tensor([0, 2, 4, 6], device=device)
    beliefs = torch.rand(num_nodes, 2, NUM_HANDS, device=device)
    beliefs /= beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    stacks = torch.rand(num_nodes, 2, device=device) * 1000.0 + 1.0
    pot = torch.rand(num_nodes, device=device) * 120.0
    starting_stacks = torch.full((num_nodes, 2), 1000.0, device=device)
    env_scale = torch.full((num_nodes,), 1000.0, device=device)
    compatible = combo_compatible_tensor(device=device)

    def masked_random_tables(
        shape: tuple[int, int, int],
        *,
        dtype: torch.dtype = torch.int8,
    ) -> torch.Tensor:
        if dtype == torch.int8:
            table = torch.randint(-127, 128, shape, device=device, dtype=dtype)
        else:
            table = torch.randint(-32768, 32768, shape, device=device, dtype=dtype)
        return torch.where(
            compatible,
            table,
            torch.zeros((), device=device, dtype=dtype),
        )

    def expected_values(
        tables: torch.Tensor,
        *,
        scale: float = FLOP_I8_SCALE,
    ) -> torch.Tensor:
        values = allin_values_from_payoff_batch(
            tables,
            beliefs[node_indices],
            scale=scale,
        )
        potential = (
            stacks[node_indices]
            + pot[node_indices, None]
            - starting_stacks[node_indices]
        )
        values *= potential[:, :, None] / env_scale[node_indices, None, None]
        return values

    tables = masked_random_tables((num_allin, NUM_HANDS, NUM_HANDS))
    latest_values = torch.empty(num_nodes, 2, NUM_HANDS, device=device)
    write_allin_table_values_triton_(
        table=tables,
        beliefs=beliefs,
        node_indices=node_indices,
        latest_values=latest_values,
        stacks=stacks,
        pot=pot,
        starting_stacks=starting_stacks,
        env_scale=env_scale,
        table_scale=FLOP_I8_SCALE,
        batch_table=True,
    )
    torch.testing.assert_close(
        latest_values[node_indices],
        expected_values(tables),
        rtol=3e-6,
        atol=3e-6,
    )

    canon_tables = masked_random_tables((3, NUM_HANDS, NUM_HANDS))
    canon_ids = torch.tensor([0, 1, 2, 0], device=device)
    perm_ids = torch.tensor([0, 5, 12, 23], device=device)
    combo_perms = combo_suit_permutation_tensor(device=device)
    perms = combo_perms.index_select(0, perm_ids)
    permuted_tables = canon_tables.index_select(0, canon_ids)
    permuted_tables = permuted_tables.gather(
        1,
        perms[:, :, None].expand(-1, -1, NUM_HANDS),
    ).gather(
        2,
        perms[:, None, :].expand(-1, NUM_HANDS, -1),
    )

    latest_values_perm = torch.empty(num_nodes, 2, NUM_HANDS, device=device)
    write_allin_table_values_triton_(
        table=canon_tables,
        beliefs=beliefs,
        node_indices=node_indices,
        latest_values=latest_values_perm,
        stacks=stacks,
        pot=pot,
        starting_stacks=starting_stacks,
        env_scale=env_scale,
        table_scale=FLOP_I8_SCALE,
        canon_ids=canon_ids,
        perm_ids=perm_ids,
        combo_perms=combo_perms,
    )
    torch.testing.assert_close(
        latest_values_perm[node_indices],
        expected_values(permuted_tables),
        rtol=3e-6,
        atol=3e-6,
    )

    latest_values_card = torch.empty(num_nodes, 2, NUM_HANDS, device=device)
    write_allin_table_values_card_denom_triton_(
        table=canon_tables,
        beliefs=beliefs,
        node_indices=node_indices,
        latest_values=latest_values_card,
        stacks=stacks,
        pot=pot,
        starting_stacks=starting_stacks,
        env_scale=env_scale,
        table_scale=FLOP_I8_SCALE,
        canon_ids=canon_ids,
    )
    torch.testing.assert_close(
        latest_values_card[node_indices],
        expected_values(canon_tables.index_select(0, canon_ids)),
        rtol=3e-6,
        atol=3e-6,
    )

    latest_values_dot = torch.empty(num_nodes, 2, NUM_HANDS, device=device)
    write_allin_table_values_card_denom_dot_triton_(
        table=canon_tables,
        beliefs=beliefs,
        node_indices=node_indices,
        latest_values=latest_values_dot,
        stacks=stacks,
        pot=pot,
        starting_stacks=starting_stacks,
        env_scale=env_scale,
        table_scale=FLOP_I8_SCALE,
        canon_ids=canon_ids,
    )
    torch.testing.assert_close(
        latest_values_dot[node_indices],
        expected_values(canon_tables.index_select(0, canon_ids)),
        rtol=1e-3,
        atol=5e-4,
    )

    turn_tables = masked_random_tables(
        (3, NUM_HANDS, NUM_HANDS),
        dtype=torch.int16,
    )
    latest_values_turn_dot = torch.empty(num_nodes, 2, NUM_HANDS, device=device)
    write_allin_table_values_card_denom_dot_triton_(
        table=turn_tables,
        beliefs=beliefs,
        node_indices=node_indices,
        latest_values=latest_values_turn_dot,
        stacks=stacks,
        pot=pot,
        starting_stacks=starting_stacks,
        env_scale=env_scale,
        table_scale=I16_SCALE,
        canon_ids=canon_ids,
        block_h=64,
        block_k=128,
        block_p=8,
    )
    torch.testing.assert_close(
        latest_values_turn_dot[node_indices],
        expected_values(turn_tables.index_select(0, canon_ids), scale=I16_SCALE),
        rtol=1e-3,
        atol=1e-3,
    )

    stats_buffer = torch.empty(num_allin, 2, 53, dtype=torch.float32, device=device)
    empty_nodes = torch.empty(0, dtype=torch.long, device=device)
    empty_stats = torch.empty(0, 2, 53, dtype=torch.float32, device=device)
    write_allin_belief_card_stats_split_triton_(
        beliefs=beliefs,
        node_indices0=empty_nodes,
        stats_buffer0=empty_stats,
        node_indices1=node_indices,
        stats_buffer1=stats_buffer,
    )
    latest_values_turn_values = torch.empty(num_nodes, 2, NUM_HANDS, device=device)
    write_allin_table_values_card_denom_dot_values_triton_(
        table=turn_tables,
        beliefs=beliefs,
        node_indices=node_indices,
        latest_values=latest_values_turn_values,
        stacks=stacks,
        pot=pot,
        starting_stacks=starting_stacks,
        env_scale=env_scale,
        table_scale=I16_SCALE,
        canon_ids=canon_ids,
        stats_buffer=stats_buffer,
        block_h=64,
        block_k=64,
        block_p=8,
    )

    group_positions = torch.tensor(
        [[0, 3], [1, -1], [2, -1]],
        dtype=torch.long,
        device=device,
    )
    latest_values_turn_grouped = torch.empty(num_nodes, 2, NUM_HANDS, device=device)
    write_allin_grouped_table_values_card_denom_dot_values_triton_(
        table=turn_tables,
        beliefs=beliefs,
        node_indices=node_indices,
        group_positions=group_positions,
        latest_values=latest_values_turn_grouped,
        stacks=stacks,
        pot=pot,
        starting_stacks=starting_stacks,
        env_scale=env_scale,
        table_scale=I16_SCALE,
        stats_buffer=stats_buffer,
    )
    torch.testing.assert_close(
        latest_values_turn_grouped[node_indices],
        latest_values_turn_values[node_indices],
        rtol=1e-3,
        atol=1e-3,
    )

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.card_utils import (
    combo_lookup_tensor,
    hand_combos_tensor,
    mask_conflicting_combos,
)
from alphaholdem.env.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.cnn.cnn_embedding_data import CNNEmbeddingData
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.token_sequence_builder import TokenSequenceBuilder
from alphaholdem.core.structured_config import SearchConfig
import torch.nn.functional as F
from alphaholdem.search.dcfr import DCFRResult, run_dcfr
from alphaholdem.rl.popart_normalizer import PopArtNormalizer


class CFRManager:
    def __init__(
        self,
        batch_size: int,
        env_proto: HUNLTensorEnv,
        bet_bins: list[float],
        sequence_length: int,
        device: torch.device,
        float_dtype: torch.dtype,
        cfg: SearchConfig,
        popart_normalizer: Optional[PopArtNormalizer] = None,
        use_rebel_features: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.float_dtype = float_dtype
        self.bet_bins = bet_bins
        self.branching = cfg.branching
        self.depth = cfg.depth
        self.popart = popart_normalizer
        self.root_batch_size = batch_size
        self.belief_samples = max(1, cfg.belief_samples if use_rebel_features else 1)
        self.num_players = 2
        self._sample_cards: Optional[torch.Tensor] = None
        self._sample_weights: Optional[torch.Tensor] = None
        self._pbs_beliefs: Optional[torch.Tensor] = None
        self._roots: Optional[torch.Tensor] = None
        self.all_cards = torch.arange(52, device=self.device)
        self.combos = hand_combos_tensor(device=self.device)
        self.combo_lookup = combo_lookup_tensor(device=self.device)
        # total nodes M = B * sum_{i=0..D} branching^i
        factors = torch.tensor([self.branching**i for i in range(self.depth + 1)])
        total_factor = int(factors.sum().item())
        self.batch_size = batch_size
        self.M = batch_size * total_factor

        # Allocate internal env with same parameters as env_proto
        self.env = HUNLTensorEnv(
            num_envs=self.M,
            starting_stack=env_proto.starting_stack,
            sb=env_proto.sb,
            bb=env_proto.bb,
            default_bet_bins=self.bet_bins,
            device=device,
            rng=env_proto.rng,
            float_dtype=float_dtype,
            debug_step_table=False,
            flop_showdown=env_proto.flop_showdown,
        )

        self.use_rebel_features = use_rebel_features
        if self.use_rebel_features:
            self.tsb = None
            self.rebel_encoder = RebelFeatureEncoder(
                env=self.env,
                device=self.device,
                dtype=self.float_dtype,
            )
        else:
            # Token sequence builder aligned to env
            self.tsb = TokenSequenceBuilder(
                tensor_env=self.env,
                sequence_length=sequence_length,
                bet_bins=self.bet_bins,
                device=self.device,
                float_dtype=self.float_dtype,
            )
            self.rebel_encoder = None

        # Track which rows are initialized with valid state
        self.initialized = torch.zeros(self.M, dtype=torch.bool, device=self.device)

        # Depth offsets to slice nodes per depth
        self.depth_offsets = [0]
        accum = 0
        for i in range(self.depth + 1):
            span = batch_size * (self.branching**i)
            accum += span
            self.depth_offsets.append(accum)

    def depth_slice(self, d: int) -> slice:
        return slice(self.depth_offsets[d], self.depth_offsets[d + 1])

    def seed_roots(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
        src_tokens: Optional[Union[StructuredEmbeddingData, CNNEmbeddingData]] = None,
    ) -> torch.Tensor:
        """Copy root minibatch states into depth-0 slice and seed tokens.

        Returns the tensor of root indices in the manager env.
        """
        assert src_indices.shape[0] == self.batch_size
        roots = torch.arange(
            self.depth_offsets[0], self.depth_offsets[1], device=self.device
        )
        # Validate source rows then copy
        src_env.sanity_check(indices=src_indices, label="cfr seed_roots src")
        self.env.copy_state_from(src_env, src_indices, roots, copy_deck=True)
        # Validate destination rows after copy
        self.env.sanity_check(indices=roots, label="cfr seed_roots dst")

        # Seed tokens for roots
        if src_tokens is not None and self.tsb is not None:
            if isinstance(src_tokens, StructuredEmbeddingData):
                self.tsb.copy_from_structured(roots, src_tokens)
            elif isinstance(src_tokens, CNNEmbeddingData):
                raise NotImplementedError(
                    "CNN replay seeding not supported in CFR manager"
                )
        elif src_tokens is not None and self.tsb is None:
            raise ValueError(
                "Token data provided but manager is configured for rebel features"
            )

        # Mark roots initialized
        self.initialized[roots] = True
        self._roots = roots
        self._sample_cards = None
        self._sample_weights = None
        self._pbs_beliefs = None
        if self.use_rebel_features and self.belief_samples > 0:
            self._prepare_belief_samples(roots)
        return roots

    def _prepare_belief_samples(self, roots: torch.Tensor) -> None:
        num_roots = roots.numel()
        if num_roots == 0:
            return
        cards = torch.full(
            (self.belief_samples, num_roots, self.num_players, 2),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        weights = torch.zeros(
            (self.belief_samples, num_roots),
            dtype=self.float_dtype,
            device=self.device,
        )
        actual_cards = self.env.hole_indices[roots]  # [num_roots, num_players, 2]
        board_cards = self.env.board_indices[roots]

        for root_offset in range(num_roots):
            board = board_cards[root_offset]
            board_known = board[board >= 0].to(torch.long)
            base_mask = mask_conflicting_combos(board_known, device=self.device)
            valid_combo_indices = torch.where(base_mask)[0]
            if valid_combo_indices.numel() == 0:
                continue

            samples: list[tuple[int, int]] = []
            seen_pairs: set[tuple[int, int]] = set()

            # Include actual hole cards if both players known
            actual_pair: list[int] = []
            valid_actual = True
            for player in range(self.num_players):
                hole = actual_cards[root_offset, player]
                if (hole >= 0).all():
                    combo_idx = int(self.combo_lookup[hole[0], hole[1]].item())
                    actual_pair.append(combo_idx)
                else:
                    valid_actual = False
                    break
            if valid_actual:
                pair_tuple = (actual_pair[0], actual_pair[1])
                samples.append(pair_tuple)
                seen_pairs.add(pair_tuple)

            max_attempts = max(10 * self.belief_samples, 20)
            attempts = 0
            while len(samples) < self.belief_samples and attempts < max_attempts:
                attempts += 1
                perm0 = torch.randperm(
                    valid_combo_indices.numel(),
                    generator=self.env.rng,
                    device=self.device,
                )
                idx0 = int(valid_combo_indices[perm0[0]].item())
                occupied = self.combos[idx0].to(torch.long)
                if board_known.numel() > 0:
                    occupied = torch.cat((occupied, board_known))
                mask_p1 = mask_conflicting_combos(occupied, device=self.device)
                valid_p1 = torch.where(mask_p1)[0]
                if valid_p1.numel() == 0:
                    continue
                perm1 = torch.randperm(
                    valid_p1.numel(), generator=self.env.rng, device=self.device
                )
                idx1 = int(valid_p1[perm1[0]].item())
                pair = (idx0, idx1)
                if pair in seen_pairs:
                    continue
                samples.append(pair)
                seen_pairs.add(pair)

            count = len(samples)
            if count == 0:
                continue
            weight = 1.0 / float(count)
            for sample_idx, (idx0, idx1) in enumerate(samples):
                cards[sample_idx, root_offset, 0] = self.combos[idx0]
                cards[sample_idx, root_offset, 1] = self.combos[idx1]
                weights[sample_idx, root_offset] = weight

        self._sample_cards = cards
        self._sample_weights = weights
        if self.rebel_encoder is not None:
            belief_tensors = []
            for player in range(self.num_players):
                belief = self.rebel_encoder.aggregate_beliefs_from_samples(
                    cards[:, :, player], weights
                )
                belief_tensors.append(belief)
            self._pbs_beliefs = torch.stack(belief_tensors, dim=0)
        else:
            self._pbs_beliefs = None

    def _apply_belief_sample(self, sample_idx: int) -> None:
        if (
            self._sample_cards is None
            or self._sample_weights is None
            or self._roots is None
        ):
            return
        weights = self._sample_weights[sample_idx]
        active_mask = weights > 0
        if not torch.any(active_mask):
            return
        roots = self._roots[active_mask]
        cards = self._sample_cards[sample_idx, active_mask]
        for player in range(self.num_players):
            player_cards = cards[:, player]
            players = torch.full(
                (roots.shape[0],),
                player,
                dtype=torch.long,
                device=self.device,
            )
            self._set_player_hole(roots, players, player_cards)

    def _set_player_hole(
        self, idxs: torch.Tensor, players: torch.Tensor, cards: torch.Tensor
    ) -> None:
        if idxs.numel() == 0:
            return
        for env_idx, player, pair in zip(
            idxs.tolist(), players.tolist(), cards.tolist()
        ):
            c0, c1 = pair
            if c0 < 0 or c1 < 0:
                continue
            p = int(player)
            env_idx_int = int(env_idx)
            card0 = int(c0)
            card1 = int(c1)
            self.env.hole_indices[env_idx_int, p, 0] = card0
            self.env.hole_indices[env_idx_int, p, 1] = card1
            self.env.hole_onehot[env_idx_int, p, 0] = self.env.card_onehot_cache[card0]
            self.env.hole_onehot[env_idx_int, p, 1] = self.env.card_onehot_cache[card1]
            # Rebuild deck to maintain uniqueness and random remainder
            p0_cards = self.env.hole_indices[env_idx_int, 0]
            p1_cards = self.env.hole_indices[env_idx_int, 1]
            used_mask = torch.zeros(52, dtype=torch.bool, device=self.device)
            used_mask[p0_cards] = True
            used_mask[p1_cards] = True
            board_cards = self.env.board_indices[env_idx_int]
            board_known = board_cards[board_cards >= 0]
            if board_known.numel() > 0:
                used_mask[board_known] = True
            available = self.all_cards[~used_mask]
            assert available.numel() >= 5, "Not enough cards available to rebuild deck"
            perm = torch.randperm(
                available.numel(), generator=self.env.rng, device=self.device
            )
            extra = available[perm[:5]]
            new_deck = torch.cat((p0_cards, p1_cards, extra), dim=0)
            self.env.deck[env_idx_int, :9] = new_deck
            self.env.deck_pos[env_idx_int] = 4

    def expand_children(self, parents: torch.Tensor, depth: int) -> torch.Tensor:
        """Clone parent nodes into 4 children each and return child indices."""
        assert parents.numel() == self.batch_size * (self.branching**depth)
        # Children range at next depth
        children = torch.arange(
            self.depth_offsets[depth + 1],
            self.depth_offsets[depth + 2],
            device=self.device,
        )
        # Map each parent to 4 consecutive children
        rep_parents = parents.repeat_interleave(self.branching)
        # Shape sanity
        assert (
            children.numel() == rep_parents.numel() == parents.numel() * self.branching
        ), "children/rep_parents size mismatch"
        assert (
            children[0] == self.depth_offsets[depth + 1]
            and children[-1] == self.depth_offsets[depth + 2] - 1
        ), "children slice bounds mismatch"
        # Sanity: parents rows valid before cloning
        self.env.sanity_check(
            indices=rep_parents, label=f"cfr clone depth {depth} parents"
        )
        self.env.clone_states(children, rep_parents)
        # Sanity: children rows valid after cloning
        self.env.sanity_check(
            indices=children, label=f"cfr clone depth {depth} children"
        )
        if self.tsb is not None:
            self.tsb.clone_tokens(children, rep_parents)
        # Mark children initialized
        self.initialized[children] = self.initialized[rep_parents]
        # Verify critical fields cloned correctly (spot-check)
        assert torch.equal(
            self.env.to_act[children], self.env.to_act[rep_parents]
        ), "to_act mismatch after clone_states"
        assert torch.equal(
            self.env.street[children], self.env.street[rep_parents]
        ), "street mismatch after clone_states"
        # Deck/deck_pos must exactly match after clone
        assert torch.equal(
            self.env.deck[children], self.env.deck[rep_parents]
        ), "deck mismatch after clone_states"
        assert torch.equal(
            self.env.deck_pos[children], self.env.deck_pos[rep_parents]
        ), "deck_pos mismatch after clone_states"
        return children

    def encode_states(
        self, player: int, idxs: torch.Tensor
    ) -> Union[StructuredEmbeddingData, torch.Tensor]:
        if idxs.numel() == 0:
            if self.use_rebel_features:
                return torch.empty(
                    0,
                    self.rebel_encoder.feature_dim,
                    device=self.device,
                    dtype=self.float_dtype,
                )
            return self.tsb.encode_tensor_states(player=player, idxs=idxs)
        if self.use_rebel_features:
            agents = torch.full(
                (idxs.numel(),),
                player,
                dtype=torch.long,
                device=self.device,
            )
            hero_belief = self.get_beliefs_for_player(player, idxs)
            opp_belief = self.get_beliefs_for_player(1 - player, idxs)
            return self.rebel_encoder.encode(
                idxs,
                agents,
                hero_beliefs=hero_belief,
                opp_beliefs=opp_belief,
            )
        return self.tsb.encode_tensor_states(player=player, idxs=idxs)

    def get_beliefs_for_player(
        self, player: int, idxs: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if (
            not self.use_rebel_features
            or self.rebel_encoder is None
            or self._pbs_beliefs is None
        ):
            return None
        if idxs.numel() == 0:
            return torch.empty(
                0,
                self.rebel_encoder.belief_dim,
                device=self.device,
                dtype=self.float_dtype,
            )

        root_ids = self._resolve_root_indices(idxs)
        belief = self._pbs_beliefs[player, root_ids]
        return belief.to(self.device, dtype=self.float_dtype)

    def _resolve_root_indices(self, idxs: torch.Tensor) -> torch.Tensor:
        root_ids = torch.empty_like(idxs, dtype=torch.long)
        for depth in range(self.depth + 1):
            sl = self.depth_slice(depth)
            mask = (idxs >= sl.start) & (idxs < sl.stop)
            if not torch.any(mask):
                continue
            local = idxs[mask] - sl.start
            nodes_per_root = self.branching**depth
            root_ids[mask] = local // nodes_per_root
        return root_ids

    def _select_hand_value(
        self, rows: torch.Tensor, hand_values: torch.Tensor, player: int
    ) -> torch.Tensor:
        if hand_values is None:
            raise ValueError("Model must return hand_values when using ReBeL features.")
        cards = self.env.hole_indices[rows, player]
        c0 = torch.minimum(cards[:, 0], cards[:, 1])
        c1 = torch.maximum(cards[:, 0], cards[:, 1])
        combos_idx = self.combo_lookup[c0, c1].to(torch.long)
        row_count = hand_values.shape[0]
        result = torch.zeros(row_count, dtype=hand_values.dtype, device=self.device)
        valid = combos_idx >= 0
        if torch.any(valid):
            row_ids = torch.arange(row_count, device=self.device, dtype=torch.long)
            result[valid] = hand_values[row_ids[valid], player, combos_idx[valid]]
        return result

    def reset_for_new_search(self) -> None:
        """Clear cached buffers so the manager can be reused across calls."""
        self.env.reset()
        if self.tsb is not None:
            self.tsb.reset()
        self.initialized.fill_(False)
        self._roots = None
        self._sample_cards = None
        self._sample_weights = None
        self._pbs_beliefs = None

    def legal_mask_full(self, idxs: torch.Tensor) -> torch.Tensor:
        return self.env.legal_mask_bins_for(idxs, self.bet_bins)

    def step_collapsed(
        self, idxs: torch.Tensor, collapsed_bins: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map collapsed actions to env step and execute.

        collapsed_bins: 0=fold,1=call,2=betpot(aggregate presets),3=all-in
        """
        B = len(self.bet_bins) + 3
        mapped = torch.full_like(collapsed_bins, -1)
        mapped[collapsed_bins == 0] = 0
        mapped[collapsed_bins == 1] = 1
        mapped[collapsed_bins == 3] = B - 1
        # For betpot, use 1x pot raise logic as per plan
        betpot_rows = torch.where(collapsed_bins == 2)[0]
        if betpot_rows.numel() > 0:
            # Use legal_bins_amounts_and_mask([1.0]) for 1x pot raise mapping
            amounts_1x, mask_1x = self.env.legal_bins_amounts_and_mask([1.0])
            # Find the bin index that corresponds to 1x pot (should be first legal preset)
            preset_mask = mask_1x[betpot_rows, 2:-1]  # restrict to presets 2..B-2
            # Choose first legal preset as fallback for 1x pot
            first_idx = torch.argmax(preset_mask.int(), dim=1)
            mapped[betpot_rows] = first_idx + 2

        return self.env.step_bins(mapped, bet_bins=self.bet_bins)

    @staticmethod
    def collapse_policy_full_to_4(prob_full: torch.Tensor) -> torch.Tensor:
        """Collapse full-bin probs [N, B] to 4-action probs [N, 4]."""
        N, B = prob_full.shape
        out = torch.zeros(N, 4, dtype=prob_full.dtype, device=prob_full.device)
        out[:, 0] = prob_full[:, 0]
        out[:, 1] = prob_full[:, 1]
        if B > 3:
            out[:, 2] = prob_full[:, 2:-1].sum(dim=1)
        out[:, 3] = prob_full[:, -1]
        s = out.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return out / s

    @staticmethod
    def expand_collapsed_to_full(
        prob_collapsed: torch.Tensor, legal_mask_full: torch.Tensor
    ) -> torch.Tensor:
        """Distribute collapsed 4-action probabilities back over full action bins."""
        N, B = legal_mask_full.shape
        out = torch.zeros(
            N, B, dtype=prob_collapsed.dtype, device=prob_collapsed.device
        )
        out[:, 0] = prob_collapsed[:, 0]
        out[:, 1] = prob_collapsed[:, 1]
        out[:, -1] = prob_collapsed[:, 3]
        if B > 3:
            presets_mask = legal_mask_full[:, 2:-1]
            counts = presets_mask.sum(dim=1, keepdim=True).clamp_min(1)
            share = prob_collapsed[:, 2].unsqueeze(1) / counts
            out[:, 2:-1] = torch.where(
                presets_mask,
                share.expand_as(presets_mask),
                torch.zeros_like(presets_mask, dtype=prob_collapsed.dtype),
            )
        return out

    def logits_to_collapsed_probs(
        self, logits_full: torch.Tensor, legal_mask_full: torch.Tensor
    ) -> torch.Tensor:
        """Softmax over full legal set, then collapse to 4 actions."""
        masked_logits = logits_full.masked_fill(~legal_mask_full, float("-inf"))
        prob_full = F.softmax(masked_logits, dim=1)
        return self.collapse_policy_full_to_4(prob_full)

    def _current_terminal_reward(self, idxs: torch.Tensor) -> torch.Tensor:
        # Rewards are from p0 perspective: replicate finish_and_assign_rewards math without mutation
        pot = self.env.pot[idxs].to(self.float_dtype)
        my_stack = self.env.stacks[idxs, 0].to(self.float_dtype)
        winners = self.env.winner[idxs]
        pot_share = torch.where(
            winners == 0,
            pot,
            torch.where(winners == 2, pot / 2.0, torch.zeros_like(pot)),
        )
        return (my_stack + pot_share - float(self.env.starting_stack)) / self.env.scale

    def expand_and_step_all_branches(
        self, parents: torch.Tensor, depth: int
    ) -> torch.Tensor:
        """Expand parents to 4 children each, add action tokens pre-step, then step env on children.

        Returns children indices.
        """
        # (debug assertions removed)

        children = self.expand_children(parents, depth)
        # Build collapsed action per child: [0,1,2,3] per parent
        pattern = torch.tensor([0, 1, 2, 3], device=self.device, dtype=torch.long)
        collapsed_bins = pattern.repeat(parents.numel())
        # Pre-step legal masks and amounts for tokens
        child_masks = self.legal_mask_full(children)
        # Sanity: child_masks shape and bin dimension
        B = len(self.bet_bins) + 3
        assert child_masks.shape == (
            children.numel(),
            B,
        ), f"child_masks shape mismatch: got {tuple(child_masks.shape)}, expected ({children.numel()}, {B})"
        amounts, full_mask = self.env.legal_bins_amounts_and_mask(self.bet_bins)
        # Map collapsed->bin index
        B = len(self.bet_bins) + 3
        assert (
            B == self.env.num_bet_bins
        ), f"num_bet_bins mismatch: computed {B} from bet_bins, env has {self.env.num_bet_bins}"
        mapped = torch.full_like(collapsed_bins, -1)
        # Stop if parents are done: mark children as no-op (-1)
        parent_done = self.env.done[parents]
        child_done = parent_done.repeat_interleave(self.branching)
        nd = ~child_done
        # Use legality guard for each collapsed action
        # Fold legal where child_masks[:,0]
        fold_rows = (collapsed_bins == 0) & nd & child_masks[:, 0]
        mapped[fold_rows] = 0
        # Call legal where child_masks[:,1]
        call_rows = (collapsed_bins == 1) & nd & child_masks[:, 1]
        mapped[call_rows] = 1
        # All-in legal where child_masks[:,-1]
        allin_rows = (collapsed_bins == 3) & nd & child_masks[:, -1]
        mapped[allin_rows] = B - 1
        betpot_rows = torch.where(collapsed_bins == 2)[0]
        if betpot_rows.numel() > 0:
            bp_rows_nd = betpot_rows[nd[betpot_rows]]
            if bp_rows_nd.numel() > 0:
                # Use 1x pot raise logic for betpot actions
                amounts_1x, mask_1x = self.env.legal_bins_amounts_and_mask([1.0])
                preset_mask = mask_1x[
                    children[bp_rows_nd], 2:-1
                ]  # restrict to presets 2..B-2
                any_preset = preset_mask.any(dim=1)
                valid_rows = bp_rows_nd[any_preset]
                if valid_rows.numel() > 0:
                    first_idx = torch.argmax(preset_mask[any_preset].int(), dim=1)
                    mapped[valid_rows] = first_idx + 2
        # Ensure every live child has a concrete legal action so token/history stays canonical.
        fallback_rows = torch.where((mapped < 0) & nd)[0]
        if fallback_rows.numel() > 0:
            legal_subset = child_masks[fallback_rows]
            assert legal_subset.any(
                dim=1
            ).all(), "No legal actions available for fallback"
            fallback_indices = torch.argmax(legal_subset.int(), dim=1)
            mapped[fallback_rows] = fallback_indices
        action_ids = mapped
        token_streets = self.env.street[children]
        action_amounts = torch.zeros_like(children, dtype=torch.long)
        # gather amounts for mapped bins
        gather_mask = (action_ids >= 0) & nd
        if gather_mask.any():
            rows = children[gather_mask]
            cols = action_ids[gather_mask]
            action_amounts[gather_mask] = amounts[rows, cols]
        actors = self.env.to_act[children]
        # Add tokens pre-step
        if self.tsb is not None:
            nd_rows = children[nd]
            if nd_rows.numel() > 0:
                # Only add tokens for valid mapped actions (>= 0)
                valid_token_mask = (action_ids >= 0) & nd
                if valid_token_mask.any():
                    self.tsb.add_action(
                        idxs=children[valid_token_mask],
                        actors=actors[valid_token_mask],
                        action_ids=action_ids[valid_token_mask],
                        legal_masks=child_masks[valid_token_mask],
                        action_amounts=action_amounts[valid_token_mask],
                        token_streets=token_streets[valid_token_mask],
                    )
        # Step env
        # Build full-size action vector for step (others -1)
        full_actions = torch.full((self.M,), -1, dtype=torch.long, device=self.device)
        full_actions[children] = mapped

        # (debug assertions removed)

        self.env.step_bins(full_actions, bet_bins=self.bet_bins)
        # Return children
        return children

    def build_tree_tensors(
        self, model
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand full tree and compute per-node logits, legal masks, values, and to_act.

        Returns (logits_full[M,B], legal_mask_full[M,B], values[M], to_act[M]).
        """
        M = self.M
        B = len(self.bet_bins) + 3
        logits_full = torch.zeros(M, B, dtype=self.float_dtype, device=self.device)
        legal_full = torch.zeros(M, B, dtype=torch.bool, device=self.device)
        values = torch.zeros(M, dtype=self.float_dtype, device=self.device)
        to_act = torch.zeros(M, dtype=torch.long, device=self.device)

        # Depth 0 roots already seeded with context
        for d in range(self.depth + 1):
            sl = self.depth_slice(d)
            idxs = torch.arange(sl.start, sl.stop, device=self.device)
            if idxs.numel() == 0:
                continue
            # Ensure all rows at this depth are initialized before use
            assert self.initialized[idxs].all(), (
                f"Uninitialized rows at depth {d}: "
                f"count={(~self.initialized[idxs]).sum().item()}"
            )
            # Snapshot context at this depth for non-terminal nodes only
            not_done_mask = ~self.env.done[idxs]
            not_done_rows = idxs[not_done_mask]
            if self.tsb is not None and not_done_rows.numel() > 0:
                self.tsb.add_context(not_done_rows)
            # Legal masks and logits
            legal = self.legal_mask_full(idxs)
            legal_full[sl] = legal
            to_act_slice = self.env.to_act[idxs]
            to_act[sl] = to_act_slice
            # Evaluate model logits/values respecting player perspective
            ta = to_act_slice
            p0_rows = idxs[(ta == 0) & not_done_mask]
            p1_rows = idxs[(ta == 1) & not_done_mask]
            chunk_size = max(1, 2 * self.batch_size)
            if p0_rows.numel() > 0:
                for start in range(0, p0_rows.numel(), chunk_size):
                    rows = p0_rows[start : start + chunk_size]
                    emb = self.encode_states(player=0, idxs=rows)
                    out = model(emb)
                    logits = out.policy_logits
                    if logits.dim() == 3:
                        logits = logits.mean(dim=-2)
                    logits_full[rows] = logits.float()
                # Assign values for non-terminal leaves at last depth later
            if p1_rows.numel() > 0:
                for start in range(0, p1_rows.numel(), chunk_size):
                    rows = p1_rows[start : start + chunk_size]
                    emb = self.encode_states(player=1, idxs=rows)
                    out = model(emb)
                    logits = out.policy_logits
                    if logits.dim() == 3:
                        logits = logits.mean(dim=-2)
                    logits_full[rows] = logits.float()

            # Expand and step children for next depth
            if d < self.depth:
                parents = idxs
                _children = self.expand_and_step_all_branches(parents, d)

        # Set terminal node values where done
        all_idxs = torch.arange(0, M, device=self.device)
        done_rows = all_idxs[self.env.done]
        if done_rows.numel() > 0:
            values[done_rows] = self._current_terminal_reward(done_rows)

        # For remaining (non-terminal) nodes at deepest depth, use model values
        leaf_sl = self.depth_slice(self.depth)
        leaf_idxs = torch.arange(leaf_sl.start, leaf_sl.stop, device=self.device)
        leaf_not_done = leaf_idxs[~self.env.done[leaf_idxs]]
        if leaf_not_done.numel() > 0:
            to_act = self.env.to_act[leaf_not_done]
            p0 = leaf_not_done[to_act == 0]
            p1 = leaf_not_done[to_act == 1]
            chunk_size = max(1, 2 * self.batch_size)
            if p0.numel() > 0:
                for start in range(0, p0.numel(), chunk_size):
                    rows = p0[start : start + chunk_size]
                    emb = self.encode_states(0, rows)
                    out = model(emb)
                    if out.hand_values is None:
                        raise ValueError(
                            "Model must provide hand_values for ReBeL search."
                        )
                    hv = out.hand_values.float()
                    selected = self._select_hand_value(
                        rows=rows, hand_values=hv, player=0
                    )
                    values[rows] = selected.to(self.float_dtype)
            if p1.numel() > 0:
                for start in range(0, p1.numel(), chunk_size):
                    rows = p1[start : start + chunk_size]
                    emb = self.encode_states(1, rows)
                    out = model(emb)
                    if out.hand_values is None:
                        raise ValueError(
                            "Model must provide hand_values for ReBeL search."
                        )
                    hv = out.hand_values.float()
                    selected = self._select_hand_value(
                        rows=rows, hand_values=hv, player=1
                    )
                    values[rows] = (-selected).to(self.float_dtype)

        return logits_full, legal_full, values, to_act

    def _run_search_once(self, model) -> "DCFRResult":
        """Single DCFR run for current environment state."""
        logits_full, legal_full, values_full, to_act = self.build_tree_tensors(model)

        def _leaf_value_cb_factory():
            # Recompute deepest leaf values each iteration using current model
            leaf_sl = self.depth_slice(self.depth)
            leaf_idxs = torch.arange(leaf_sl.start, leaf_sl.stop, device=self.device)

            def _cb(_it: int) -> torch.Tensor:
                # Start with original values_full and overwrite leaf slice
                updated_values = values_full.clone()
                leaf_not_done = leaf_idxs[~self.env.done[leaf_idxs]]
                if leaf_not_done.numel() == 0:
                    return updated_values
                to_act_leaf = self.env.to_act[leaf_not_done]
                chunk_size = max(1, 2 * self.root_batch_size)
                # Player 0
                p0 = leaf_not_done[to_act_leaf == 0]
                if p0.numel() > 0:
                    for start in range(0, p0.numel(), chunk_size):
                        rows = p0[start : start + chunk_size]
                        emb = self.encode_states(0, rows)
                        out = model(emb)
                        if out.hand_values is None:
                            raise ValueError(
                                "Model must provide hand_values for ReBeL search."
                            )
                        hv = out.hand_values.float()
                        selected = self._select_hand_value(
                            rows=rows, hand_values=hv, player=0
                        )
                        updated_values[rows] = selected.to(self.float_dtype)
                # Player 1
                p1 = leaf_not_done[to_act_leaf == 1]
                if p1.numel() > 0:
                    for start in range(0, p1.numel(), chunk_size):
                        rows = p1[start : start + chunk_size]
                        emb = self.encode_states(1, rows)
                        out = model(emb)
                        if out.hand_values is None:
                            raise ValueError(
                                "Model must provide hand_values for ReBeL search."
                            )
                        hv = out.hand_values.float()
                        selected = self._select_hand_value(
                            rows=rows, hand_values=hv, player=1
                        )
                        updated_values[rows] = (-selected).to(self.float_dtype)
                return updated_values

            return _cb

        leaf_cb = _leaf_value_cb_factory()

        res = run_dcfr(
            logits_full=logits_full,
            legal_mask_full=legal_full,
            values=values_full,
            to_act=to_act,
            depth_offsets=self.depth_offsets,
            depth=self.depth,
            iterations=self.cfg.iterations,
            alpha=self.cfg.dcfr_alpha,
            beta=self.cfg.dcfr_beta,
            gamma=self.cfg.dcfr_gamma,
            include_average=self.cfg.include_average_policy,
            leaf_value_callback=leaf_cb,
        )

        root_sl = slice(self.depth_offsets[0], self.depth_offsets[1])
        root_prior = self.logits_to_collapsed_probs(
            logits_full[root_sl], legal_full[root_sl]
        )
        root_avg = (
            res.root_policy_avg_collapsed
            if res.root_policy_avg_collapsed is not None
            else res.root_policy_collapsed
        )
        root_sampled = (
            res.root_policy_sampled_collapsed
            if res.root_policy_sampled_collapsed is not None
            else res.root_policy_collapsed
        )
        collapsed_target = res.root_policy_collapsed

        # Error handling: check for valid CFR target
        if collapsed_target is None or collapsed_target.numel() == 0:
            print("Warning: CFR returned empty target, falling back to PPO")
            zeros = torch.zeros(
                root_prior.size(0), dtype=self.float_dtype, device=self.device
            )
            ones = torch.ones(
                root_prior.size(0), dtype=self.float_dtype, device=self.device
            )
            return DCFRResult(
                root_policy_collapsed=root_prior,
                root_policy_avg_collapsed=root_prior,
                root_policy_sampled_collapsed=root_prior,
                root_values_p0=zeros,
                root_values_actor=zeros,
                root_sample_weights=ones,
                regrets=None,
            )
        elif torch.isnan(collapsed_target).any() or torch.isinf(collapsed_target).any():
            print("Warning: CFR returned invalid target (NaN/Inf), falling back to PPO")
            zeros = torch.zeros(
                root_prior.size(0), dtype=self.float_dtype, device=self.device
            )
            ones = torch.ones(
                root_prior.size(0), dtype=self.float_dtype, device=self.device
            )
            return DCFRResult(
                root_policy_collapsed=root_prior,
                root_policy_avg_collapsed=root_prior,
                root_policy_sampled_collapsed=root_prior,
                root_values_p0=zeros,
                root_values_actor=zeros,
                root_sample_weights=ones,
                regrets=None,
            )
        elif (collapsed_target < 0).any():
            print("Warning: CFR returned negative probabilities, falling back to PPO")
            zeros = torch.zeros(
                root_prior.size(0), dtype=self.float_dtype, device=self.device
            )
            ones = torch.ones(
                root_prior.size(0), dtype=self.float_dtype, device=self.device
            )
            return DCFRResult(
                root_policy_collapsed=root_prior,
                root_policy_avg_collapsed=root_prior,
                root_policy_sampled_collapsed=root_prior,
                root_values_p0=zeros,
                root_values_actor=zeros,
                root_sample_weights=ones,
                regrets=None,
            )

        if res.root_policy_avg_collapsed is None:
            res.root_policy_avg_collapsed = root_avg
        if res.root_sample_weights is None:
            res.root_sample_weights = torch.ones(
                root_prior.size(0), dtype=self.float_dtype, device=self.device
            )

        return res

    def run_search(self, model) -> "DCFRResult":
        if not (
            self.use_rebel_features
            and self._sample_cards is not None
            and self._sample_weights is not None
        ):
            return self._run_search_once(model)

        batch = self.batch_size
        belief_dim = self.rebel_encoder.belief_dim if self.rebel_encoder else 0
        policy_sum = torch.zeros(batch, 4, dtype=self.float_dtype, device=self.device)
        policy_avg_sum = torch.zeros_like(policy_sum)
        value_p0_sum = torch.zeros(batch, dtype=self.float_dtype, device=self.device)
        value_actor_sum = torch.zeros_like(value_p0_sum)
        total_weights = torch.zeros(batch, dtype=self.float_dtype, device=self.device)
        hand_value_sum = torch.zeros(
            batch,
            self.num_players,
            belief_dim,
            dtype=self.float_dtype,
            device=self.device,
        )
        hand_weight_sum = torch.zeros_like(hand_value_sum)

        for sample_idx in range(self.belief_samples):
            weights = self._sample_weights[sample_idx]
            if not torch.any(weights > 0):
                continue
            self._apply_belief_sample(sample_idx)
            res = self._run_search_once(model)
            policy_sum += weights.unsqueeze(1) * res.root_policy_collapsed
            avg_component = (
                res.root_policy_avg_collapsed
                if res.root_policy_avg_collapsed is not None
                else res.root_policy_collapsed
            )
            policy_avg_sum += weights.unsqueeze(1) * avg_component
            value_p0_sum += weights * res.root_values_p0
            value_actor_sum += weights * res.root_values_actor
            total_weights += weights

            active = weights > 0
            if not torch.any(active):
                continue
            active_indices = torch.where(active)[0]
            cards_active = self._sample_cards[sample_idx, active_indices]
            values_p0_active = res.root_values_p0[active_indices]
            weights_active = weights[active_indices]
            for player in range(self.num_players):
                combos_idx = self.combo_lookup[
                    cards_active[:, player, 0], cards_active[:, player, 1]
                ].to(torch.long)
                player_values = values_p0_active if player == 0 else -values_p0_active
                scatter_values = weights_active * player_values
                batch_idx = active_indices
                player_idx = torch.full_like(
                    batch_idx, player, dtype=torch.long, device=self.device
                )
                hand_value_sum.index_put_(
                    (batch_idx, player_idx, combos_idx),
                    scatter_values,
                    accumulate=True,
                )
                hand_weight_sum.index_put_(
                    (batch_idx, player_idx, combos_idx),
                    weights_active,
                    accumulate=True,
                )

        # Restore environment to first sample (typically actual cards) for downstream use.
        self._apply_belief_sample(0)

        weight_mask = total_weights > 0
        totals = total_weights.unsqueeze(1).clamp_min(1e-12)
        default_policy = torch.full(
            (batch, 4), 0.25, dtype=self.float_dtype, device=self.device
        )
        policy_mean = torch.where(
            weight_mask.unsqueeze(1),
            policy_sum / totals,
            default_policy,
        )
        policy_mean = policy_mean / policy_mean.sum(dim=1, keepdim=True).clamp_min(
            1e-12
        )

        policy_avg_mean = torch.where(
            weight_mask.unsqueeze(1),
            policy_avg_sum / totals,
            default_policy,
        )
        policy_avg_mean = policy_avg_mean / policy_avg_mean.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)

        value_p0_mean = torch.where(
            weight_mask,
            value_p0_sum / total_weights.clamp_min(1e-12),
            torch.zeros_like(value_p0_sum),
        )
        value_actor_mean = torch.where(
            weight_mask,
            value_actor_sum / total_weights.clamp_min(1e-12),
            torch.zeros_like(value_actor_sum),
        )
        hand_value_mean = torch.zeros_like(hand_value_sum)
        positive_mask = hand_weight_sum > 0
        hand_value_mean[positive_mask] = (
            hand_value_sum[positive_mask] / hand_weight_sum[positive_mask]
        )

        sample_weights = torch.where(
            weight_mask,
            total_weights,
            torch.ones_like(total_weights),
        )

        return DCFRResult(
            root_policy_collapsed=policy_mean,
            root_policy_avg_collapsed=policy_avg_mean,
            root_values_p0=value_p0_mean,
            root_values_actor=value_actor_mean,
            root_sample_weights=sample_weights,
            root_hand_values=hand_value_mean,
            root_hand_value_weights=hand_weight_sum,
            regrets=None,
        )

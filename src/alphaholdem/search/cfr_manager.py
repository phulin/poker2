from __future__ import annotations

from typing import Optional, Tuple

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.cnn.cnn_embedding_data import CNNEmbeddingData
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.token_sequence_builder import TokenSequenceBuilder
from alphaholdem.core.structured_config import SearchConfig
import torch.nn.functional as F
from alphaholdem.search.dcfr import run_dcfr
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
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.float_dtype = float_dtype
        self.bet_bins = bet_bins
        self.branching = cfg.branching
        self.depth = cfg.depth
        self.popart = popart_normalizer
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

        # Token sequence builder aligned to env
        self.tsb = TokenSequenceBuilder(
            tensor_env=self.env,
            sequence_length=sequence_length,
            bet_bins=self.bet_bins,
            device=self.device,
            float_dtype=self.float_dtype,
        )

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
        src_tokens: StructuredEmbeddingData | CNNEmbeddingData,
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
        if isinstance(src_tokens, StructuredEmbeddingData):
            self.tsb.copy_from_structured(roots, src_tokens)

        # Mark roots initialized
        self.initialized[roots] = True
        return roots

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

    def encode_states(self, player: int, idxs: torch.Tensor) -> StructuredEmbeddingData:
        return self.tsb.encode_tensor_states(player=player, idxs=idxs)

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

        rewards, dones, _, _, _ = self.env.step_bins(
            full_actions, bet_bins=self.bet_bins
        )
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
            if not_done_rows.numel() > 0:
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
                    logits_full[rows] = out.policy_logits.float()
                # Assign values for non-terminal leaves at last depth later
            if p1_rows.numel() > 0:
                for start in range(0, p1_rows.numel(), chunk_size):
                    rows = p1_rows[start : start + chunk_size]
                    emb = self.encode_states(player=1, idxs=rows)
                    out = model(emb)
                    logits_full[rows] = out.policy_logits.float()

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
                    v = out.value.float().squeeze(-1)
                    if self.popart is not None:
                        v = self.popart.denormalize_value(v)
                    values[rows] = v
            if p1.numel() > 0:
                for start in range(0, p1.numel(), chunk_size):
                    rows = p1[start : start + chunk_size]
                    emb = self.encode_states(1, rows)
                    out = model(emb)
                    v = out.value.float().squeeze(-1)
                    if self.popart is not None:
                        v = self.popart.denormalize_value(v)
                    # Flip perspective so values are always from p0 perspective
                    values[rows] = -v

        return logits_full, legal_full, values, to_act

    def run_search(self, model) -> torch.Tensor:
        """Build tree tensors and run DCFR; returns root collapsed policy [B,4]."""
        logits_full, legal_full, values_full, to_act = self.build_tree_tensors(model)
        res = run_dcfr(
            logits_full=logits_full,
            legal_mask_full=legal_full,
            values=values_full,
            to_act=to_act,
            depth_offsets=self.depth_offsets,
            depth=self.depth,
            iterations=self.cfg.iterations,
        )

        root_sl = slice(self.depth_offsets[0], self.depth_offsets[1])
        root_prior = self.logits_to_collapsed_probs(
            logits_full[root_sl], legal_full[root_sl]
        )
        collapsed_target = res.root_policy_collapsed

        # Error handling: check for valid CFR target
        if collapsed_target is None or collapsed_target.numel() == 0:
            print("Warning: CFR returned empty target, falling back to PPO")
            return root_prior
        elif torch.isnan(collapsed_target).any() or torch.isinf(collapsed_target).any():
            print("Warning: CFR returned invalid target (NaN/Inf), falling back to PPO")
            return root_prior
        elif (collapsed_target < 0).any():
            print("Warning: CFR returned negative probabilities, falling back to PPO")
            return root_prior

        return collapsed_target

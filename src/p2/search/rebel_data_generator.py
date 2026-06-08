from collections.abc import Callable

import torch

from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    preflop_class_multiplicity_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_replay import RebelReplayBuffer
from p2.search.cfr_evaluator import CFREvaluator, PublicBeliefState
from p2.utils.profiling import profile


_ENV_STATE_FIELDS = (
    "deck",
    "deck_pos",
    "button",
    "street",
    "to_act",
    "last_to_act",
    "pot",
    "min_raise",
    "actions_this_round",
    "actions_last_round",
    "acted_since_reset",
    "stacks",
    "committed",
    "has_folded",
    "is_allin",
    "starting_stacks",
    "scale",
    "board_onehot",
    "hole_onehot",
    "board_indices",
    "last_board_indices",
    "hole_indices",
    "chips_placed",
    "done",
    "winner",
)


class RebelDataGenerator:
    def __init__(
        self,
        env_proto: HUNLTensorEnv | PBSEnv,
        evaluator: CFREvaluator,
        value_buffer: RebelReplayBuffer | None,
        policy_buffer: RebelReplayBuffer | None,
        warmup: bool = True,
        root_sampler: Callable[[int], PublicBeliefState] | None = None,
        include_pre_chance_value_batches: bool = True,
        store_replay: bool = True,
        sample_continuations: bool = True,
        record_batch_diag: bool = True,
    ):
        self.env_proto = env_proto
        self.evaluator = evaluator
        self.value_buffer = value_buffer
        self.policy_buffer = policy_buffer
        self.device = evaluator.device
        self.target_batch_size = int(evaluator.root_nodes)
        self.root_sampler = root_sampler
        self.include_pre_chance_value_batches = bool(include_pre_chance_value_batches)
        self.store_replay = bool(store_replay)
        self.sample_continuations = bool(sample_continuations)
        self.record_batch_diag = bool(record_batch_diag)
        initial_pbs = self._sample_roots(self.target_batch_size)
        self.current_pbs = initial_pbs
        self.last_extra = 0
        if warmup and self.root_sampler is None:
            self._warmup_current_pbs()

    def _sample_roots(self, target_batch_size: int) -> PublicBeliefState:
        if self.root_sampler is not None:
            return self.root_sampler(target_batch_size)
        return self._new_pbs(target_batch_size)

    @torch.no_grad()
    def _record_batch_diag(self, refilled: bool) -> None:
        """Record the just-solved subgame's root composition into the
        evaluator's stats dict (under ``gen_batch/*``) so the trainer logs it to
        wandb alongside ``action_mix``. Reads the evaluator's current subgame
        env, whose first N rows are the roots that were just solved.

        Used to characterize the bimodal ``action_mix/allin``: these fields let
        us see, per step, whether HI-mode batches are short-SPR / big-pot /
        shove-dominant compositions and whether they coincide with refills.
        """
        env = self.evaluator.env
        N = self.evaluator.root_nodes
        required = (
            "street",
            "pot",
            "stacks",
            "to_act",
            "is_allin",
            "committed",
        )
        if not all(hasattr(env, name) for name in required):
            return
        street = env.street[:N]
        pot = env.pot[:N].float().clamp(min=1.0)
        # Effective stack = min of the two stacks; SPR = eff_stack / pot.
        eff_stack = env.stacks[:N].min(dim=1).values.float()
        spr = eff_stack / pot
        ar = torch.arange(N, device=env.device)
        me = env.to_act[:N]
        me_allin = env.is_allin[ar, me]
        player_ids = torch.arange(env.num_players, device=env.device)
        other_players = player_ids[None, :] != me[:, None]
        opp_allin = (env.is_allin[:N] & other_players).any(dim=1)
        committed = env.committed[:N].float().sum(dim=1)
        commit_frac = committed / (committed + 2.0 * eff_stack + 1.0)
        diag = torch.stack(
            (
                (street == 0).float().mean(),
                (street == 1).float().mean(),
                (street == 2).float().mean(),
                (street == 3).float().mean(),
                spr.median(),
                (spr < 1.0).float().mean(),
                commit_frac.mean(),
                (me_allin | opp_allin).float().mean(),
                (opp_allin & ~me_allin).float().mean(),
            )
        )
        (
            street_preflop,
            street_flop,
            street_turn,
            street_river,
            spr_p50,
            spr_lt1_frac,
            commit_frac_mean,
            any_allin_frac,
            facing_allin_frac,
        ) = diag.detach().cpu().tolist()
        self.evaluator.stats["gen_batch"] = {
            "refilled": float(refilled),
            "street_preflop": float(street_preflop),
            "street_flop": float(street_flop),
            "street_turn": float(street_turn),
            "street_river": float(street_river),
            "spr_p50": float(spr_p50),
            "spr_lt1_frac": float(spr_lt1_frac),
            "commit_frac_mean": float(commit_frac_mean),
            "any_allin_frac": float(any_allin_frac),
            "facing_allin_frac": float(facing_allin_frac),
        }

    def _new_pbs(self, target_batch_size: int) -> PublicBeliefState:
        hand_dim = int(getattr(self.evaluator, "hand_dim", NUM_HANDS))
        if hand_dim == PREFLOP_HANDS:
            prior = preflop_class_multiplicity_tensor(device=self.device).to(
                dtype=getattr(self.evaluator, "float_dtype", torch.float32)
            )
            prior = prior / prior.sum().clamp(min=1.0)
            beliefs = prior.expand(
                target_batch_size,
                self.evaluator.num_players,
                PREFLOP_HANDS,
            ).clone()
        else:
            beliefs = torch.full(
                (target_batch_size, self.evaluator.num_players, hand_dim),
                1.0 / hand_dim,
                device=self.device,
            )
        pbs = PublicBeliefState.from_proto(
            env_proto=self.env_proto,
            beliefs=beliefs,
            num_envs=target_batch_size,
        )
        pbs.env.reset()
        return pbs

    def _extend_pbs(
        self, pbs: PublicBeliefState, desired_size: int
    ) -> PublicBeliefState:
        current_size = pbs.env.N
        indices = torch.arange(current_size, device=self.device)
        new_pbs = self._new_pbs(desired_size)
        new_pbs.env.copy_state_from(pbs.env, indices, indices)
        new_pbs.beliefs[:current_size] = pbs.beliefs
        return new_pbs

    def _advance_current_pbs_once(self) -> None:
        assert self.current_pbs is not None
        n = self.current_pbs.env.N
        if n == 0:
            return
        root_indices = torch.arange(n, device=self.device)
        self.evaluator.initialize_subgame(
            self.current_pbs.env,
            root_indices,
            self.current_pbs.beliefs,
        )
        self.current_pbs = self.evaluator.evaluate_cfr()

    def _warmup_current_pbs(self) -> None:
        target = self.target_batch_size
        quarter = target // 4
        if quarter == 0:
            return

        counts = (quarter, quarter, quarter, target - 3 * quarter)
        self.current_pbs = self._new_pbs(counts[0])
        for next_count in counts[1:3]:
            self._advance_current_pbs_once()
            if self.current_pbs is None:
                self.current_pbs = self._new_pbs(next_count)
            else:
                self.current_pbs = self._extend_pbs(
                    self.current_pbs,
                    self.current_pbs.env.N + next_count,
                )

        self._advance_current_pbs_once()
        if self.current_pbs is None:
            self.current_pbs = self._new_pbs(counts[3])
        else:
            self.current_pbs = self._extend_pbs(
                self.current_pbs,
                self.current_pbs.env.N + counts[3],
            )

    def _pbs_state_dict(self, pbs: PublicBeliefState | None) -> dict | None:
        if pbs is None:
            return None
        return {
            "beliefs": pbs.beliefs.detach().cpu(),
            "env": {
                name: getattr(pbs.env, name).detach().cpu()
                for name in _ENV_STATE_FIELDS
                if hasattr(pbs.env, name)
            },
        }

    def _pbs_from_state_dict(self, state: dict | None) -> PublicBeliefState | None:
        if state is None:
            return None
        beliefs = state["beliefs"].to(device=self.device)
        n = int(beliefs.shape[0])
        pbs = PublicBeliefState.from_proto(
            env_proto=self.env_proto,
            beliefs=beliefs,
            num_envs=n,
        )
        env_state = state["env"]
        for name in env_state:
            getattr(pbs.env, name).copy_(env_state[name].to(device=self.device))
        return pbs

    def state_dict(self) -> dict:
        return {
            "last_extra": int(self.last_extra),
            "target_batch_size": self.target_batch_size,
            "current_pbs": self._pbs_state_dict(self.current_pbs),
        }

    def load_state_dict(self, state: dict) -> None:
        self.last_extra = int(state.get("last_extra", 0))
        self.target_batch_size = int(
            state.get("target_batch_size", self.target_batch_size)
        )
        self.current_pbs = self._pbs_from_state_dict(state.get("current_pbs"))

    @profile
    def generate_data(
        self,
        value_sample_count: int,
        return_value_batch: bool = True,
        return_policy_batch: bool = True,
        max_return_policy_samples: int | None = None,
    ) -> tuple[RebelBatch | None, RebelBatch | None]:
        target_batch_size = self.target_batch_size
        collected = self.last_extra

        value_batches = []
        policy_batches = []
        returned_policy_samples = 0

        while collected < value_sample_count:
            refilled = False
            if self.current_pbs is None:
                self.current_pbs = self._sample_roots(target_batch_size)
                refilled = True
            elif self.current_pbs.env.N < target_batch_size:
                if self.root_sampler is not None:
                    self.current_pbs = self._sample_roots(target_batch_size)
                else:
                    self.current_pbs = self._extend_pbs(
                        self.current_pbs, target_batch_size
                    )
                refilled = True

            root_count = int(self.current_pbs.env.N)
            root_indices = torch.arange(root_count, device=self.device)
            self.evaluator.initialize_subgame(
                self.current_pbs.env,
                root_indices,
                self.current_pbs.beliefs[:root_count],
            )

            next_pbs = self.evaluator.evaluate_cfr(
                sample_continuation=self.sample_continuations
            )
            self.current_pbs = None if self.root_sampler is not None else next_pbs
            if self.record_batch_diag:
                self._record_batch_diag(refilled)

            need_policy_batch = self.store_replay or return_policy_batch
            need_pre_chance_value_batch = self.include_pre_chance_value_batches and (
                self.store_replay or return_value_batch
            )
            value_batch, augmented_value_batch, policy_batch = (
                self.evaluator.training_data(
                    include_pre_chance_value_batch=need_pre_chance_value_batch,
                    include_policy_batch=need_policy_batch,
                )
            )
            if policy_batch is not None and policy_batch.policy_targets is not None:
                valid_actor = (policy_batch.features.to_act >= 0) & (
                    policy_batch.features.to_act < self.evaluator.num_players
                )
                valid_policy = (
                    valid_actor
                    & policy_batch.legal_masks.any(dim=-1)
                    & (policy_batch.policy_targets.sum(dim=(1, 2)) > 0)
                )
                policy_batch = policy_batch[valid_policy]
            if self.store_replay:
                if self.policy_buffer is None or self.value_buffer is None:
                    raise RuntimeError("store_replay=True requires replay buffers")
                if policy_batch is None:
                    raise RuntimeError("store_replay=True requires policy batch")
                self.policy_buffer.add_batch(policy_batch)
                self.value_buffer.add_batch(value_batch)
                if self.include_pre_chance_value_batches:
                    if augmented_value_batch is None:
                        raise RuntimeError(
                            "include_pre_chance_value_batches=True requires "
                            "augmented value batch"
                        )
                    self.value_buffer.add_batch(augmented_value_batch)

            if return_policy_batch and policy_batch is not None:
                remaining = (
                    None
                    if max_return_policy_samples is None
                    else max_return_policy_samples - returned_policy_samples
                )
                if remaining is None or remaining > 0:
                    if remaining is not None and len(policy_batch) > remaining:
                        idx = torch.randperm(
                            len(policy_batch),
                            device=policy_batch.features.context.device,
                            generator=self.evaluator.generator,
                        )[:remaining]
                        policy_batch = policy_batch[idx]
                    policy_batches.append(policy_batch)
                    returned_policy_samples += len(policy_batch)
            if return_value_batch:
                value_batches.append(value_batch)
                if self.include_pre_chance_value_batches:
                    if augmented_value_batch is None:
                        raise RuntimeError(
                            "include_pre_chance_value_batches=True requires "
                            "augmented value batch"
                        )
                    value_batches.append(augmented_value_batch)

            collected += len(value_batch)

        self.last_extra = collected - value_sample_count

        fresh_value_batch = (
            RebelBatch.cat(value_batches)
            if return_value_batch and value_batches
            else None
        )
        fresh_policy_batch = (
            RebelBatch.cat(policy_batches)
            if return_policy_batch and policy_batches
            else None
        )

        return fresh_value_batch, fresh_policy_batch

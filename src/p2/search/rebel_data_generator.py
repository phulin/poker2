import torch

from p2.env.card_utils import NUM_HANDS
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_replay import RebelReplayBuffer
from p2.search.cfr_evaluator import CFREvaluator, PublicBeliefState
from p2.utils.profiling import profile


class RebelDataGenerator:
    def __init__(
        self,
        env_proto: HUNLTensorEnv,
        evaluator: CFREvaluator,
        value_buffer: RebelReplayBuffer,
        policy_buffer: RebelReplayBuffer,
    ):
        self.env_proto = env_proto
        self.evaluator = evaluator
        self.value_buffer = value_buffer
        self.policy_buffer = policy_buffer
        self.device = evaluator.device
        initial_pbs = self._new_pbs(evaluator.root_nodes)
        self.current_pbs = initial_pbs
        self.last_extra = 0

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
        opp = 1 - me
        me_allin = env.is_allin[ar, me]
        opp_allin = env.is_allin[ar, opp]
        committed = env.committed[:N].float().sum(dim=1)
        commit_frac = committed / (committed + 2.0 * eff_stack + 1.0)
        denom = float(N)
        self.evaluator.stats["gen_batch"] = {
            "refilled": float(refilled),
            "street_preflop": float((street == 0).float().mean().item()),
            "street_flop": float((street == 1).float().mean().item()),
            "street_turn": float((street == 2).float().mean().item()),
            "street_river": float((street == 3).float().mean().item()),
            "spr_p50": float(spr.median().item()),
            "spr_lt1_frac": float((spr < 1.0).float().mean().item()),
            "commit_frac_mean": float(commit_frac.mean().item()),
            "any_allin_frac": float((me_allin | opp_allin).float().sum().item() / denom),
            "facing_allin_frac": float((opp_allin & ~me_allin).float().sum().item() / denom),
        }

    def _new_pbs(self, target_batch_size: int) -> PublicBeliefState:
        beliefs = torch.full(
            (target_batch_size, self.evaluator.num_players, NUM_HANDS),
            1.0 / NUM_HANDS,
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

    @profile
    def generate_data(
        self,
        value_sample_count: int,
        return_value_batch: bool = True,
        return_policy_batch: bool = True,
    ) -> tuple[RebelBatch | None, RebelBatch | None]:
        N = self.evaluator.root_nodes
        root_indices = torch.arange(N, device=self.device)
        collected = self.last_extra

        value_batches = []
        policy_batches = []

        while collected < value_sample_count:
            refilled = False
            if self.current_pbs is None:
                self.current_pbs = self._new_pbs(N)
                refilled = True
            elif self.current_pbs.env.N < N:
                self.current_pbs = self._extend_pbs(self.current_pbs, N)
                refilled = True

            self.evaluator.initialize_subgame(
                self.current_pbs.env,
                root_indices,
                self.current_pbs.beliefs,
            )

            self.current_pbs = self.evaluator.evaluate_cfr()
            self._record_batch_diag(refilled)

            value_batch, augmented_value_batch, policy_batch = (
                self.evaluator.training_data()
            )
            self.policy_buffer.add_batch(policy_batch)
            self.value_buffer.add_batch(value_batch)
            self.value_buffer.add_batch(augmented_value_batch)

            if return_policy_batch:
                policy_batches.append(policy_batch)
            if return_value_batch:
                value_batches.append(value_batch)
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

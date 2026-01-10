import torch

from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.rl.rebel_batch import RebelBatch
from alphaholdem.rl.rebel_replay import RebelReplayBuffer
from alphaholdem.search.cfr_evaluator import CFREvaluator, PublicBeliefState
from alphaholdem.utils.profiling import profile


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
        self, value_sample_count: int
    ) -> tuple[RebelBatch | None, RebelBatch | None]:
        N = self.evaluator.root_nodes
        root_indices = torch.arange(N, device=self.device)
        collected = self.last_extra

        value_batches = []
        policy_batches = []

        while collected < value_sample_count:
            if self.current_pbs is None:
                self.current_pbs = self._new_pbs(N)
            elif self.current_pbs.env.N < N:
                self.current_pbs = self._extend_pbs(self.current_pbs, N)

            self.evaluator.initialize_subgame(
                self.current_pbs.env,
                root_indices,
                self.current_pbs.beliefs,
            )

            self.current_pbs = self.evaluator.evaluate_cfr()

            value_batch, augmented_value_batch, policy_batch = (
                self.evaluator.training_data()
            )
            self.policy_buffer.add_batch(policy_batch)
            self.value_buffer.add_batch(value_batch)
            self.value_buffer.add_batch(augmented_value_batch)

            policy_batches.append(policy_batch)
            value_batches.append(value_batch)
            value_batches.append(augmented_value_batch)

            collected += len(value_batch)

        self.last_extra = collected - value_sample_count

        fresh_value_batch = RebelBatch.cat(value_batches) if value_batches else None
        fresh_policy_batch = RebelBatch.cat(policy_batches) if policy_batches else None

        return fresh_value_batch, fresh_policy_batch

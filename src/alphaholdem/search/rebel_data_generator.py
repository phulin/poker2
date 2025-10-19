import torch
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS
from alphaholdem.rl.rebel_replay import RebelBatch, RebelReplayBuffer
from alphaholdem.search.rebel_cfr_evaluator import PublicBeliefState, RebelCFREvaluator
from alphaholdem.utils.profiling import profile


class RebelDataGenerator:
    def __init__(
        self,
        env_proto: HUNLTensorEnv,
        evaluator: RebelCFREvaluator,
        buffer: RebelReplayBuffer,
    ):
        self.env_proto = env_proto
        self.evaluator = evaluator
        self.buffer = buffer
        self.device = evaluator.device
        initial_pbs = self._new_pbs(evaluator.search_batch_size)
        self.current_pbs = initial_pbs

    def _new_pbs(self, target_batch_size: int) -> PublicBeliefState:
        pbs = PublicBeliefState.from_proto(
            env_proto=self.env_proto,
            beliefs=torch.full(
                (target_batch_size, self.evaluator.num_players, NUM_HANDS),
                1.0 / NUM_HANDS,
                device=self.device,
            ),
            num_envs=target_batch_size,
        )
        pbs.env.reset()
        return pbs

    @profile
    def generate_data(self) -> None:
        batch_size = self.evaluator.search_batch_size
        root_indices = torch.arange(batch_size, device=self.device)
        collected = 0

        while collected < batch_size:
            self.evaluator.initialize_search(
                self.current_pbs.env,
                root_indices,
                self.current_pbs.beliefs,
            )

            while collected < batch_size:
                next_pbs = self.evaluator.self_play_iteration()
                batch = self.evaluator.sample_data()
                self.buffer.add_batch(batch)
                collected += len(batch)

                if next_pbs is None or collected >= batch_size:
                    break

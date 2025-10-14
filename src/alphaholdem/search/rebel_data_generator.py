from dataclasses import dataclass
import torch
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS
from alphaholdem.rl.rebel_replay import RebelReplayBuffer
from alphaholdem.search.rebel_cfr_evaluator import PublicBeliefState, RebelCFREvaluator


@dataclass
class TrainingData:
    features: torch.Tensor
    legal_masks: torch.Tensor
    values: torch.Tensor
    policy: torch.Tensor


class RebelDataGenerator:
    def __init__(
        self,
        env_proto: HUNLTensorEnv,
        evaluator: RebelCFREvaluator,
        buffer: RebelReplayBuffer,
    ):
        self.env_proto = env_proto
        self.next_pbs_idx = 0
        self.evaluator = evaluator
        self.buffer = buffer
        self.device = evaluator.device
        self.pbs_queue = [
            PublicBeliefState.from_proto(
                env_proto=self.env_proto,
                beliefs=torch.full(
                    (
                        self.evaluator.search_batch_size,
                        self.evaluator.num_players,
                        NUM_HANDS,
                    ),
                    1.0 / NUM_HANDS,
                    device=self.device,
                ),
                num_envs=self.evaluator.search_batch_size,
            )
        ]

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

    def generate_data(self) -> TrainingData:
        batch_size = self.evaluator.search_batch_size
        accum_start = 0
        total_street = 0

        while accum_start < batch_size:
            next_pbs = self._new_pbs(batch_size)
            self.evaluator.initialize_search(
                next_pbs.env, torch.arange(batch_size), next_pbs.beliefs
            )
            while next_pbs is not None:
                next_pbs = self.evaluator.self_play_iteration()
                batch = self.evaluator.sample_data()
                self.buffer.add_batch(batch)
                accum_start += len(batch)
                print("self_play_iteration", accum_start)
                print(
                    "street",
                    self.evaluator.env.street[:batch_size].float().mean().item(),
                )
                total_street += self.evaluator.env.street[:batch_size].sum().item()

        print(
            f"=> collected {accum_start}/{batch_size} samples, average street {total_street / accum_start}"
        )

        # Snapshot tensors for supervised targets; detach to break autograd graph.
        root_indices = torch.arange(batch_size, device=self.device)
        features = self.evaluator.encode_current_states(root_indices)
        legal_masks = self.evaluator.env.legal_bins_mask()[:batch_size]
        values = self.evaluator.values[:batch_size].detach()
        policy = self.evaluator.policy_probs_avg[:batch_size].detach()

        return TrainingData(
            features=features,
            legal_masks=legal_masks,
            values=values,
            policy=policy,
        )

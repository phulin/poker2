from dataclasses import dataclass
import torch
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.search.rebel_cfr_evaluator import PublicBeliefState, RebelCFREvaluator

NUM_HANDS = 1326


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
    ):
        self.env_proto = env_proto
        self.next_pbs_idx = 0
        self.evaluator = evaluator
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

    def generate_data(self) -> TrainingData:
        batch_size = self.evaluator.search_batch_size
        total_size = self.evaluator.total_nodes

        pbs = PublicBeliefState.from_proto(
            env_proto=self.env_proto,
            beliefs=torch.zeros(batch_size, self.evaluator.num_players, NUM_HANDS),
            num_envs=batch_size,
        )
        accum_start = 0

        # greedily take samples from the queue
        while accum_start < batch_size and len(self.pbs_queue) > 0:
            needed = batch_size - accum_start
            next_pbs = self.pbs_queue[0]
            start = self.next_pbs_idx
            end = min(start + needed, next_pbs.env.N)
            accum_end = accum_start + end - start

            pbs.env.copy_state_from(
                next_pbs.env,
                torch.arange(start, end),
                torch.arange(accum_start, accum_end),
            )
            pbs.beliefs[accum_start:accum_end] = next_pbs.beliefs[start:end]

            consumed = end - start
            if consumed == 0:
                # Nothing copied; discard exhausted PBS to avoid infinite loop.
                self.next_pbs_idx = 0
                self.pbs_queue.pop(0)
                continue

            if end == next_pbs.env.N:
                self.next_pbs_idx = 0
                self.pbs_queue.pop(0)
            else:
                self.next_pbs_idx = end

            accum_start = accum_end

        # any remaining: initialize uniform beliefs
        if accum_start < batch_size:
            pbs.env.reset(torch.arange(accum_start, batch_size))
            pbs.beliefs[accum_start:batch_size] = 1.0 / NUM_HANDS

        self.evaluator.initialize_search(pbs.env, torch.arange(pbs.env.N), pbs.beliefs)

        next_pbs = self.evaluator.self_play_iteration()
        if next_pbs is not None:
            self.pbs_queue.append(next_pbs)

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

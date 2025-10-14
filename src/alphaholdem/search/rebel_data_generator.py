import torch
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS
from alphaholdem.rl.rebel_replay import RebelBatch, RebelReplayBuffer
from alphaholdem.search.rebel_cfr_evaluator import PublicBeliefState, RebelCFREvaluator


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
        initial_pbs = PublicBeliefState.from_proto(
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
        initial_pbs.env.reset()
        self.pbs_queue = [initial_pbs]

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

    def generate_data(self) -> RebelBatch:
        batch_size = self.evaluator.search_batch_size
        root_indices = torch.arange(batch_size, device=self.device)
        collected = 0

        while collected < batch_size:
            if self.next_pbs_idx >= len(self.pbs_queue):
                self.pbs_queue.append(self._new_pbs(batch_size))

            current_pbs = self.pbs_queue[self.next_pbs_idx]
            self.next_pbs_idx += 1
            self.evaluator.initialize_search(
                current_pbs.env,
                root_indices,
                current_pbs.beliefs,
            )

            while collected < batch_size:
                next_pbs = self.evaluator.self_play_iteration()
                batch = self.evaluator.sample_data()
                batch_len = len(batch)
                if batch_len > 0:
                    self.buffer.add_batch(batch)
                    collected += batch_len
                if next_pbs is not None:
                    self.pbs_queue.append(next_pbs)
                if next_pbs is None or collected >= batch_size:
                    break

        if self.next_pbs_idx > 0:
            self.pbs_queue = self.pbs_queue[self.next_pbs_idx :]
            self.next_pbs_idx = 0

        # Snapshot tensors for supervised targets; detach to break autograd graph.
        features = self.evaluator.encode_current_states(root_indices).detach()
        legal_masks = self.evaluator.env.legal_bins_mask()[:batch_size].detach()
        values = self.evaluator.values[:batch_size].detach()
        policy = self.evaluator.policy_probs_avg[:batch_size].detach()
        if hasattr(self.evaluator.env, "to_act"):
            acting_players = self.evaluator.env.to_act[root_indices].detach()
        else:
            acting_players = torch.zeros(
                batch_size, dtype=torch.long, device=self.device
            )

        return RebelBatch(
            features=features,
            policy_targets=policy,
            value_targets=values,
            legal_masks=legal_masks,
            acting_players=acting_players,
        )

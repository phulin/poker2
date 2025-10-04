"""Fast diagnostics for transformer card awareness and routing.

Run:
  python debugging/transformer_diagnostics.py

Diagnostics:
  1) Card-shuffle invariance (Δlogits ~ 0 => card-blind policy)
  2) Attention to hole cards (mean attention from CLS to hole cards)
  3) Embedding grads on hole-card components (grad norms ~ 0 => broken routing)
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ExploiterConfig,
    ModelConfig,
    TrainingConfig,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.token_sequence_builder import TokenSequenceBuilder
from alphaholdem.models.transformer.tokens import (
    HOLE0_INDEX,
    HOLE1_INDEX,
    Special,
    get_action_token_id_offset,
    get_card_token_id_offset,
)
from alphaholdem.rl.self_play import SelfPlayTrainer


@torch.no_grad()
def build_preflop_batch(
    device: torch.device,
    batch_size: int,
    seq_len: int,
    env_cfg: EnvConfig,
) -> Tuple[TokenSequenceBuilder, torch.Tensor, int, int]:
    # Build env using provided env config
    env = HUNLTensorEnv(
        num_envs=batch_size,
        starting_stack=env_cfg.stack,
        sb=env_cfg.sb,
        bb=env_cfg.bb,
        bet_bins=env_cfg.bet_bins or [0.5, 1.0, 1.5, 2.0],
        device=device,
    )
    env.reset()

    encoder = TokenSequenceBuilder(
        tensor_env=env,
        sequence_length=seq_len,
        num_bet_bins=8,
        device=device,
        float_dtype=torch.float32,
    )
    idxs = torch.arange(env.N, device=device)

    # Minimal preflop decision sequence: CLS -> STREET_PREFLOP -> HOLE cards -> CONTEXT
    encoder.add_cls(idxs)
    encoder.add_street(idxs, torch.zeros_like(idxs))
    encoder.add_card(idxs, torch.zeros_like(idxs))
    encoder.add_card(idxs, torch.ones_like(idxs))
    encoder.add_context(idxs)

    data = encoder.encode_tensor_states(player=0, idxs=idxs)
    card_offset = get_card_token_id_offset()
    action_offset = get_action_token_id_offset()
    # Ensure HOLE0/HOLE1 positions contain concrete card tokens (rank/suit)
    p = 0
    hole0 = encoder.tensor_env.hole_indices[idxs, p, 0]
    hole1 = encoder.tensor_env.hole_indices[idxs, p, 1]
    data.token_ids[:, HOLE0_INDEX] = card_offset + hole0
    data.token_ids[:, HOLE1_INDEX] = card_offset + hole1
    data.card_ranks[:, HOLE0_INDEX] = hole0 % 13
    data.card_suits[:, HOLE0_INDEX] = hole0 // 13
    data.card_ranks[:, HOLE1_INDEX] = hole1 % 13
    data.card_suits[:, HOLE1_INDEX] = hole1 // 13
    return encoder, data, card_offset, action_offset


def assert_hole_positions_are_cards(
    data: StructuredEmbeddingData, card_offset: int
) -> None:
    action_offset = get_action_token_id_offset()
    tokens0 = data.token_ids[:, HOLE0_INDEX]
    tokens1 = data.token_ids[:, HOLE1_INDEX]
    assert (
        ((tokens0 >= card_offset) & (tokens0 < action_offset)).all().item()
    ), "HOLE0_INDEX tokens must be card tokens"
    assert (
        ((tokens1 >= card_offset) & (tokens1 < action_offset)).all().item()
    ), "HOLE1_INDEX tokens must be card tokens"


@torch.no_grad()
def card_shuffle_invariance(
    model: PokerTransformerV1, data: StructuredEmbeddingData, card_offset: int
) -> float:
    """Shuffle only hole-card token values across the batch and return mean L2 ‖Δlogits‖."""
    model.eval()
    logits_orig = model(data).policy_logits.float()

    assert_hole_positions_are_cards(data, card_offset)
    # Create shuffled copy
    shuffled = data.clone()
    B = shuffled.token_ids.shape[0]
    arange = torch.arange(B, device=shuffled.device)
    perm = torch.randperm(B, device=shuffled.device)
    shuffled.token_ids[arange, HOLE0_INDEX] = data.token_ids[perm, HOLE0_INDEX]
    shuffled.token_ids[arange, HOLE1_INDEX] = data.token_ids[perm, HOLE1_INDEX]
    shuffled.card_ranks[arange, HOLE0_INDEX] = data.card_ranks[perm, HOLE0_INDEX]
    shuffled.card_ranks[arange, HOLE1_INDEX] = data.card_ranks[perm, HOLE1_INDEX]
    shuffled.card_suits[arange, HOLE0_INDEX] = data.card_suits[perm, HOLE0_INDEX]
    shuffled.card_suits[arange, HOLE1_INDEX] = data.card_suits[perm, HOLE1_INDEX]

    logits_shuf = model(shuffled).policy_logits.float()
    delta = (logits_shuf - logits_orig).float()
    return float(delta.norm(dim=1).mean().item())


@torch.no_grad()
def card_randomize_invariance(
    model: PokerTransformerV1, data: StructuredEmbeddingData, card_offset: int
) -> float:
    """Randomize hole-card tokens entirely and return mean L2 ‖Δlogits‖."""
    model.eval()
    logits_orig = model(data).policy_logits.float()

    assert_hole_positions_are_cards(data, card_offset)
    # Create randomized copy
    randomized = data.clone()

    for b in range(randomized.token_ids.shape[0]):
        # Generate random card tokens within valid range (changes rank/suit)
        random_cards = card_offset + torch.randperm(52, device=randomized.device)[:2]
        randomized.token_ids[b, HOLE0_INDEX] = random_cards[0]
        randomized.token_ids[b, HOLE1_INDEX] = random_cards[1]
        randomized.card_ranks[b, HOLE0_INDEX] = (random_cards[0] - card_offset) % 13
        randomized.card_ranks[b, HOLE1_INDEX] = (random_cards[1] - card_offset) % 13
        randomized.card_suits[b, HOLE0_INDEX] = (random_cards[0] - card_offset) // 13
        randomized.card_suits[b, HOLE1_INDEX] = (random_cards[1] - card_offset) // 13

    logits_rand = model(randomized).policy_logits.float()
    delta = (logits_rand - logits_orig).float()
    return float(delta.norm(dim=1).mean().item())


class AttnCapture(nn.Module):
    """Wrapper to capture average attention probs from CLS to given positions per layer.

    This re-computes attention weights from q, k inside SDPA by a separate matmul for logging only.
    """

    def __init__(self, attn_module: nn.Module):
        super().__init__()
        self.attn = attn_module
        self.last_mean_cls_to_holes = None

    def forward(self, x, attention_mask, cos, sin, kv_cache=None):  # type: ignore[override]
        # Run original
        out, new_cache = self.attn(x, attention_mask, cos, sin, kv_cache)
        # Cannot access internal q/k here without editing module; fallback: skip if unavailable
        # We approximate attention observation by using output context correlation at positions of interest in the caller.
        return out, new_cache


def mean_attention_to_holes(
    model: PokerTransformerV1,
    data: StructuredEmbeddingData,
    hole_positions: List[Tuple[int, int]],
) -> float:
    """Proxy metric: use final-layer token representations to approximate focus on holes.

    As we can't directly access SDPA attn weights without editing the module,
    we measure cosine similarity between CLS representation and hole token representations.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        # Recompute last hidden states by tapping pre-head tensors if available; fallback to embedding via heads inputs
        # Here, use logits gradient w.r.t. token states is not available; so use fused embeddings + FFN output captured via model internals
        # As a proxy, re-embed and project through input_ffn only
        hidden = model.input_ffn(model.embedding(data))  # [B,S,d]
        B, S, D = hidden.shape
        # CLS assumed at position where token == Special.CLS.value
        cls_pos = []
        for b in range(B):
            pos = int(torch.where(data.token_ids[b] == Special.CLS.value)[0][0].item())
            cls_pos.append(pos)
        sims = []
        for b in range(B):
            p0, p1 = hole_positions[b]
            if p0 < 0 or p1 < 0:
                continue
            v_cls = hidden[b, cls_pos[b]]
            v_h0 = hidden[b, p0]
            v_h1 = hidden[b, p1]

            def cos_sim(a, c):
                return torch.nn.functional.cosine_similarity(
                    a.unsqueeze(0), c.unsqueeze(0)
                ).item()

            sims.append(0.5 * (cos_sim(v_cls, v_h0) + cos_sim(v_cls, v_h1)))
        return float(sum(sims) / max(1, len(sims)))


def embedding_grad_norms(
    model: PokerTransformerV1, data: StructuredEmbeddingData
) -> Tuple[float, float]:
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    out = model(data)
    # Sum of logits to have non-zero grads broadly
    loss = out.policy_logits.float().mean()
    loss.backward()
    rank_norm = (
        float(model.embedding.card_rank_emb.weight.grad.norm().item())
        if model.embedding.card_rank_emb.weight.grad is not None
        else 0.0
    )
    suit_norm = (
        float(model.embedding.card_suit_emb.weight.grad.norm().item())
        if model.embedding.card_suit_emb.weight.grad is not None
        else 0.0
    )
    return rank_norm, suit_norm


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer do-now diagnostics")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint (.pt) to load before running diagnostics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (e.g., cpu, cuda, mps)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config_transformer_hp.yaml",
        help="Path to YAML config (defaults to conf/config_transformer_hp.yaml)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    # Load config YAML similar to train_kbest
    cfg_yaml_path = os.path.expanduser(args.config)
    if not os.path.isfile(cfg_yaml_path):
        raise FileNotFoundError(f"Config not found: {cfg_yaml_path}")
    cfg_dict = OmegaConf.to_container(OmegaConf.load(cfg_yaml_path), resolve=True)

    # Construct structured Config
    train_cfg = TrainingConfig(**cfg_dict.get("train", {}))
    model_cfg = ModelConfig(**cfg_dict.get("model", {}))
    env_cfg = EnvConfig(**cfg_dict.get("env", {}))
    exploiter_cfg = ExploiterConfig(**cfg_dict.get("exploiter", {}))

    cfg = Config(
        train=train_cfg,
        model=model_cfg,
        env=env_cfg,
        exploiter=exploiter_cfg,
        **{
            k: v
            for k, v in cfg_dict.items()
            if k not in ["train", "model", "env", "state_encoder", "exploiter"]
        },
    )

    # Override device via CLI and set seed
    if args.device is not None:
        cfg.device = str(device)
    torch.manual_seed(cfg.seed)

    cfg.num_envs = 64
    cfg.use_wandb = False
    # Build trainer and optionally load checkpoint via trainer API
    trainer = SelfPlayTrainer(cfg=cfg, device=device)
    model = trainer.model
    if args.checkpoint is not None:
        ckpt_path = os.path.expanduser(args.checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        step, _ = trainer.load_checkpoint(ckpt_path)
        print(f"Loaded checkpoint via trainer at step {step}")

    _, data, card_offset, _ = build_preflop_batch(
        device,
        batch_size=cfg.num_envs,
        seq_len=cfg.train.max_sequence_length,
        env_cfg=cfg.env,
    )

    # 1) Card-shuffle invariance
    delta_logits = card_shuffle_invariance(model, data.clone(), card_offset)
    print(f"[Card-shuffle invariance] mean ‖Δlogits‖ = {delta_logits:.6f}")

    # 2) Card-randomize invariance
    delta_logits_rand = card_randomize_invariance(model, data.clone(), card_offset)
    print(f"[Card-randomize invariance] mean ‖Δlogits‖ = {delta_logits_rand:.6f}")

    # 3) Attention (proxy) from CLS to hole cards
    assert_hole_positions_are_cards(data, card_offset)
    holes = [(HOLE0_INDEX, HOLE1_INDEX) for _ in range(data.token_ids.shape[0])]
    mean_sim = mean_attention_to_holes(model, data, holes)
    print(f"[Attention to hole cards] mean cosine(CLS, holes) = {mean_sim:.6f}")

    # 4) Embedding grads
    rank_norm, suit_norm = embedding_grad_norms(model, data)
    print(
        f"[Embedding grads] card_rank_emb ‖grad‖ = {rank_norm:.6e} | card_suit_emb ‖grad‖ = {suit_norm:.6e}"
    )


if __name__ == "__main__":
    main()

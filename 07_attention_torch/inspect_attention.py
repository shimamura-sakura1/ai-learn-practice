"""Attention 语义小实验；这里的 reference 与待实现函数完全独立。"""

from __future__ import annotations

import math

import torch


torch.set_printoptions(precision=4, sci_mode=False)


def _reference_trace(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False):
    k_t = k.transpose(-2, -1)
    raw_scores = q @ k_t
    scaled_scores = raw_scores / math.sqrt(q.shape[-1])
    masked_scores = scaled_scores
    if causal:
        sq, sk = q.shape[-2], k.shape[-2]
        future = torch.ones(sq, sk, dtype=torch.bool, device=q.device).triu(1)
        masked_scores = scaled_scores.masked_fill(future, float("-inf"))
    probs = torch.softmax(masked_scores, dim=-1)
    out = probs @ v
    return k_t, raw_scores, scaled_scores, masked_scores, probs, out


def _show(name: str, tensor: torch.Tensor, shape_note: str):
    print(f"{name}  # shape: {shape_note} -> {tuple(tensor.shape)}")
    print(tensor)


def experiment_1_hand_calculation():
    print("\n=== 实验 1：手算友好案例 ===")
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]])
    v = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    k_t, raw, scaled, _, probs, out = _reference_trace(q, k, v)
    _show("Q", q, "[B, S, Dh]")
    _show("K", k, "[B, S, Dh]")
    _show("K^T", k_t, "[B, Dh, S]")
    _show("raw scores", raw, "[B, S, S]")
    _show("scaled scores", scaled, "[B, S, S]")
    _show("attention probs", probs, "[B, S, S]")
    _show("output", out, "[B, S, Dv]")


def experiment_2_orthogonal_tokens():
    print("\n=== 实验 2：正交 token ===")
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    v = torch.eye(3).unsqueeze(0)
    *_, probs, _ = _reference_trace(q, k, v)
    _show("Q", q, "[1, 3, 2]")
    _show("K", k, "[1, 3, 2]")
    _show("attention probs", probs, "[1, 3, 3]")
    print("每个 query 最偏好的 key index:", probs.argmax(dim=-1).squeeze(0).tolist())


def experiment_3_causal_mask():
    print("\n=== 实验 3：causal mask ===")
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    k = q.clone()
    v = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    _, _, scaled, masked, probs, _ = _reference_trace(q, k, v, causal=True)
    _show("scores before mask", scaled, "[1, 3, 3]")
    _show("scores after mask", masked, "[1, 3, 3]")
    _show("probs after mask", probs, "[1, 3, 3]")


def _entropy(probs: torch.Tensor):
    return -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1).mean()


def experiment_4_scaling():
    print("\n=== 实验 4：scale 的作用 ===")
    generator = torch.Generator().manual_seed(123)
    print(f"{'Dh':>6} {'scaled?':>9} {'mean max probability':>22} {'mean entropy':>14}")
    for dh in (8, 32, 128, 512):
        q = torch.randn(4, 32, dh, generator=generator)
        k = torch.randn(4, 32, dh, generator=generator)
        raw = q @ k.transpose(-2, -1)
        for label, scores in (("no", raw), ("yes", raw / math.sqrt(dh))):
            probs = torch.softmax(scores, dim=-1)
            print(f"{dh:6d} {label:>9} {probs.max(dim=-1).values.mean().item():22.6f} {_entropy(probs).item():14.6f}")


def main():
    experiment_1_hand_calculation()
    experiment_2_orthogonal_tokens()
    experiment_3_causal_mask()
    experiment_4_scaling()


if __name__ == "__main__":
    main()


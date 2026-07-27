from __future__ import annotations

import torch

from model import DecoderOnlyTransformer, TransformerConfig


def _run_once(device: torch.device, dtype: torch.dtype) -> None:
    config = TransformerConfig(
        vocab_size=128,
        batch_size=2,
        sequence_length=16,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        dtype=dtype,
    )
    model = DecoderOnlyTransformer(config).to(device=device, dtype=dtype)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.sequence_length),
        device=device,
    )  # [B, S]
    targets = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.sequence_length),
        device=device,
    )  # [B, S]

    logits = model(input_ids)  # [B, S, V]
    expected_shape = (config.batch_size, config.sequence_length, config.vocab_size)
    assert logits.shape == expected_shape, f"expected {expected_shape}, got {tuple(logits.shape)}"

    loss = model.loss(input_ids, targets)
    assert torch.isfinite(loss).item(), f"loss is not finite: {loss.item()}"

    loss.backward()
    params_to_check = {
        "token_embedding.weight": model.token_embedding.weight,
        "blocks.0.attention.q_proj.weight": model.blocks[0].attention.q_proj.weight,
        "blocks.0.mlp.down_proj.weight": model.blocks[0].mlp.down_proj.weight,
        "final_norm.weight": model.final_norm.weight,
        "lm_head.weight": model.lm_head.weight,
    }
    for name, param in params_to_check.items():
        assert param.grad is not None, f"{name} grad is None"
        assert torch.isfinite(param.grad).all().item(), f"{name} grad has non-finite values"


def main() -> None:
    torch.manual_seed(0)
    _run_once(torch.device("cpu"), torch.float32)
    print("cpu fp32 smoke test passed")

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        _run_once(torch.device("cuda"), dtype)
        torch.cuda.synchronize()
        print(f"cuda {dtype} smoke test passed")
    else:
        print("cuda not available; skipped cuda smoke test")


if __name__ == "__main__":
    main()

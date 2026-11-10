"""SDPA at the shape of the Wan VAE's mid-block attention on gfx1100, against an fp32 reference.

The block is single-head with 384 channels and hands SDPA chunk() views of one qkv
tensor. Tiled encoding (inference) gives it 30x52 = 1560 tokens per frame; untiled
encoding (DiffSynth's training data_process uses tiled=False) gives 60x104 = 6240.
Each fused backend runs twice to show whether it is even deterministic. head_dim 128 at
the same lengths is the control.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dev, dt = "cuda", torch.bfloat16
BACKENDS = {"default": None, "efficient": [SDPBackend.EFFICIENT_ATTENTION], "flash": [SDPBackend.FLASH_ATTENTION],
            "math_bf16": [SDPBackend.MATH]}


def run(q, k, v, backend):
    if backend is None:
        return F.scaled_dot_product_attention(q, k, v)
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v)


for dim, batch, tokens in ((384, 4, 1560), (384, 1, 3120), (384, 1, 4096), (384, 1, 5120), (384, 1, 6240), (384, 4, 6240),
                           (128, 1, 6240), (128, 4, 6240)):
    torch.manual_seed(0)
    qkv = torch.randn(batch, 1, tokens, 3 * dim, device=dev, dtype=dt)
    q, k, v = qkv.chunk(3, dim=-1)
    with sdpa_kernel([SDPBackend.MATH]):
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    scale = ref.abs().max().item()
    for name, b in BACKENDS.items():
        try:
            o1, o2 = run(q, k, v, b).float(), run(q, k, v, b).float()
            err = (o1 - ref).abs().nan_to_num(float("inf")).max().item()
            same = torch.equal(o1.nan_to_num(123.0), o2.nan_to_num(123.0))
            print(f"d={dim} b={batch} n={tokens:5d} {name:9s} nan={torch.isnan(o1).sum().item():>8d} "
                  f"max_err={err:10.3e} (|ref|max {scale:.2f}) repeatable={same}")
        except Exception as e:
            print(f"d={dim} b={batch} n={tokens:5d} {name:9s} unavailable: {type(e).__name__}: {str(e)[:80]}")
        torch.cuda.empty_cache()
print("VAE_ATTN_PROBE_OK")

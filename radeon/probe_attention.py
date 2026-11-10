"""SDPA backends at Wan2.2-A14B scale on one Radeon, plus one FFN-sized GEMM.

832x480x81 -> latent 21x60x104 -> 21*30*52 = 32760 video tokens, plus 30*52 = 1560
reference tokens from ref_conv: 34320 tokens, 40 heads, head_dim 128, bf16.
The math backend would materialise 40*34320^2 bf16 scores (~94 GB), so only the
fused kernels are timed at full size.
"""

import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dev = "cuda"
print("device", torch.cuda.get_device_name(0), "torch", torch.__version__, "hip", torch.version.hip)
print("arch", torch.cuda.get_device_properties(0).gcnArchName)
print("flash_sdp_enabled", torch.backends.cuda.flash_sdp_enabled(), "mem_efficient_sdp_enabled", torch.backends.cuda.mem_efficient_sdp_enabled())


def timed(fn, iters=3):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


# Numerics on a small problem against an fp32 math reference.
torch.manual_seed(0)
q, k, v = (torch.randn(1, 40, 2048, 128, device=dev, dtype=torch.bfloat16) for _ in range(3))
with sdpa_kernel([SDPBackend.MATH]):
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
for name, backend in [("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)]:
    try:
        with sdpa_kernel([backend]):
            out = F.scaled_dot_product_attention(q, k, v)
        print(f"small {name}: max_abs_err {(out.float() - ref).abs().max().item():.4e}")
    except Exception as e:
        print(f"small {name}: FAILED {type(e).__name__}: {str(e)[:200]}")

# Full Wan2.2-A14B self-attention size.
N, H, D = 34320, 40, 128
q, k, v = (torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
flops = 4 * N * N * D * H
for name, backends in [("flash", [SDPBackend.FLASH_ATTENTION]), ("efficient", [SDPBackend.EFFICIENT_ATTENTION]),
                       ("default_without_math", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])]:
    try:
        torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel(backends):
            s = timed(lambda: F.scaled_dot_product_attention(q, k, v))
        print(f"full {name}: {s*1000:.0f} ms/call, {flops/s/1e12:.1f} TFLOPS, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    except Exception as e:
        print(f"full {name}: FAILED {type(e).__name__}: {str(e)[:200]}")

try:
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
    print("can_use_flash_attention", torch.backends.cuda.can_use_flash_attention(params, True))
    print("can_use_efficient_attention", torch.backends.cuda.can_use_efficient_attention(params, True))
except Exception as e:
    print("SDPAParams probe skipped:", type(e).__name__, str(e)[:160])
del q, k, v

# One FFN up-projection at the same token count.
x = torch.randn(N, 5120, device=dev, dtype=torch.bfloat16)
w = torch.randn(13824, 5120, device=dev, dtype=torch.bfloat16)
s = timed(lambda: F.linear(x, w))
print(f"gemm {N}x5120x13824: {s*1000:.0f} ms, {2*N*5120*13824/s/1e12:.1f} TFLOPS")
print("PROBE_OK")

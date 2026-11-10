"""Does the q/k/v memory layout change SDPA speed on gfx1100 at Wan2.2-A14B size?

DiffSynth's Wan DiT projects q, k, v as (batch, seq, heads*dim) and hands SDPA the view
rearrange(x, "b s (n d) -> b n s d"), whose sequence stride is heads*dim. probe_attention.py
times freshly allocated (batch, heads, seq, dim) tensors instead. This times both layouts
for self-attention (34320 tokens) and for the text cross-attention (512 keys), plus the
strided view made contiguous first, copy included.
"""

import time

import torch
import torch.nn.functional as F

dev, dt = "cuda", torch.bfloat16
N, H, D, L = 34320, 40, 128, 512


def timed(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


def heads(x):
    return x.view(x.shape[0], x.shape[1], H, D).transpose(1, 2)


torch.manual_seed(0)
xq, xk, xv = (torch.randn(1, N, H * D, device=dev, dtype=dt) for _ in range(3))
ck, cv = (torch.randn(1, L, H * D, device=dev, dtype=dt) for _ in range(2))
q, k, v = heads(xq), heads(xk), heads(xv)
qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
print("strided q stride", tuple(q.stride()), "contiguous q stride", tuple(qc.stride()))

ref = F.scaled_dot_product_attention(qc, kc, vc)
out = F.scaled_dot_product_attention(q, k, v)
print(f"self strided vs contiguous: max_abs_diff {(out.float() - ref.float()).abs().max().item():.3e}")

self_flops = 4 * N * N * D * H
cross_flops = 4 * N * L * D * H
cases = {
    "self  contiguous (B,H,S,D)": (lambda: F.scaled_dot_product_attention(qc, kc, vc), self_flops),
    "self  strided view (DiffSynth)": (lambda: F.scaled_dot_product_attention(q, k, v), self_flops),
    "self  view + .contiguous()": (lambda: F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous()), self_flops),
    "cross strided view (DiffSynth)": (lambda: F.scaled_dot_product_attention(q, heads(ck), heads(cv)), cross_flops),
    "cross view + .contiguous()": (lambda: F.scaled_dot_product_attention(q.contiguous(), heads(ck).contiguous(), heads(cv).contiguous()), cross_flops),
}
for name, (fn, flops) in cases.items():
    s = timed(fn)
    print(f"{name:32s} {s * 1000:8.1f} ms  {flops / s / 1e12:6.1f} TFLOPS")
print("LAYOUT_OK")

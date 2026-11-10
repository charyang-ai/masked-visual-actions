"""Where a denoising step goes: infer.py's own pipeline, two steps, every DiT forward timed.

  python radeon/profile_step.py /workspace/data/samples/towel /workspace/results/profile

Two steps cover both experts (t=1000 on the high-noise one, t=833 on the low-noise one),
each with a conditional and an unconditional forward. The first forward of each expert
also moves its layers onto the GPU, so the kernel profile is taken on the second
(unconditional, high-noise) forward, when everything is resident.
"""

import json
import sys
import time
from collections import defaultdict

import torch
from PIL import Image
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/workspace/src/inference")
from infer import DEFAULT_NEGATIVE_PROMPT, build_pipeline, load_frames  # noqa: E402

PROFILE_AT = 1
CATEGORIES = [
    ("attention", ("attn_fwd", "aotriton", "flash", "fmha", "attention")),
    ("gemm", ("cijk", "gemm", "hipblaslt", "rocblas", "matmul", "_mm_")),
    ("norm", ("layer_norm", "layernorm", "rms", "norm_kernel")),
    ("copy/cast", ("copy", "memcpy", "cast", "convert")),
    ("elementwise", ("elementwise", "vectorized", "unrolled", "reduce", "softmax", "pointwise", "triton")),
]


def category(name):
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


def main():
    sample, out = sys.argv[1], sys.argv[2]
    pipe = build_pipeline("PAI/Wan2.2-Fun-A14B-Control", low_vram=True)
    pipe.load_lora(pipe.dit, "/workspace/checkpoints/masked_world_lora_high.safetensors", alpha=1.0)
    pipe.load_lora(pipe.dit2, "/workspace/checkpoints/masked_world_lora_low.safetensors", alpha=1.0)
    control = load_frames(f"{sample}/control.mp4", 480, 832, 81)
    reference = Image.open(f"{sample}/reference.png").convert("RGB").resize((832, 480))
    prompt = open(f"{sample}/prompt.txt", encoding="utf-8").read()

    original = pipe.model_fn
    state = {"n": 0, "forwards": []}

    def timed_model_fn(**kwargs):
        torch.cuda.synchronize()
        t = time.time()
        if state["n"] == PROFILE_AT:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                result = original(**kwargs)
                torch.cuda.synchronize()
            state["prof"] = prof
        else:
            result = original(**kwargs)
        torch.cuda.synchronize()
        state["forwards"].append({"index": state["n"], "timestep": float(kwargs["timestep"].item()),
                                  "seconds": round(time.time() - t, 2),
                                  "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 2)})
        print("forward", state["forwards"][-1], flush=True)
        state["n"] += 1
        return result

    pipe.model_fn = timed_model_fn
    pipe(prompt=prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT, control_video=control, reference_image=reference,
         height=480, width=832, num_inference_steps=2, seed=123, tiled=True)

    # A CPU op and the kernel it launches both carry the kernel's device time, so the
    # breakdown sums GPU-side (kernel) events only; ops are listed separately.
    kernels, ops = [], []
    for e in state["prof"].key_averages():
        dev = getattr(e, "self_device_time_total", None)
        if dev is None:
            dev = getattr(e, "self_cuda_time_total", 0)
        if dev <= 0:
            continue
        row = (e.key, dev / 1e6, e.count)
        (kernels if e.device_type != DeviceType.CPU else ops).append(row)
    kernels.sort(key=lambda r: -r[1])
    ops.sort(key=lambda r: -r[1])
    totals = defaultdict(float)
    for name, sec, _ in kernels:
        totals[category(name)] += sec
    kernel_total = sum(totals.values())
    report = {
        "forwards": state["forwards"],
        "profiled_forward_kernel_s": round(kernel_total, 2),
        "kernels_by_category_s": {k: round(v, 2) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])},
        "top_kernels": [{"name": n[:160], "s": round(s, 3), "calls": c, "category": category(n)} for n, s, c in kernels[:25]],
        "top_ops": [{"name": n[:120], "s": round(s, 3), "calls": c} for n, s, c in ops[:15]],
    }
    with open(f"{out}.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("profiled_forward_kernel_s", "kernels_by_category_s")}, indent=1))
    for r in report["top_kernels"][:12]:
        print(f"{r['s']:8.3f}s {r['calls']:5d}  {r['category']:11s} {r['name'][:110]}")


if __name__ == "__main__":
    main()

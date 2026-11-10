"""How much the Wan VAE's attention kernel matters on gfx1100, tiled (infer.py) and untiled (training).

  python radeon/probe_vae_encode.py /workspace/data/samples/towel/control.mp4

The VAE's mid-block attention is single-head with 384 channels. This loads only the VAE
the way infer.py does and runs its attention either through the default (fused) SDPA or
through the exact math kernel, then compares latents and decoded pixels, and repeats the
untiled encode to see whether NaNs come and go.
"""

import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import diffsynth.models.wan_video_vae as wvae
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData

_sdpa = F.scaled_dot_product_attention
MODE = {"kernel": "fused"}
stats = {"calls": 0, "max_ref": 0.0, "max_err": 0.0, "nan": 0}


def vae_sdpa(q, k, v, *args, **kwargs):
    if MODE["kernel"] == "math":
        with sdpa_kernel([SDPBackend.MATH]):
            return _sdpa(q, k, v, *args, **kwargs)
    out = _sdpa(q, k, v, *args, **kwargs)
    if MODE.get("check"):
        with sdpa_kernel([SDPBackend.MATH]):
            ref = _sdpa(q.float(), k.float(), v.float(), *args, **kwargs)
        stats["calls"] += 1
        stats["max_ref"] = max(stats["max_ref"], ref.abs().max().item())
        stats["max_err"] = max(stats["max_err"], (out.float() - ref).abs().nan_to_num(float("inf")).max().item())
        stats["nan"] += torch.isnan(out).sum().item()
    return out


wvae.F.scaled_dot_product_attention = vae_sdpa


def run(kernel, fn, check=False):
    MODE.update(kernel=kernel, check=check)
    stats.update(calls=0, max_ref=0.0, max_err=0.0, nan=0)
    with torch.no_grad():
        return fn()


def psnr(a, b):
    mse = ((a.float().cpu() - b.float().cpu()) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()  # inputs in [-1, 1]


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="Wan2.1_VAE.pth")],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)
frames = VideoData(sys.argv[1], height=480, width=832)
video = pipe.preprocess_video([frames[i] for i in range(len(frames))])
tiled = dict(device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))

lat_fused = run("fused", lambda: pipe.vae.encode(video, **tiled), check=True)
print(f"tiled encode, fused: attention calls={stats['calls']} |ref|max={stats['max_ref']:.1f} "
      f"max_err={stats['max_err']:.3e} nan={stats['nan']} latents_nan={torch.isnan(lat_fused).sum().item()}")
lat_math = run("math", lambda: pipe.vae.encode(video, **tiled))
d = (lat_fused.float() - lat_math.float()).abs()
print(f"tiled encode, fused vs math latents: max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e} "
      f"latent |max|={lat_math.abs().max().item():.2f}")

dec_fused = run("fused", lambda: pipe.vae.decode(lat_math, **tiled), check=True)
print(f"tiled decode, fused: attention calls={stats['calls']} |ref|max={stats['max_ref']:.1f} "
      f"max_err={stats['max_err']:.3e} nan={stats['nan']} pixels_nan={torch.isnan(dec_fused).sum().item()}")
dec_math = run("math", lambda: pipe.vae.decode(lat_math, **tiled))
print(f"tiled decode, fused vs math pixels: PSNR={psnr(dec_fused, dec_math):.2f} dB "
      f"max_abs={(dec_fused.float().cpu() - dec_math.float().cpu()).abs().max().item():.3e}")
print(f"tiled encode+decode round trip vs input: fused {psnr(run('fused', lambda: pipe.vae.decode(lat_fused, **tiled)), video):.2f} dB, "
      f"math {psnr(dec_math, video):.2f} dB")

for i in range(3):
    lat = run("fused", lambda: pipe.vae.encode(video, device=pipe.device, tiled=False), check=True)
    print(f"untiled encode #{i}, fused: attention max_err={stats['max_err']:.3e} nan={stats['nan']} latents_nan={torch.isnan(lat).sum().item()}")
lat = run("math", lambda: pipe.vae.encode(video, device=pipe.device, tiled=False))
print(f"untiled encode, math: latents_nan={torch.isnan(lat).sum().item()} vs tiled math max_abs={(lat.float().cpu() - lat_math.float().cpu()).abs().max().item():.3e}")
print("VAE_ENCODE_PROBE_OK")

# Results

All runs on one AMD Radeon Pro W7900D (gfx1100) inside one container that sees only that card.
Each run directory holds the generated clip (`<sample>.mp4`), the inputs as `infer.py --save-inputs`
wrote them, `infer.log`, `time.json` (wall time, peak host RSS), `vram.log` (device VRAM every 10 s),
`metrics.json` / `metrics.log` (`radeon/metrics.py`) and `compare_<sample>.jpg` where present.

| Directory | DiffSynth | Sample | Steps |
| --- | --- | --- | ---: |
| `smoke_towel_2step/` | as pinned | towel | 2 |
| `full_towel_50step/` | as pinned | towel | 50 |
| `full_behavior_50step/` | as pinned | behavior | 50 |
| `full_towel_50step_sdpa_contiguous/` | SDPA patch | towel | 50 |
| `full_towel_50step_final/` | both patches | towel | 50 |
| `full_behavior_50step_final/` | both patches | behavior | 50 |

| File | What it is |
| --- | --- |
| `towel_pinned_sdpa_final.jpg` | towel at frames 0/20/40/60/80 for the three configurations, the authors' clip and the real video |
| `towel_unpatched_vs_patched.json` | as pinned vs SDPA patch, decoded frame by frame (`radeon/compare_runs.py`) |
| `profile_towel_unpatched.json`, `profile_towel_patched.json` | kernel profile of one DiT forward without and with the SDPA patch (`radeon/profile_step.py`) |
| `train_smoke_time.json` | wall time and peak host RSS of the training smoke |
| `logs/isolation.log`, `logs/setup.log`, `logs/setup_final.log` | render node in the container, torch/HIP versions, installed patches |
| `logs/download.log` | weights fetched and SHA-256 checked |
| `logs/probe_attention.log` | SDPA backends and one GEMM at Wan2.2-A14B size |
| `logs/probe_layout.log` | strided vs contiguous q/k/v for SDPA |
| `logs/probe_vae_attention.log` | fused SDPA vs fp32 reference at head size 384 |
| `logs/probe_vae_encode.log` | fused vs exact VAE attention on real latents and pixels |
| `logs/profile_towel_*.log` | console output of the two profiles |
| `logs/pipeline.log` | setup, smoke and the two as-pinned 50-step runs |
| `logs/verify_patched.log`, `logs/verify_final.log` | the SDPA-patch run and the two both-patches runs |
| `logs/train_smoke_f49.log` | single-card training smoke, 49 frames |

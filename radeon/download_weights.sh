#!/bin/bash
# Host or container. Resumable downloads, each file checked against its published
# SHA-256, into the layout DiffSynth looks up under DIFFSYNTH_MODEL_BASE_PATH:
#
#   bash radeon/download_weights.sh /path/to/workspace
#
# Base model and tokenizer come from ModelScope, the two LoRAs from the Hugging Face
# Hub (set HF_ENDPOINT=https://hf-mirror.com where huggingface.co is unreachable).
#
# WanVideoPipeline.from_pretrained redirects the PAI repo's T5 and VAE .pth files to
# DiffSynth-Studio/Wan-Series-Converted-Safetensors (redirect_common_files defaults to
# True), so with DIFFSYNTH_SKIP_DOWNLOAD=true those converted files must be present.
# The .pth originals are not needed by infer.py. About 72 GB in total.
set -euo pipefail
ROOT=${1:?usage: download_weights.sh WORKSPACE}
MS=${MODELSCOPE_ENDPOINT:-https://www.modelscope.cn}/models
HF=${HF_ENDPOINT:-https://huggingface.co}
BASE=$ROOT/models/PAI/Wan2.2-Fun-A14B-Control
CONV=$ROOT/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors
TOK=$ROOT/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
CKPT=$ROOT/checkpoints
LORA_REV=624c110495ea7e29d1a0f1b3f2743daa1471c1f0
mkdir -p "$BASE/high_noise_model" "$BASE/low_noise_model" "$CONV" "$TOK" "$CKPT"

fetch() {
  local url=$1 dest=$2 sha=$3
  if [ -f "$dest" ] && [ -f "$dest.sha256ok" ]; then echo "have $dest"; return 0; fi
  local tries=0
  until curl -fL -sS --retry 20 --retry-all-errors --retry-delay 10 --speed-limit 20000 --speed-time 120 \
      -C - -o "$dest.partial" "$url"; do
    tries=$((tries + 1))
    if [ "$tries" -ge 30 ]; then echo "FAILED $url" >&2; return 1; fi
    sleep 15
  done
  local got
  got=$(sha256sum "$dest.partial" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then
    echo "SHA_MISMATCH $dest got=$got want=$sha" >&2
    mv "$dest.partial" "$dest.bad"
    return 1
  fi
  mv "$dest.partial" "$dest" && touch "$dest.sha256ok"
  echo "ok $dest $(stat -c%s "$dest")"
}

P=$MS/PAI/Wan2.2-Fun-A14B-Control/resolve/master
C=$MS/DiffSynth-Studio/Wan-Series-Converted-Safetensors/resolve/master
T=$MS/Wan-AI/Wan2.1-T2V-1.3B/resolve/master/google/umt5-xxl
fetch "$P/configuration.json" "$BASE/configuration.json" 0459fd62549fa78e18181a0892aed1d60d430c8f4eacc6509a75e2f813476e76
fetch "$P/high_noise_model/config.json" "$BASE/high_noise_model/config.json" dd2e482e8eedc156cb4ab64ac73b52cf428f90a5cffb170e928529fdb4e06733
fetch "$P/low_noise_model/config.json" "$BASE/low_noise_model/config.json" dd2e482e8eedc156cb4ab64ac73b52cf428f90a5cffb170e928529fdb4e06733
fetch "$T/special_tokens_map.json" "$TOK/special_tokens_map.json" 7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d
fetch "$T/spiece.model" "$TOK/spiece.model" e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458
fetch "$T/tokenizer.json" "$TOK/tokenizer.json" 6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b
fetch "$T/tokenizer_config.json" "$TOK/tokenizer_config.json" ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05
fetch "$C/Wan2.1_VAE.safetensors" "$CONV/Wan2.1_VAE.safetensors" a506bb08599a262f843f9c13c9a8c17e4ebdb07cc0e3612e19f79f29f9db1669

pids=()
fetch "$P/high_noise_model/diffusion_pytorch_model.safetensors" "$BASE/high_noise_model/diffusion_pytorch_model.safetensors" e4be07ca25d67aaf3461ef87fbb5ad8762f602a5efa620aea9dc250ac2747a9d & pids+=($!)
fetch "$P/low_noise_model/diffusion_pytorch_model.safetensors" "$BASE/low_noise_model/diffusion_pytorch_model.safetensors" d82bc21cd703e089eee4f6fabc25e90ebbd00965623fdc5e83d46d01dbeac2c3 & pids+=($!)
fetch "$C/models_t5_umt5-xxl-enc-bf16.safetensors" "$CONV/models_t5_umt5-xxl-enc-bf16.safetensors" d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23 & pids+=($!)
fetch "$HF/HadiZayer/masked-visual-actions/resolve/$LORA_REV/masked_world_lora_high.safetensors" "$CKPT/masked_world_lora_high.safetensors" 9ab44e8593f1c14370f89f2aac46bfa82d6006c026f2ef692fb19eda4ef0e752 & pids+=($!)
fetch "$HF/HadiZayer/masked-visual-actions/resolve/$LORA_REV/masked_world_lora_low.safetensors" "$CKPT/masked_world_lora_low.safetensors" 3ff0c821f043301854bb88433d84608eb55c569dada90989d0e2bb2ebfd857ff & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo DOWNLOAD_FAILED; exit 1; }

# Training only: DiffSynth's train.py always builds a Wav2Vec2Processor from
# Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/, even for this video-only model.
# Online it would pull the whole folder (~4 GB); the processor needs these four files.
W2V=$ROOT/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english
A=$MS/Wan-AI/Wan2.2-S2V-14B/resolve/master/wav2vec2-large-xlsr-53-english
mkdir -p "$W2V"
fetch "$A/preprocessor_config.json" "$W2V/preprocessor_config.json" ca5999a45e98bb76ea87a461ba28a23ad32a5bb9f733b8e0f6546ff38b6c612d
fetch "$A/vocab.json" "$W2V/vocab.json" 1ec5eeced764da0e4a4d2e95c1c3470c3235222db0182088beac3e7c67eda8ff
fetch "$A/special_tokens_map.json" "$W2V/special_tokens_map.json" bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675
fetch "$A/config.json" "$W2V/config.json" ffcc5c417fe11433447975d5053b2279fbeafd6bca03dd2753082e72ad2d36b7

du -sh "$ROOT/models" "$CKPT"
echo DOWNLOAD_OK

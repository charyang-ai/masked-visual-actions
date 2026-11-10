#!/bin/bash
# Inside the container. Runs upstream inference/infer.py unchanged, with its own
# --low-vram path: the two 14B experts are 53 GB in bf16 and a W7900 has 48 GB.
#
#   bash radeon/run_verify.sh smoke               2 steps on towel: both experts + VAE
#   bash radeon/run_verify.sh full [samples...]   50 steps (upstream default), towel behavior
#
# Each run writes results/<name>$RUN_TAG/{<sample>.mp4, infer.log, time.json, vram.log, metrics.json}.
set -euo pipefail
MODE=${1:-full}
shift || true
SAMPLES=${*:-towel behavior}
cd /workspace/src
PY=/workspace/venv/bin/python
H=/workspace/src/radeon
S=/workspace/data/samples
LORA=(--lora-high /workspace/checkpoints/masked_world_lora_high.safetensors --lora-low /workspace/checkpoints/masked_world_lora_low.safetensors)
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

vram_sampler() {
  while true; do
    echo "$(date +%s) $(rocm-smi --showmeminfo vram --csv 2>/dev/null | grep -E '^card' | tr '\n' ' ')"
    sleep 10
  done
}

run_one() {
  local name=$1${RUN_TAG:-} sample=$2 steps=$3
  local out=/workspace/results/$name
  mkdir -p "$out"
  if [ -f "$out/done" ]; then echo "skip $name"; return 0; fi
  vram_sampler > "$out/vram.log" &
  local sp=$!
  echo "=== $name sample=$sample steps=$steps $(date -Is)"
  $PY "$H/timed.py" "$out/time.json" $PY inference/infer.py "${LORA[@]}" \
    --control-video "$S/$sample/control.mp4" \
    --reference-image "$S/$sample/reference.png" \
    --prompt "$(cat "$S/$sample/prompt.txt")" \
    --seed "$(cat "$S/$sample/seed.txt")" \
    --num-inference-steps "$steps" \
    --low-vram --save-inputs \
    --output "$out/$sample.mp4" 2>&1 | tee "$out/infer.log"
  kill "$sp" 2>/dev/null || true
  $PY "$H/metrics.py" --sample "$S/$sample" --pred "$out/$sample.mp4" --out "$out/metrics.json" 2>&1 | tee "$out/metrics.log" \
    || echo "METRICS_FAILED $name"
  touch "$out/done"
  echo "=== $name done $(date -Is)"
}

if [ "$MODE" = "smoke" ]; then
  run_one smoke_towel_2step towel 2
else
  for s in $SAMPLES; do run_one "full_${s}_50step" "$s" 50; done
fi
echo "VERIFY_${MODE^^}_OK"

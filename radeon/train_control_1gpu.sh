#!/bin/bash
# Inside the container. Upstream's training/train_control.sh on one 48 GB card.
#
#   DATASET_CSV=/path/train.csv OUTPUT_DIR=/path/out EXPERT=low bash radeon/train_control_1gpu.sh
#
# Same base model, LoRA rank and targets, resolution, frame count (NUM_FRAMES, default 81
# as upstream), learning rate, checkpoint cadence and timestep bands as upstream. At 81
# frames one step peaks at 44.1 GiB before Adam allocates its 4.6 GiB of state, which does
# not fit 48 GiB; NUM_FRAMES=49 does. What changes is how it fits one card:
#   - one process instead of --multi_gpu;
#   - DiffSynth's split training (examples/wanvideo/model_training/special/split_training):
#     T5 and the VAE encode every clip once (sft:data_process), then sft:train loads only
#     the DiT expert. In one stage, T5 + VAE + a 14B expert + rank-256 LoRA state fill the
#     card before any activation, and --offload_models on T5 fails in this DiffSynth version
#     ("Unexpected key(s) in state_dict: weight" for its Embedding);
#   - gradient checkpoints go to host RAM (--use_gradient_checkpointing_offload).
# The cache stores the timestep band and the checkpointing flags of the encode stage, so
# both stages get them and each expert gets its own cache.
# Each optimisation step logs a TRAIN_STEP line (radeon/train_loss_log.py).
set -euo pipefail
DATASET_CSV=${DATASET_CSV:?set DATASET_CSV to your training metadata csv}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/control}
DIFFSYNTH_DIR=${DIFFSYNTH_DIR:-/workspace/DiffSynth-Studio}
NUM_EPOCHS=${NUM_EPOCHS:-5}
NUM_FRAMES=${NUM_FRAMES:-81}
EXPERT=${EXPERT:-both}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=PAI/Wan2.2-Fun-A14B-Control
T5="${BASE}:models_t5_umt5-xxl-enc-bf16.pth"
VAE="${BASE}:Wan2.1_VAE.pth"
LAUNCH=(accelerate launch --mixed_precision=bf16 --num_processes=1 --num_machines=1 --dynamo_backend=no)

train_expert () {
  local noise=$1 max_b min_b
  if [ "$noise" = "high" ]; then max_b=0.358; min_b=0; else max_b=1; min_b=0.358; fi
  local dit="${BASE}:${noise}_noise_model/diffusion_pytorch_model*.safetensors"
  local cache="${OUTPUT_DIR}_${noise}_noise_cache"
  local common=(--height 480 --width 832 --num_frames "${NUM_FRAMES}" --dataset_repeat 1
    --model_id_with_origin_paths "${dit},${T5},${VAE}"
    --learning_rate 1e-4 --num_epochs "${NUM_EPOCHS}" --remove_prefix_in_ckpt "pipe.dit."
    --lora_base_model "dit" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 256
    --save_steps 250 --extra_inputs "control_video,reference_image"
    --use_gradient_checkpointing_offload
    --max_timestep_boundary ${max_b} --min_timestep_boundary ${min_b})

  if [ ! -d "$cache/0" ]; then
    echo "=== encoding clips once for the ${noise}-noise expert into ${cache} ==="
    (cd "$DIFFSYNTH_DIR" && "${LAUNCH[@]}" examples/wanvideo/model_training/train.py \
      --dataset_base_path "" --dataset_metadata_path "${DATASET_CSV}" \
      --data_file_keys "video,control_video,reference_image" \
      "${common[@]}" --offload_models "${dit}" \
      --output_path "${cache}" --task "sft:data_process")
  fi

  echo "=== training ${noise}-noise expert (timestep boundary ${min_b}-${max_b}) ==="
  (cd "$DIFFSYNTH_DIR" && "${LAUNCH[@]}" "$HERE/train_loss_log.py" examples/wanvideo/model_training/train.py \
    --dataset_base_path "${cache}" \
    "${common[@]}" --offload_models "${T5},${VAE}" \
    --output_path "${OUTPUT_DIR}_${noise}_noise" --task "sft:train")
}

if [ "$EXPERT" = "both" ] || [ "$EXPERT" = "high" ]; then train_expert high; fi
if [ "$EXPERT" = "both" ] || [ "$EXPERT" = "low" ];  then train_expert low;  fi

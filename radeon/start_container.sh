#!/bin/bash
# Host side. One ROCm container that sees exactly one Radeon render node.
#
#   bash radeon/start_container.sh HOST_DIR /dev/dri/renderD135
#
# HOST_DIR is mounted at /workspace and holds everything that should outlive the
# container: this repo (src/), DiffSynth-Studio, the venv, weights, samples, results
# and the MIOpen/Triton/pip caches. Pick the render node with `rocm-smi --showbus`
# and `ls -l /dev/dri/by-path`. Core dumps are off: a crashed HIP process can write
# a multi-gigabyte dump.
set -euo pipefail
HOST_DIR=${1:?usage: start_container.sh HOST_DIR RENDER_NODE}
RENDER_NODE=${2:?usage: start_container.sh HOST_DIR RENDER_NODE}
NAME=${NAME:-radeon-masked-visual-actions}
IMAGE=${IMAGE:-rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0}
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)
mkdir -p "$HOST_DIR"/{models,checkpoints,data,results,logs} \
  "$HOST_DIR"/cache/{pip,miopen,triton,torch,hf}

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME" >/dev/null
else
  docker run -d --name "$NAME" \
    --entrypoint bash \
    --device=/dev/kfd \
    --device="$RENDER_NODE" \
    --group-add "$VIDEO_GID" \
    --group-add "$RENDER_GID" \
    --ipc=host \
    --ulimit core=0 \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=0 \
    -e ROCR_VISIBLE_DEVICES=0 \
    -e MIOPEN_USER_DB_PATH=/workspace/cache/miopen \
    -e MIOPEN_CUSTOM_CACHE_DIR=/workspace/cache/miopen \
    -e TRITON_CACHE_DIR=/workspace/cache/triton \
    -e TORCH_HOME=/workspace/cache/torch \
    -e XDG_CACHE_HOME=/workspace/cache \
    -e PIP_CACHE_DIR=/workspace/cache/pip \
    -e HF_HOME=/workspace/cache/hf \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e DIFFSYNTH_MODEL_BASE_PATH=/workspace/models \
    -e DIFFSYNTH_SKIP_DOWNLOAD=true \
    -e PYTHONUNBUFFERED=1 \
    -v "$HOST_DIR":/workspace \
    -w /workspace \
    "$IMAGE" \
    -lc 'sleep infinity'
fi
docker inspect --format 'devices={{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$NAME"
docker exec "$NAME" /opt/venv/bin/python -c 'import torch; print("torch", torch.__version__, "hip", torch.version.hip, "devices", torch.cuda.device_count(), torch.cuda.get_device_name(0))'

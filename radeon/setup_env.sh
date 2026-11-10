#!/bin/bash
# Inside the container. A venv on top of the image's ROCm PyTorch, plus DiffSynth-Studio
# at the commit upstream pins, with the Radeon patches in radeon/patches/:
#   diffsynth-wan-sdpa-contiguous.patch   DiT attention 1.8x faster on gfx1100
#   diffsynth-wan-vae-attention-math.patch  VAE attention correct on gfx1100
#
# A venv made from /opt/venv does not see the image's torch unless the image
# site-packages path is added back with a .pth file. DiffSynth goes in with --no-deps so
# pip cannot replace that torch with a CUDA build. NO_PATCH=1 keeps DiffSynth pristine.
set -euo pipefail
cd /workspace
DIFFSYNTH_COMMIT=3743b1307caf2562af60d475b22d4b6be68e7cd0
if [ ! -d DiffSynth-Studio ]; then
  git clone https://github.com/modelscope/DiffSynth-Studio.git
  git -C DiffSynth-Studio checkout "$DIFFSYNTH_COMMIT"
fi
if [ "${NO_PATCH:-0}" != "1" ]; then
  for PATCH in /workspace/src/radeon/patches/*.patch; do
    if git -C DiffSynth-Studio apply --reverse --check "$PATCH" 2>/dev/null; then
      echo "patch already applied: $(basename "$PATCH")"
    else
      git -C DiffSynth-Studio apply "$PATCH" && echo "patch applied: $(basename "$PATCH")"
    fi
  done
fi
if [ ! -x venv/bin/python ]; then
  /opt/venv/bin/python -m venv --system-site-packages venv
fi
SITE=$(/opt/venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
echo "$SITE" > venv/lib/python3.12/site-packages/rocm-torch.pth
PY=venv/bin/python
$PY -m pip install -q -r src/radeon/requirements-rocm.txt
$PY -m pip install -q --no-deps -e DiffSynth-Studio
$PY - <<'PY'
import torch, transformers, diffsynth
print("torch", torch.__version__, "hip", torch.version.hip, "devices", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "diffsynth", diffsynth.__file__)
import diffsynth.models.wan_video_dit as dit
print("attention: flash_attn3", dit.FLASH_ATTN_3_AVAILABLE, "flash_attn2", dit.FLASH_ATTN_2_AVAILABLE,
      "sageattention", dit.SAGE_ATTN_AVAILABLE, "-> torch SDPA" if not (dit.FLASH_ATTN_3_AVAILABLE or dit.FLASH_ATTN_2_AVAILABLE or dit.SAGE_ATTN_AVAILABLE) else "")
PY
echo SETUP_OK

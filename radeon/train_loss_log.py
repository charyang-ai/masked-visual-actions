"""Run DiffSynth's train.py unchanged, printing loss, step time and peak VRAM per step.

  accelerate launch ... radeon/train_loss_log.py examples/wanvideo/model_training/train.py [train.py args]

DiffSynth's training loop shows only a tqdm bar; this wraps Accelerator.backward so every
optimisation step leaves a TRAIN_STEP line in the log.
"""

import runpy
import sys
import time

import accelerate
import torch

_backward = accelerate.Accelerator.backward
_state = {"step": 0, "t": time.time()}


def backward(self, loss, **kwargs):
    _backward(self, loss, **kwargs)
    torch.cuda.synchronize()
    _state["step"] += 1
    now = time.time()
    print(f"TRAIN_STEP {_state['step']} loss={loss.item():.5f} dt={now - _state['t']:.1f}s "
          f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
    _state["t"] = now


accelerate.Accelerator.backward = backward
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

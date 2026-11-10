"""Compare two generated clips decoded frame by frame: python radeon/compare_runs.py A.mp4 B.mp4 [OUT.json]

Reports how many decoded frames are identical, the largest and mean absolute 8-bit
pixel difference, and PSNR. Both clips went through the same mp4 encoder, so identical
generator outputs give identical decoded frames.
"""

import json
import sys

import cv2
import numpy as np


def frames(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, im = cap.read()
        if not ok:
            return np.stack(out)
        out.append(im)


a, b = frames(sys.argv[1]), frames(sys.argv[2])
assert a.shape == b.shape, (a.shape, b.shape)
diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
mse = float((diff.astype(np.float64) ** 2).mean())
res = {
    "frames": int(len(a)),
    "identical_frames": int(sum(not d.any() for d in diff)),
    "max_abs_diff": int(diff.max()),
    "mean_abs_diff": round(float(diff.mean()), 4),
    "psnr_db": None if mse == 0 else round(10 * np.log10(255.0**2 / mse), 2),
}
print(json.dumps(res))
if len(sys.argv) > 3:
    with open(sys.argv[3], "w") as f:
        json.dump(res, f, indent=1)

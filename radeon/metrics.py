"""PSNR / SSIM / LPIPS(alex) between generated, published and real videos.

  python metrics.py --sample DIR --pred OURS.mp4 --out metrics.json

Pairs: ours vs gt, upstream vs gt, ours vs upstream, and a static baseline
(reference frame repeated) vs gt that shows how much the background alone scores.
Frames are compared at 832x480 RGB in [0, 1]. SSIM uses an 11x11 Gaussian window
(sigma 1.5) per channel. Means are over frames 1..80; frame 0 is the reference.
"""

import argparse
import json

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F


def read_video(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0


def gaussian_window(size=11, sigma=1.5):
    g = torch.exp(-((torch.arange(size) - size // 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(x, y, win):
    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = F.conv2d(x, win, groups=3), F.conv2d(y, win, groups=3)
    sxx = F.conv2d(x * x, win, groups=3) - mu_x**2
    syy = F.conv2d(y * y, win, groups=3) - mu_y**2
    sxy = F.conv2d(x * y, win, groups=3) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3))


@torch.no_grad()
def compare(a, b, net, win, dev):
    n = min(len(a), len(b))
    out = {"psnr": [], "ssim": [], "lpips": []}
    for i in range(0, n, 8):
        x, y = a[i:i + 8].to(dev), b[i:i + 8].to(dev)
        mse = ((x - y) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
        out["psnr"] += (10 * torch.log10(1 / mse)).tolist()
        out["ssim"] += ssim(x, y, win.to(dev)).tolist()
        out["lpips"] += net(x * 2 - 1, y * 2 - 1).flatten().tolist()
    return {k: float(np.mean(v[1:])) for k, v in out.items()} | {"frames": n - 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    win = gaussian_window()
    ours = read_video(args.pred)
    gt = read_video(f"{args.sample}/gt.mp4")
    upstream = read_video(f"{args.sample}/upstream.mp4")
    ref = cv2.cvtColor(cv2.imread(f"{args.sample}/reference.png"), cv2.COLOR_BGR2RGB)
    static = (torch.from_numpy(ref).permute(2, 0, 1).float() / 255.0)[None].expand(len(gt), -1, -1, -1)
    res = {
        "ours_vs_gt": compare(ours, gt, net, win, dev),
        "upstream_vs_gt": compare(upstream, gt, net, win, dev),
        "ours_vs_upstream": compare(ours, upstream, net, win, dev),
        "static_reference_vs_gt": compare(static, gt, net, win, dev),
        "shapes": {"ours": list(ours.shape), "gt": list(gt.shape), "upstream": list(upstream.shape)},
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        if k != "shapes":
            print(f"{k:24s} PSNR {v['psnr']:6.2f}  SSIM {v['ssim']:.3f}  LPIPS {v['lpips']:.4f}")


if __name__ == "__main__":
    main()

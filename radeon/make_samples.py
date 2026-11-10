"""Build the verification samples from the project page's published clips.

  git clone https://github.com/masked-visual-actions/masked-visual-actions.github.io site
  python radeon/make_samples.py site /workspace/data/samples

Each sample gets control.mp4 (the masked visual action), reference.png (first real
frame), gt.mp4 (the real video), upstream.mp4 (the authors' published output for the
same control and seed), prompt.txt, seed.txt and source.txt. All clips are 832x480,
81 frames, 15 fps, which are infer.py's defaults. Seeds come from the published file
names. The page gives no prompts, and the paper trains on one high-level prompt per
clip, so these are short task descriptions. failure1 ships its own reference frame; the
other two use frame 0 of the real video.
"""

import shutil
import sys
from pathlib import Path

import cv2

SAMPLES = {
    "towel": dict(
        control="ablation_actions/our_real_world/towel/render_ep02_seed123_control.mp4",
        gt="ablation_actions/our_real_world/towel/ep02_gt.mp4",
        upstream="ablation_actions/our_real_world/towel/render_ep02_seed123.mp4",
        reference=None,
        seed=123,
        prompt="A robot arm picks up the towel and places it in the bowl.",
    ),
    "behavior": dict(
        control="comparison_with_baselines/BEHAVIOR/ep50/rendering_000501_seed456_control.mp4",
        gt="comparison_with_baselines/BEHAVIOR/ep50/gt_episode_000501_seed456_og_video.mp4",
        upstream="comparison_with_baselines/BEHAVIOR/ep50/rendering_episode_000501_seed456.mp4",
        reference=None,
        seed=456,
        prompt="A humanoid robot reaches into the open fridge in a kitchen.",
    ),
    "failure1": dict(
        control="failure_cases/example1/ep00_seed123_control.mp4",
        gt="failure_cases/example1/ep00_seed123_gt.mp4",
        upstream="failure_cases/example1/ep00_seed123.mp4",
        reference="failure_cases/example1/ep00_seed123_reference.png",
        seed=123,
        prompt="A robot arm picks up the bread from the table.",
    ),
}


def first_frame(video, dst):
    ok, im = cv2.VideoCapture(str(video)).read()
    assert ok, video
    cv2.imwrite(str(dst), im)


def main():
    assets = Path(sys.argv[1]) / "assets"
    out = Path(sys.argv[2])
    for name, s in SAMPLES.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for key in ("control", "gt", "upstream"):
            shutil.copyfile(assets / s[key], d / f"{key}.mp4")
        if s["reference"]:
            shutil.copyfile(assets / s["reference"], d / "reference.png")
        else:
            first_frame(assets / s["gt"], d / "reference.png")
        (d / "prompt.txt").write_text(s["prompt"], encoding="utf-8")
        (d / "seed.txt").write_text(str(s["seed"]), encoding="utf-8")
        (d / "source.txt").write_text(
            "\n".join(f"{k}: {s[k]}" for k in ("control", "gt", "upstream", "reference")) + "\n", encoding="utf-8")
        print(name, sorted(p.name for p in d.iterdir()))


if __name__ == "__main__":
    main()

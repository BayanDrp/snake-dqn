#!/usr/bin/env python3
"""Plot the learning curve from a training CSV log.

Input  : CSV with columns episode,reward,best (produced by train.py --log)
Output : PNG learning curve at img/learning_curve.png

Uses matplotlib if available, otherwise falls back to a pure-PIL render.

Usage:
    python scripts/make_plot.py                 # reads training_log.csv -> img/learning_curve.png
    python scripts/make_plot.py logs/a.csv out.png
"""

import csv
import os
import sys


def read_csv(path):
    eps, rew, best = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            eps.append(float(row["episode"]))
            rew.append(float(row["reward"]))
            best.append(float(row["best"]))
    return eps, rew, best


def smooth(values, window=20):
    """Moving average (window trailing); keeps the raw curve's start."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


def plot_matplotlib(eps, reward, best, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, best, color="tab:red", lw=1.2, label="best reward")
    ax.plot(eps, smooth(reward), color="tab:blue", lw=1.4, label="reward (smoothed)")
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("DQN Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"saved {out_path} (matplotlib)")


def plot_pil(eps, reward, best, out_path):
    """Pure-PIL cartoon plot so we never hard-depend on matplotlib."""
    from PIL import Image, ImageDraw, ImageFont

    W, H, pad_l, pad_b = 860, 540, 70, 46
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = font_b = ImageFont.load_default()

    def norm(vals):
        lo = min(min(vals), 0.0)
        hi = max(max(vals), 1.0)
        span = (hi - lo) or 1.0
        return [(v - lo) / span for v in vals]

    yr = norm(reward)
    yb = norm(best)

    def pt(x, y):
        px = pad_l + x * (W - pad_l - 20) / (len(eps) - 1 or 1)
        py = H - pad_b - y * (H - pad_b - 34)
        return px, py

    d.rectangle([0, 0, W - 1, H - 1], outline="black")
    for i in range(0, 5):
        gy = H - pad_b - i * (H - pad_b - 34) / 4
        d.line([pad_l, gy, W - 20, gy], fill="#dddddd")
        d.text((4, gy - 7), f"{1 - i/4:.0%}", fill="black", font=font)

    pts_r = [pt(x, y) for x, y in zip(eps, yr)]
    pts_b = [pt(x, y) for x, y in zip(eps, yb)]
    d.line(pts_r, fill=(30, 90, 200), width=2)
    d.line(pts_b, fill=(200, 40, 40), width=1)
    d.text((pad_l + 8, 8), "DQN Learning Curve", fill="black", font=font_b)

    for i, lbl in ((0, "0"), (len(eps) // 2, str(int(eps[len(eps) // 2]))), (len(eps) - 1, str(int(eps[-1])))):
        x, y = pt(eps[i], 0)
        d.text((x - 14, H - pad_b - 22), lbl, fill="black", font=font)

    img.save(out_path)
    print(f"saved {out_path} (PIL fallback)")


def main(argv):
    csv_path = argv[0] if argv else "training_log.csv"
    out_path = argv[1] if len(argv) > 1 else os.path.join("img", "learning_curve.png")

    if not os.path.exists(csv_path):
        print(f"No log found at {csv_path}. Train with: python train.py --log training_log.csv")
        sys.exit(1)

    eps, reward, best = read_csv(csv_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        plot_matplotlib(eps, reward, best, out_path)
    except ImportError:
        plot_pil(eps, reward, best, out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
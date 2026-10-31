"""単元四 寄り道 A3：合図ゲームの多数の走行で、s の分布が一つの山から二つの山に割れていく様子を描く。

s = （送り手の玉のうち言葉 I に沿う割合）−（言葉 II に沿う割合）= (x_1A + x_2B) − (x_1B + x_2A)。
s = +1 が言葉 I、−1 が言葉 II、0 が全部半々。二つの言葉を結ぶ線（line_flow.py）の上では、この s と一致する。
規則は手順二と同じ（論文どおり）。走行の本数と時刻は config.py の U4_A3_*。

使い方（sync-lab/unit4 で）:  python3 a3_split.py   → results/a3_split.png
"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
from signaling import SignalingUrns  # noqa: E402

RUNS = config.U4_A3_RUNS
SNAPS = config.U4_A3_SNAPSHOTS


def s_of(m):
    x = m.sender / m.sender.sum(axis=(1, 2), keepdims=True)
    return (x[:, 0, 0] + x[:, 1, 1]) - (x[:, 0, 1] + x[:, 1, 0])


def main():
    rng = np.random.default_rng(config.U4_A3_SEED)
    m = SignalingUrns(RUNS)
    snaps = {}
    for n in range(1, SNAPS[-1] + 1):
        m.step(rng)
        if n in SNAPS:
            snaps[n] = s_of(m)
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    fig, axes = plt.subplots(1, len(SNAPS), figsize=(14, 2.8), sharey=True)
    bins = np.linspace(-1, 1, 41)
    for ax, n in zip(axes, SNAPS):
        ax.hist(snaps[n], bins=bins, color="#2a78d6", density=True)
        ax.set_title(f"after {n} plays", fontsize=10)
        ax.set_xlabel("s")
        ax.set_xticks([-1, 0, 1])
        ax.grid(color="0.92", lw=0.8, axis="y")
    axes[0].set_ylabel(f"density over {RUNS} runs")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "a3_split.png"), dpi=150)
    for n in SNAPS:
        s = snaps[n]
        print(n, "|s|<0.2:", round(float((np.abs(s) < 0.2).mean()), 3), " |s|>0.8:", round(float((np.abs(s) > 0.8).mean()), 3))


if __name__ == "__main__":
    main()

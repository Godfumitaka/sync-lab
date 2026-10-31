"""単元四 寄り道 A1：ポリアの壺と「x² の壺」で、白の割合の分布が引くたびにどう変わるかを厳密に追う。

どちらも白 1・黒 1 から始め、引いた色の玉を 1 個足す。違うのは引く確率だけ。
    ポリアの壺：白を引く確率 = w / (w + b)
    x² の壺    ：白を引く確率 = w² / (w² + b²)
n 回引いたあとの白の回数 k の確率 P_n(k) を、マスター方程式
    P_{n+1}(k+1) += P_n(k)·p(k, n)、 P_{n+1}(k) += P_n(k)·(1 − p(k, n))   （w = k+1、b = n−k+1）
で厳密に計算する（乱数は使わない）。

使い方（sync-lab/unit4 で）:  python3 urn_dist.py   → results/urn_dist.png
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
N_MAX = 2000
SNAPSHOTS = [2, 10, 100, 2000]


def evolve(power):
    P = np.array([1.0])
    snaps = {}
    for n in range(N_MAX):
        k = np.arange(n + 1)
        w, b = k + 1.0, n - k + 1.0
        p = w ** power / (w ** power + b ** power)
        Q = np.zeros(n + 2)
        Q[1:] += P * p
        Q[:-1] += P * (1 - p)
        P = Q
        if n + 1 in SNAPSHOTS:
            snaps[n + 1] = P.copy()
    return snaps


def main():
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (power, title) in zip(axes, [(1, "Pólya urn:  P(white) = w/(w+b)"),
                                        (2, "x² urn:  P(white) = w²/(w²+b²)")]):
        snaps = evolve(power)
        for c, n in zip(colors, SNAPSHOTS):
            P = snaps[n]
            frac = (np.arange(n + 1) + 1) / (n + 2)        # その時点の壺の白の割合
            ax.plot(frac, P * (n + 1), color=c, lw=2, label=f"after {n} draws")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("fraction of white balls in the urn")
        ax.grid(color="0.92", lw=0.8)
    axes[0].set_ylim(0, 2)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-3, 1e3)
    axes[0].set_ylabel("probability density")
    axes[1].set_ylabel("probability density (log scale)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "results", "urn_dist.png"), dpi=150)


if __name__ == "__main__":
    main()

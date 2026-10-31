"""単元四：二つの言葉を結ぶ一本の線の上の、平均の流れ（論文 §2 の F を、その線に制限したもの）。

壺の状態は、送り手の玉の割合 x = (x_1A, x_1B, x_2A, x_2B)（論文の X_n、和は 1）で表せる。
線：x(s) = ((1+s)/4, (1−s)/4, (1−s)/4, (1+s)/4)。s = 1 が言葉 I、s = −1 が言葉 II、s = 0 が全部半々。
この線の上では、論文の F は線に沿った向きを持ち、s の流れは ds/dt = 4·F_1A になる（一回ごとに Δt = 1/(玉の数)。玉は勝ったときだけ増えるので log(玉の数) ≈ ψt）。
F は論文 §2 の式（F(x) = E(|ψ_n|(ψ_n − X_n) | X_n = x)）をそのまま数値で計算する。比べのためにポリアの壺（流れ 0）も描く。

使い方（sync-lab/unit4 で）:  python3 line_flow.py   → results/line_flow.png
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def F(x):
    """論文 §2 の F。x = (x_1A, x_1B, x_2A, x_2B)。"""
    a, b, c, d = x
    # 送り手と受け手で同じ対が引かれて勝つ確率（論文 (2.2)）の 2 倍
    t = np.array([a * a / ((a + b) * (a + c)), b * b / ((b + a) * (b + d)),
                  c * c / ((c + d) * (c + a)), d * d / ((d + c) * (d + b))])
    return 0.5 * t - 0.5 * t.sum() * np.asarray(x)


def main():
    s = np.linspace(-0.999, 0.999, 401)
    ds = np.array([4 * F([(1 + v) / 4, (1 - v) / 4, (1 - v) / 4, (1 + v) / 4])[0] for v in s])
    V = -np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(s))])

    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(s, ds, color="#2a78d6", lw=2, label="signaling game")
    a1.plot(s, 0 * s, color="#eb6834", lw=2, ls="--", label="Pólya urn (no flow)")
    a1.axhline(0, color="0.6", lw=0.8)
    a1.set_xlabel("s   (−1: language II,  0: all urns half-half,  +1: language I)")
    a1.set_ylabel("ds/dt  (Δt = 1/balls per play)")
    a1.legend(frameon=False, fontsize=8)
    a2.plot(s, V - V[len(s) // 2], color="#2a78d6", lw=2)
    a2.set_xlabel("s")
    a2.set_ylabel("V(s),  ds/dt = −dV/ds")
    for ax in (a1, a2):
        ax.grid(color="0.92", lw=0.8)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "results", "line_flow.png"), dpi=150)


if __name__ == "__main__":
    main()

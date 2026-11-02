"""単元四 寄り道：手順三 (a) の壺の中身から、渡っている情報の量 I（状態 ; 行為）と得 ψ を計算する。

I（状態 ; 行為）：受け手の行為を見ると、状態について何ビットわかるか。
    p(i, j) = π_i Σ_x P(合図 x | 状態 i) P(行為 j | 合図 x)、 I = Σ p(i,j) log₂ [p(i,j) / (p(i) p(j))]
受け手の壺は対称性から送り手と同じ数（step3a_urns.csv には送り手の数だけを保存している）。
比べのため、状態のエントロピー H(π) = −π log₂ π − (1−π) log₂(1−π) を描く。

使い方（sync-lab/unit4 で）:  python3 info_step3a.py   → results/info_step3a.png
"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def info_and_success(V, pi):
    """V: (走行, 2, 2) の送り手の玉の数 [状態, 合図]。受け手の壺 [合図, 行為] は V の転置。"""
    ps = V / V.sum(axis=2, keepdims=True)                     # P(合図 | 状態)
    R = np.transpose(V, (0, 2, 1))
    pr = R / R.sum(axis=2, keepdims=True)                     # P(行為 | 合図)
    p_state = np.array([pi, 1 - pi])
    joint = p_state[None, :, None] * np.einsum("rix,rxj->rij", ps, pr)   # p(状態 i, 行為 j)
    pi_ = joint.sum(axis=2, keepdims=True)
    pj_ = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(joint > 0, joint * np.log2(joint / (pi_ * pj_)), 0.0)
    return term.sum(axis=(1, 2)), np.trace(joint, axis1=1, axis2=2)


def H(p):
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main():
    U = np.genfromtxt(os.path.join(RESULTS, "step3a_urns.csv"), delimiter=",", names=True)
    n_last = U["n"].max()
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    rng = np.random.default_rng(0)       # 点を横に少しずらすためだけ
    for k, pi in enumerate(config.U4_STEP3A_PIS):
        d = U[(U["pi"] == pi) & (U["n"] == n_last)]
        V = np.stack([d["V_1A"], d["V_1B"], d["V_2A"], d["V_2B"]], axis=1).reshape(-1, 2, 2)
        I, psi = info_and_success(V, pi)
        x = k + rng.uniform(-0.25, 0.25, len(I))
        a1.plot(x, I, ".", ms=3, color="#2a78d6", alpha=0.4)
        a1.plot([k - 0.35, k + 0.35], [H(pi)] * 2, color="#eb6834", lw=2)
        a2.plot(I, psi, ".", ms=3, alpha=0.5, color=config.SERIES_COLORS_6[k], label=f"π = {pi:g}")
        print(f"π = {pi}: H = {H(pi):.3f} bit,  I の中央値 {np.median(I):.3f},  I < 0.05 の走行 {int((I < 0.05).sum())} 本"
              f"（その ψ の中央値 {np.median(psi[I < 0.05]) if (I < 0.05).any() else float('nan'):.3f}）")
    a1.set_xticks(range(len(config.U4_STEP3A_PIS)))
    a1.set_xticklabels([f"π = {p:g}" for p in config.U4_STEP3A_PIS])
    a1.set_ylabel("I(state ; act)  [bit]")
    a1.set_title("dots: runs at n = 10^6,   orange: H(π)", fontsize=9)
    a2.set_xlabel("I(state ; act)  [bit]")
    a2.set_ylabel("ψ: success probability")
    a2.legend(frameon=False, fontsize=8, markerscale=4)
    for ax in (a1, a2):
        ax.grid(color="0.92", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "info_step3a.png"), dpi=150)


if __name__ == "__main__":
    main()

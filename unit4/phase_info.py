"""単元四 寄り道：合図を無視する状態の近くの相図（q_A, y）と、渡っている情報の量。

相図：a = (0.5, 0.5) のとき、状態 2 の玉の割合 y と、状態 2 で合図 A を送る割合 q_A は、平均の流れの近似で
    d(log y)/dt = λ(q_A) = 2(1 − π)(q_A² + q_B²) − π
    dq_A/dt     = 2(1 − π) q_A q_B (2 q_A − 1)
に従う（時間は一回ごとに Δt = 1/M）。この流れの矢印に、pooling_exponent.py の実際の道筋を重ねる。

使い方（sync-lab/unit4 で）:  python3 phase_info.py   → results/phase_portrait.png
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


def flow(qA, logy, pi):
    qB = 1 - qA
    dlogy = 2 * (1 - pi) * (qA ** 2 + qB ** 2) - pi
    dq = 2 * (1 - pi) * qA * qB * (2 * qA - 1)
    return dq, dlogy


def main():
    D = np.genfromtxt(os.path.join(RESULTS, "pooling_exponent.csv"), delimiter=",", names=True)
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    fig, axes = plt.subplots(1, len(config.U4_PX_PIS), figsize=(12, 5), squeeze=False)
    for ax, pi in zip(axes[0], config.U4_PX_PIS):
        d = D[D["pi"] == pi]
        Q, LY = np.meshgrid(np.linspace(0.02, 0.98, 21), np.linspace(np.log(1e-6), np.log(0.3), 21))
        dq, dly = flow(Q, LY, pi)
        ax.quiver(Q, np.exp(LY), dq, dly * np.exp(LY), angles="xy", color="0.6", width=0.003,
                  scale_units="xy", scale=None, alpha=0.8)
        # λ = 0 の線（q_A の境目）
        disc = (2 * pi - 1) / (1 - pi)
        if disc < 1:
            for sgn in (-1, 1):
                ax.axvline(0.5 * (1 + sgn * np.sqrt(disc)), color="#eb6834", lw=1.5, ls="--")
            ax.text(0.5, 0.17, "λ < 0 between the dashed lines", ha="center", fontsize=8, color="#eb6834")
        else:
            ax.text(0.5, 0.17, "λ < 0 for every q_A", ha="center", fontsize=8, color="#eb6834")
        runs = np.unique(d["run"])[: config.U4_PHASE_SHOW_RUNS]
        for k in runs:
            dk = d[d["run"] == k]
            esc = dk["y"][-1] > config.U4_PX_Y0
            ax.plot(dk["q_A"], dk["y"], color="#2a78d6" if not esc else "#1baf7a", lw=0.8, alpha=0.7)
            ax.plot(dk["q_A"][-1], dk["y"][-1], "o", ms=3, color="#2a78d6" if not esc else "#1baf7a")
        ax.plot([0.5], [config.U4_PX_Y0], "k*", ms=10)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 0.3)
        ax.set_xlim(0, 1)
        ax.set_xlabel("q_A  (state 2 → signal A)")
        ax.set_ylabel("y  (fraction of state-2 balls)")
        ax.set_title(f"π = {pi}:  arrows = mean flow,  lines = runs (green: y grew)", fontsize=9)
        ax.grid(color="0.92", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "phase_portrait.png"), dpi=150)


if __name__ == "__main__":
    main()

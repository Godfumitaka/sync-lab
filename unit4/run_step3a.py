"""単元四 手順三 (a)：状態の確率が偏っている場合（二状態・二合図・二行為）。論文では証明されておらず、§4 にシミュレーションの傾向として書かれているだけ。

規則は手順二と同じで、Nature が状態 1 を確率 π、状態 2 を確率 1 − π で選ぶことだけが違う。
π ごとに U4_STEP3A_RUNS 本を U4_STEP3A_CHECKPOINTS[-1] 回まで回し、各判定時刻で分類する（基準は ASSUMPTIONS.md の 14）。

出力（results/）
    step3a_counts.csv   π, 時刻, 分類ごとの本数, ψ の平均
    step3a_urns.csv     π, 時刻, 走行, 送り手の玉の数 V(1,A), V(1,B), V(2,A), V(2,B)（受け手は対称性から同じ）
    step3a.png          π ごとの内訳
"""

import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
from signaling import SignalingUrns  # noqa: E402
from run_step2 import win_prob  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CLASSES = ["language I", "language II", "ignore signal, act 1", "ignore signal, act 2", "other"]


def classify(m):
    th = config.U4_CLASS_PROP
    ps = m.sender / m.sender.sum(axis=2, keepdims=True)        # P(合図 | 状態)
    pr = m.receiver / m.receiver.sum(axis=2, keepdims=True)    # P(行為 | 合図)
    lang1 = (ps[:, 0, 0] >= th) & (ps[:, 1, 1] >= th) & (pr[:, 0, 0] >= th) & (pr[:, 1, 1] >= th)
    lang2 = (ps[:, 0, 1] >= th) & (ps[:, 1, 0] >= th) & (pr[:, 0, 1] >= th) & (pr[:, 1, 0] >= th)
    ign1 = (pr[:, 0, 0] >= th) & (pr[:, 1, 0] >= th)
    ign2 = (pr[:, 0, 1] >= th) & (pr[:, 1, 1] >= th)
    cls = np.full(m.n_runs, 4)
    cls[ign2] = 3
    cls[ign1] = 2
    cls[lang2] = 1
    cls[lang1] = 0
    return cls


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(config.U4_STEP3A_SEED)
    models = {pi: SignalingUrns(config.U4_STEP3A_RUNS, state_probs=[pi, 1 - pi]) for pi in config.U4_STEP3A_PIS}
    cps = set(config.U4_STEP3A_CHECKPOINTS)
    counts, urns = [], []
    t0 = time.time()
    for n in range(1, config.U4_STEP3A_CHECKPOINTS[-1] + 1):
        for m in models.values():
            m.step(rng)
        if n in cps:
            for pi, m in models.items():
                cls = classify(m)
                counts.append([pi, n] + [int((cls == k).sum()) for k in range(len(CLASSES))]
                              + [float(win_prob(m).mean())])
                for r in range(m.n_runs):
                    urns.append([pi, n, r] + list(m.sender[r].ravel()))
            print(f"n = {n}: {time.time() - t0:.0f} s", flush=True)

    with open(os.path.join(RESULTS, "step3a_counts.csv"), "w") as fh:
        fh.write("pi,n," + ",".join(c.replace(" ", "_").replace(",", "") for c in CLASSES) + ",psi_mean\n")
        for row in counts:
            fh.write(",".join(f"{v:g}" for v in row[:2]) + "," + ",".join(str(v) for v in row[2:-1])
                     + f",{row[-1]:.6f}\n")
    np.savetxt(os.path.join(RESULTS, "step3a_urns.csv"), np.array(urns), delimiter=",", comments="",
               header="pi,n,run,V_1A,V_1B,V_2A,V_2B", fmt=["%g", "%d", "%d", "%d", "%d", "%d", "%d"])

    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "0.75"]
    n_last = config.U4_STEP3A_CHECKPOINTS[-1]
    rows = [r for r in counts if r[1] == n_last]
    pis = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bottom = np.zeros(len(rows))
    for k, name in enumerate(CLASSES):
        frac = np.array([r[2 + k] for r in rows]) / config.U4_STEP3A_RUNS
        ax.bar(range(len(rows)), frac, bottom=bottom, color=colors[k], edgecolor="white", lw=2, label=name)
        bottom += frac
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"π = {p:g}" for p in pis])
    ax.set_ylabel(f"fraction of {config.U4_STEP3A_RUNS} runs at n = 10^{int(np.log10(n_last))}")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(color="0.92", lw=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "step3a.png"), dpi=150)

    print("\n内訳（本数）  " + " / ".join(CLASSES) + "   ψ の平均")
    for row in counts:
        print(f"  π = {row[0]:g}, n = {row[1]:>7d}: " + " / ".join(str(v) for v in row[2:-1]) + f"   {row[-1]:.4f}")


if __name__ == "__main__":
    main()

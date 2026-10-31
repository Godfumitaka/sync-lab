"""単元四 手順二：論文で証明された場合（二状態・二合図・二行為、状態は等確率）を走らせる。

使い方（sync-lab/unit4 で）:
    python3 run_step2.py

一本の走行：U4_STEP2_N 回。対数等間隔の時刻で、win_n/n、ψ（その時点の壺から計算した次の一回の勝つ確率、論文外の量）、
            壺の中身の割合を記録する。
多数の走行：U4_STEP2_RUNS 本を同時に U4_STEP2_N 回。U4_STEP2_CHECKPOINTS の各時刻で落ち着き先を判定する
            （判定の基準は ASSUMPTIONS.md の 10）。

出力（results/）
    step2_single.csv, step2_single.png   一本の走行の推移
    step2_many.csv                        多数の走行の、判定の時刻ごとの内訳
    step2_final.csv                       多数の走行の、最後の時刻での各走行の値
    step2_many.png                        内訳の推移と、最後の時刻での成功率の分布
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

RESULTS = os.path.join(HERE, "results")
CLASSES = ["language I", "language II", "other"]


def proportions(m):
    """送り手 P(合図 | 状態) と受け手 P(行為 | 合図)。"""
    ps = m.sender / m.sender.sum(axis=2, keepdims=True)
    pr = m.receiver / m.receiver.sum(axis=2, keepdims=True)
    return ps, pr


def win_prob(m):
    """ψ = Σ_i P(状態 i) Σ_x P(合図 x | 状態 i) P(行為 i | 合図 x)。"""
    ps, pr = proportions(m)
    pi = np.diff(np.concatenate([[0.0], m.state_cum]))
    n_states = ps.shape[1]
    hit = np.einsum("rix,rxi->ri", ps, pr[:, :, :n_states])
    return hit @ pi


def classify(m):
    ps, pr = proportions(m)
    smap = ps.argmax(axis=2)          # 送り手：状態 → 多数派の合図
    rmap = pr.argmax(axis=2)          # 受け手：合図 → 多数派の行為
    psi = win_prob(m)
    good = psi >= config.U4_CLASS_THRESHOLD
    lang1 = good & (smap == [0, 1]).all(axis=1) & (rmap == [0, 1]).all(axis=1)
    lang2 = good & (smap == [1, 0]).all(axis=1) & (rmap == [1, 0]).all(axis=1)
    cls = np.full(m.n_runs, 2)
    cls[lang1] = 0
    cls[lang2] = 1
    return cls, psi


def single_run():
    rng = np.random.default_rng(config.U4_SEED)
    m = SignalingUrns(1)
    n_end = config.U4_STEP2_N
    rec = set(np.unique(np.logspace(0, np.log10(n_end), config.U4_SINGLE_RECORD_POINTS).astype(int)))
    rows = []
    for n in range(1, n_end + 1):
        m.step(rng)
        if n in rec:
            ps, pr = proportions(m)
            rows.append((n, m.wins[0] / n, win_prob(m)[0],
                         ps[0, 0, 0], ps[0, 1, 0], pr[0, 0, 0], pr[0, 1, 0]))
    return np.array(rows)


def many_runs():
    rng = np.random.default_rng(config.U4_SEED_MANY)
    m = SignalingUrns(config.U4_STEP2_RUNS)
    cps = set(config.U4_STEP2_CHECKPOINTS)
    out = {}
    for n in range(1, config.U4_STEP2_N + 1):
        m.step(rng)
        if n in cps:
            cls, psi = classify(m)
            out[n] = (cls, psi, m.wins / n)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    single = single_run()
    t1 = time.time()
    many = many_runs()
    t2 = time.time()
    print(f"一本の走行 {t1 - t0:.1f} s、多数の走行 {t2 - t1:.1f} s")

    np.savetxt(os.path.join(RESULTS, "step2_single.csv"), single, delimiter=",", comments="",
               header="n,win_rate,psi,sender_P(A|1),sender_P(A|2),receiver_P(1|A),receiver_P(1|B)",
               fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"])
    with open(os.path.join(RESULTS, "step2_many.csv"), "w") as fh:
        fh.write("n," + ",".join(c.replace(" ", "_") for c in CLASSES) +
                 ",psi_mean,psi_min,win_rate_mean,win_rate_min\n")
        for n in config.U4_STEP2_CHECKPOINTS:
            cls, psi, wr = many[n]
            cnt = [int((cls == k).sum()) for k in range(3)]
            fh.write(f"{n}," + ",".join(map(str, cnt)) +
                     f",{psi.mean():.6f},{psi.min():.6f},{wr.mean():.6f},{wr.min():.6f}\n")
    n_last = config.U4_STEP2_CHECKPOINTS[-1]
    cls, psi, wr = many[n_last]
    np.savetxt(os.path.join(RESULTS, "step2_final.csv"), np.column_stack([np.arange(len(cls)), cls, psi, wr]),
               delimiter=",", comments="", header="run,class_0I_1II_2other,psi,win_rate",
               fmt=["%d", "%d", "%.6f", "%.6f"])

    # 図
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    C = config.SERIES_COLORS_6
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    n = single[:, 0]
    a1.plot(n, single[:, 1], color=C[0], lw=1.5, label="win_n / n (paper)")
    a1.plot(n, single[:, 2], color=C[1], lw=1.5, label="ψ: win prob. from current urns (not in paper)")
    a1.set_ylabel("success")
    a1.set_ylim(0, 1.02)
    a1.legend(frameon=False, fontsize=8)
    a1.grid(color="0.92", lw=0.8)
    labels = ["sender P(A | state 1)", "sender P(A | state 2)",
              "receiver P(act 1 | A)", "receiver P(act 1 | B)"]
    for k in range(4):
        a2.plot(n, single[:, 3 + k], color=C[2 + k], lw=1.5, ls="-" if k < 2 else "--", label=labels[k])
    a2.set_xscale("log")
    a2.set_xlabel("n (plays)")
    a2.set_ylabel("urn proportion")
    a2.set_ylim(-0.02, 1.02)
    a2.legend(frameon=False, fontsize=8)
    a2.grid(color="0.92", lw=0.8)
    fig.suptitle("one run (2 states, 2 signals, 2 acts, equiprobable states)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "step2_single.png"), dpi=150)
    plt.close(fig)

    fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cps = config.U4_STEP2_CHECKPOINTS
    bottom = np.zeros(len(cps))
    for k, name in enumerate(CLASSES):
        frac = np.array([(many[c][0] == k).mean() for c in cps])
        b1.bar(range(len(cps)), frac, bottom=bottom, color=[C[0], C[1], "0.75"][k],
               edgecolor="white", lw=2, label=name)
        bottom += frac
    b1.set_xticks(range(len(cps)))
    b1.set_xticklabels([f"10^{int(np.log10(c))}" for c in cps])
    b1.set_xlabel("n (plays)")
    b1.set_ylabel(f"fraction of {config.U4_STEP2_RUNS} runs")
    b1.legend(frameon=False, fontsize=8)
    bins = np.linspace(0.5, 1.0, 51)
    b2.hist(psi, bins=bins, color=C[1], alpha=0.85, label="ψ (not in paper)")
    b2.hist(wr, bins=bins, histtype="step", color=C[0], lw=1.5, label="win_n / n (paper)")
    b2.set_xlabel(f"success at n = 10^{int(np.log10(n_last))}")
    b2.set_ylabel("runs")
    b2.legend(frameon=False, fontsize=8)
    for ax in (b1, b2):
        ax.grid(color="0.92", lw=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "step2_many.png"), dpi=150)
    plt.close(fig)

    print("\n内訳（走行の数）")
    print("  n         " + "  ".join(f"{c:>12s}" for c in CLASSES) + "   ψ の平均   ψ の最小   win/n の平均")
    for c in cps:
        cl, ps_, wr_ = many[c]
        print(f"  {c:<8d}  " + "  ".join(f"{int((cl == k).sum()):>12d}" for k in range(3)) +
              f"   {ps_.mean():.4f}     {ps_.min():.4f}     {wr_.mean():.4f}")
    print(f"\n保存：{RESULTS}/step2_*")


if __name__ == "__main__":
    main()

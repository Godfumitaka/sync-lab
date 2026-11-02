"""単元四 寄り道：(イ) が (ア) に崩れる流れを理解するための小さな模型。受け手の壺 A だけを追い、送り手は固定する。

設定
    状態 1, 2, 3 が 1/3 ずつ来る。送り手：状態 1 → 必ず A、状態 2 → 必ず B、状態 3 → 確率 q_A で A、それ以外は B（q_A は固定）。
    受け手の壺 A：行為 1 と行為 3 の玉だけ（初期は config の U4_TOY_INIT_*）。合図 A が届いたときだけ一つ引き、
    行為が状態と一致したら勝ちで、同じ行為の玉を 1 個足す。負けや合図 B のときは何もしない。
    引いた回数 n は、合図 A が届かなかった回も含めて数える。

実験
    1. q_A ごとに U4_TOY_RUNS 本を n = U4_TOY_N まで。r_A（行為 3 の玉の割合）を横軸 log n で描き、平均の流れの ODE
           dr_A/dn = −(1/(3N_A)) r_A (1 − r_A)(1 − q_A)、 dN_A/dn = (1/3)(1 − r_A + q_A r_A)
       を数値積分した曲線を重ねる。
    2. 色ごとの「引いた回数」「勝った回数」と勝率を、予測 1/(1+q_A)（行為 1）、q_A/(1+q_A)（行為 3）と並べる。
    3. q_A = 1.0 で U4_TOY_BETA_RUNS 本、n = U4_TOY_N での r_A のヒストグラムに、Beta(初期の行為 3, 初期の行為 1) の密度を重ねる。

出力（results/）：toy_urnA_r.png、toy_urnA_beta.png、toy_urnA_winrates.csv
使い方（sync-lab/unit4 で）:  python3 toy_urnA.py
"""

import math
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def simulate(qA, runs, n_steps, rng, record=True):
    c1 = np.full(runs, float(config.U4_TOY_INIT_ACT1))
    c3 = np.full(runs, float(config.U4_TOY_INIT_ACT3))
    draws = np.zeros(2, dtype=np.int64)    # [行為 1, 行為 3]
    wins = np.zeros(2, dtype=np.int64)
    rec = sorted(set(np.unique(np.logspace(0, np.log10(n_steps), config.U4_TOY_RECORD_POINTS).astype(int))))
    rec_set, out_n, out_r = set(rec), [], []
    for n in range(1, n_steps + 1):
        u = rng.random((3, runs))
        state = (u[0] * 3).astype(int)                              # 0, 1, 2 → 状態 1, 2, 3
        sigA = (state == 0) | ((state == 2) & (u[1] < qA))         # 合図 A が届いたか
        pick3 = u[2] * (c1 + c3) < c3                                # 壺 A から行為 3 を引いたか
        drew1 = sigA & ~pick3
        drew3 = sigA & pick3
        win1 = drew1 & (state == 0)
        win3 = drew3 & (state == 2)
        c1 += win1
        c3 += win3
        draws += [drew1.sum(), drew3.sum()]
        wins += [win1.sum(), win3.sum()]
        if record and n in rec_set:
            out_n.append(n)
            out_r.append(c3 / (c1 + c3))
    return np.array(out_n), np.array(out_r), c3 / (c1 + c3), draws, wins


def ode(qA, n_steps):
    """平均の流れの ODE を n について刻み 1 で積分する（記録は対数等間隔）。"""
    N = float(config.U4_TOY_INIT_ACT1 + config.U4_TOY_INIT_ACT3)
    r = config.U4_TOY_INIT_ACT3 / N
    rec = set(np.unique(np.logspace(0, np.log10(n_steps), config.U4_TOY_RECORD_POINTS).astype(int)))
    ns, rs = [], []
    for n in range(1, n_steps + 1):
        dr = -(1.0 / (3.0 * N)) * r * (1 - r) * (1 - qA)
        dN = (1.0 / 3.0) * (1 - r + qA * r)
        r += dr
        N += dN
        if n in rec:
            ns.append(n)
            rs.append(r)
    return np.array(ns), np.array(rs)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(config.U4_TOY_SEED)
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})

    # 実験 1・2
    qs = config.U4_TOY_QS
    fig, axes = plt.subplots(1, len(qs), figsize=(5 * len(qs), 4.2))
    rows = []
    for ax, qA in zip(axes, qs):
        ns, rs, _, draws, wins = simulate(qA, config.U4_TOY_RUNS, config.U4_TOY_N, rng)
        for k in range(rs.shape[1]):
            ax.plot(ns, rs[:, k], color="#2a78d6", lw=0.8, alpha=0.45)
        on, orr = ode(qA, config.U4_TOY_N)
        ax.plot(on, orr, color="#eb6834", lw=2.4, label="mean-flow ODE")
        ax.set_xscale("log")
        if qA < 1:
            ax.set_yscale("log")
        ax.set_xlabel("n (plays, including those without signal A)")
        ax.set_ylabel("r_A (fraction of act-3 balls in urn A)")
        ax.set_title(f"q_A = {qA:g}  ({config.U4_TOY_RUNS} runs)", fontsize=10)
        ax.grid(color="0.92", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
        rows.append((qA, draws[0], wins[0], wins[0] / draws[0], 1 / (1 + qA),
                     draws[1], wins[1], wins[1] / max(draws[1], 1), qA / (1 + qA)))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "toy_urnA_r.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(RESULTS, "toy_urnA_winrates.csv"), "w") as fh:
        fh.write("q_A,act1_draws,act1_wins,act1_winrate,act1_predicted,act3_draws,act3_wins,act3_winrate,act3_predicted\n")
        for r in rows:
            fh.write("%g,%d,%d,%.4f,%.4f,%d,%d,%.4f,%.4f\n" % r)

    # 実験 3
    _, _, r_end, _, _ = simulate(1.0, config.U4_TOY_BETA_RUNS, config.U4_TOY_N, rng, record=False)
    a, b = config.U4_TOY_INIT_ACT3, config.U4_TOY_INIT_ACT1
    x = np.linspace(0, 1, 400)
    beta_pdf = x ** (a - 1) * (1 - x) ** (b - 1) / (math.gamma(a) * math.gamma(b) / math.gamma(a + b))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(r_end, bins=np.linspace(0, 1, 41), density=True, color="#2a78d6", alpha=0.85,
            label=f"r_A at n = 10^{int(np.log10(config.U4_TOY_N))} ({config.U4_TOY_BETA_RUNS} runs)")
    ax.plot(x, beta_pdf, color="#eb6834", lw=2.2, label=f"Beta({a}, {b}) density")
    ax.set_xlabel("r_A")
    ax.set_ylabel("density")
    ax.set_title("q_A = 1.0", fontsize=10)
    ax.grid(color="0.92", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "toy_urnA_beta.png"), dpi=150)
    plt.close(fig)

    print("色ごとの勝率（全走行の合計）")
    print("  q_A   行為1: 引いた / 勝った / 勝率（予測）        行為3: 引いた / 勝った / 勝率（予測）")
    for r in rows:
        print(f"  {r[0]:<4g}  {r[1]:>8d} / {r[2]:>8d} / {r[3]:.4f}（{r[4]:.4f}）    {r[5]:>7d} / {r[6]:>7d} / {r[7]:.4f}（{r[8]:.4f}）")
    print(f"\n実験 3：r_A の平均 {r_end.mean():.4f}（Beta の平均 {a / (a + b):.4f}）、"
          f"標準偏差 {r_end.std():.4f}（Beta の標準偏差 {math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))):.4f}）")


if __name__ == "__main__":
    main()

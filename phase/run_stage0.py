"""位相の模型 段階0：二つのリズム（アドラー方程式）を動かして見せる。

使い方（sync-lab/phase で）:
    python3 run_stage0.py           図とアニメーション（計算は 30 秒ほど）

出力（results/）
    stage0_cases.png    三つの例（揃う／境目のすぐ外／遠く外）：ずれの累計、ずれの速さ、流れの図
    stage0_beat.png     Δω/K を 0 から 2.5 まで振ったときの、平均のずれの速さ
    stage0_beat.csv     その数値
    stage0_anim.gif     三つの例を、円の上の二つの点として動かしたもの
    stage0_log.txt      値と計算時間
"""

import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
import plotting  # noqa: E402
from rhythm import beat, phi_rate, run_pair  # noqa: E402

RESULTS = os.path.join(HERE, "results")
C1, C2 = config.PH_COLOR_BODY1, config.PH_COLOR_BODY2
CASE_NAMES = ["揃う", "境目のすぐ外", "遠く外"]


def case_title(i, dw, K):
    return f"{CASE_NAMES[i]}（Δω = {dw / K:g} K）"


def plot_cases(K, path):
    cases = config.PH0_CASES
    t, a, b = run_pair(np.array(cases), K, config.PH0_T, config.PH0_DT)
    phi = a - b
    fig, axs = plt.subplots(3, 3, figsize=(14, 10.5), gridspec_kw={"height_ratios": [1, 1, 1.15]})
    band = config.PH0_NEAR_LOCK * K
    for j, dw in enumerate(cases):
        a0, a1, a2 = axs[:, j]
        p = phi[:, j]
        r = phi_rate(p, dw, K)
        a0.plot(t, p / (2 * np.pi), color="0.15", lw=1.8)
        a0.set_title(case_title(j, dw, K), loc="left", fontsize=12)
        a1.axhspan(-band, band, color=C1, alpha=0.12, lw=0)
        a1.plot(t, r, color="0.15", lw=1.4)
        a1.axhline(0, color="0.5", lw=0.8)
        # 流れの図：ずれ φ（円の上）に対する速さ。点は一定の時間おきの位置（込んでいるところで、ゆっくり進む）
        g = np.linspace(0, 2 * np.pi, 400)
        a2.plot(g, phi_rate(g, dw, K), color="0.15", lw=1.8)
        a2.axhline(0, color="0.5", lw=0.8)
        step = int(round(1.0 / config.PH0_DT))
        pw = np.mod(p[::step], 2 * np.pi)
        a2.plot(pw, phi_rate(pw, dw, K), "o", color=C1, ms=4, alpha=0.6)
        a2.set_xlim(0, 2 * np.pi)
        a2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        a2.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        a2.set_xlabel("ずれ φ（円の上で）")
        a1.set_xlabel("時間 t")
        a0.set_xlabel("時間 t")
        a1.set_ylim(min(-0.25, r.min() - 0.2), max(0.6, r.max() + 0.2))
        a2.set_ylim(min(dw - K, 0.0) - 0.25, dw + K + 0.25)     # 速さ 0 の線がいつも見えるように
        if abs(dw) < K:      # 止まる点：●は戻ってくる点、○は離れていく点
            ps = np.arcsin(dw / K)
            a2.plot([ps], [0], "o", color="0.15", ms=9, zorder=5)
            a2.plot([np.pi - ps], [0], "o", mfc="white", mec="0.15", mew=1.5, ms=9, zorder=5)
    axs[0, 0].set_ylabel("ずれの累計 φ/2π\n（何周ずれたか）")
    axs[1, 0].set_ylabel("ずれの速さ dφ/dt\n（帯：ほぼ揃っている）")
    axs[2, 0].set_ylabel("流れの図：ずれ φ のときの速さ\n（青い点：時間 1 おきの位置。●○：止まる点）")
    fig.suptitle(f"段階0：二つのリズム。合わせる強さ K = {K:g}、平均のリズム ω̄ = {config.PH0_OMEGA_MEAN:g}"
                 f"（一周 ≈ {2 * np.pi / config.PH0_OMEGA_MEAN:.1f}）、雑音なし", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return t, phi


def plot_beat(K, path_png, path_csv):
    x = np.linspace(config.PH0_SWEEP_MIN, config.PH0_SWEEP_MAX, config.PH0_SWEEP_N)
    rate = beat(x * K, K, config.PH0_SWEEP_T, config.PH0_SWEEP_SKIP, config.PH0_DT)
    np.savetxt(path_csv, np.column_stack([x, rate / K]), delimiter=",",
               header="d_omega_over_K,mean_phi_rate_over_K", comments="", fmt="%.6g")
    fig, a1 = plt.subplots(figsize=(8, 4.8))
    a1.plot(x, rate / K, color="0.15", lw=2)
    for j, dw in enumerate(config.PH0_CASES):
        a1.axvline(dw / K, color="0.6", lw=1, ls=":")
        a1.text(dw / K + 0.02, 0.97 - 0.07 * j, CASE_NAMES[j], transform=a1.get_xaxis_transform(), fontsize=9,
                color="0.4", va="top")
    a1.set_xlabel("リズムの差 Δω / 合わせる強さ K")
    a1.set_ylabel("平均のずれの速さ ⟨dφ/dt⟩ / K")
    a1.set_title("長く見たときの、ずれの進み方（＝ 2π × 単位時間あたりの追い越しの回数）", loc="left", fontsize=11)
    fig.suptitle(f"段階0：Δω を振る（各 {config.PH0_SWEEP_T:g} の時間、はじめの {config.PH0_SWEEP_SKIP:g} を捨てる）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    return x, rate


def make_anim(K, path):
    cases = config.PH0_CASES
    fdt = config.PH0_ANIM_FRAME_DT
    every = int(round(fdt / config.PH0_DT))
    t, a, b = run_pair(np.array(cases), K, config.PH0_ANIM_T, config.PH0_DT, every=every)
    phi = a - b
    rate = phi_rate(phi, np.array(cases)[None, :], K)
    nf = len(t)
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1], hspace=0.35, wspace=0.25,
                          left=0.06, right=0.98, top=0.9, bottom=0.1)
    circ, strip = [], []
    ring = np.linspace(0, 2 * np.pi, 200)
    lo = min(-0.3, rate.min() - 0.2)
    hi = max(0.6, rate.max() + 0.2)
    for j, dw in enumerate(cases):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(np.cos(ring), np.sin(ring), color="0.75", lw=1.5)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(case_title(j, dw, K), fontsize=12)
        arc, = ax.plot([], [], color="0.35", lw=5, alpha=0.35, solid_capstyle="butt")
        l1, = ax.plot([], [], color=C1, lw=1.2)
        l2, = ax.plot([], [], color=C2, lw=1.2)
        d1, = ax.plot([], [], "o", color=C1, ms=15)
        d2, = ax.plot([], [], "o", color=C2, ms=15)
        txt = ax.text(0, -1.3, "", ha="center", va="center", fontsize=10, color="0.25")
        circ.append((arc, l1, l2, d1, d2, txt))
        sx = fig.add_subplot(gs[1, j])
        sx.axhspan(-config.PH0_NEAR_LOCK * K, config.PH0_NEAR_LOCK * K, color=C1, alpha=0.12, lw=0)
        sx.axhline(0, color="0.5", lw=0.8)
        sx.set_xlim(0, t[-1])
        sx.set_ylim(lo, hi)
        sx.set_xlabel("時間 t", fontsize=9)
        if j == 0:
            sx.set_ylabel("ずれの速さ", fontsize=9)
        ln, = sx.plot([], [], color="0.15", lw=1.4)
        strip.append(ln)
    fig.suptitle("青＝身体1、橙＝身体2（同じ円の上を左回りに回る）。灰色の弧が二人のずれ", fontsize=11)

    def frame(k):
        for j in range(len(cases)):
            arc, l1, l2, d1, d2, txt = circ[j]
            p1, p2 = a[k, j], b[k, j]
            w = np.angle(np.exp(1j * (p1 - p2)))         # ずれを (−π, π] に折り返す
            s = np.linspace(p2, p2 + w, 40)
            arc.set_data(1.12 * np.cos(s), 1.12 * np.sin(s))
            l1.set_data([0, np.cos(p1)], [0, np.sin(p1)])
            l2.set_data([0, np.cos(p2)], [0, np.sin(p2)])
            d1.set_data([np.cos(p1)], [np.sin(p1)])
            d2.set_data([np.cos(p2)], [np.sin(p2)])
            laps = int(np.floor((phi[k, j] + np.pi) / (2 * np.pi)))
            txt.set_text(f"t = {t[k]:5.1f}   身体1が追い越した回数 {laps}")
            strip[j].set_data(t[:k + 1], rate[:k + 1, j])
        return []

    anim = FuncAnimation(fig, frame, frames=nf, interval=1000 / config.PH0_ANIM_FPS)
    anim.save(path, writer=PillowWriter(fps=config.PH0_ANIM_FPS), dpi=config.PH0_ANIM_DPI)
    plt.close(fig)
    return nf


def main():
    os.makedirs(RESULTS, exist_ok=True)
    plotting.setup()
    K = config.PH0_K
    log = open(os.path.join(RESULTS, "stage0_log.txt"), "w")

    def say(msg):
        print(msg)
        log.write(msg + "\n")

    t0 = time.time()
    say(f"段階0  K = {K}、ω̄ = {config.PH0_OMEGA_MEAN}、三つの例の Δω = {config.PH0_CASES}、"
        f"刻み {config.PH0_DT}（ルンゲ＝クッタ四次）、φ(0) = {config.PH0_PHI0}")
    t, phi = plot_cases(K, os.path.join(RESULTS, "stage0_cases.png"))
    for j, dw in enumerate(config.PH0_CASES):
        say(f"  {case_title(j, dw, K)}：時間 {config.PH0_T:g} で φ は {phi[-1, j] / (2 * np.pi):.2f} 周ずれた")
    x, rate = plot_beat(K, os.path.join(RESULTS, "stage0_beat.png"), os.path.join(RESULTS, "stage0_beat.csv"))
    t1 = time.time()
    say(f"掃引：Δω/K = {config.PH0_SWEEP_MIN}〜{config.PH0_SWEEP_MAX} の {config.PH0_SWEEP_N} 点、{t1 - t0:.0f} 秒")
    for xv in (0.5, 0.9, 0.99, 1.0, 1.01, 1.05, 1.1, 1.5, 2.0, 2.5):
        i = int(np.argmin(np.abs(x - xv)))
        say(f"  Δω/K = {x[i]:.2f}：平均のずれの速さ / K = {rate[i] / K:.4f}")
    nf = make_anim(K, os.path.join(RESULTS, "stage0_anim.gif"))
    say(f"アニメーション：{nf} コマ、{time.time() - t1:.0f} 秒")
    log.close()


if __name__ == "__main__":
    main()

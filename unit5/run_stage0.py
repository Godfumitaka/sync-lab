"""単元五 段階0：既知の再現（随意運動と世界だけ）。

使い方（sync-lab/unit5 で）:
    python3 run_stage0.py --estimate   短く走らせて、本番の時間とメモリを見積もるだけ（結果は出さない）
    python3 run_stage0.py              本番

一本の走行：config の既定（自分のチャンネル 3 本（遅れ 1, 2, 3）、世界のチャンネル 3 本、雑音 σ = 1）で
            U5_S0_T_SINGLE 歩。度合いと証拠（log₁₀ オッズ）の推移、自分の模型の予測の良さ、学んだ重みを記録する。
掃引一（雑音）：σ を U5_S0_NOISE_SDS に変え、各 U5_S0_RUNS 本を U5_S0_T_NOISE 歩。
掃引二（遅れ）：自分のチャンネル 3 本の遅れをそろえて U5_S0_DELAYS に変え、各 U5_S0_RUNS 本を U5_S0_T_DELAY 歩。
掃引では、各チャンネルが「正しく見分けられたまま最後まで保たれるようになった歩数」を数える
（自分のチャンネルは度合い ≥ U5_DECIDE、そうでないチャンネルは ≤ 1 − U5_DECIDE。ASSUMPTIONS.md の 9）。

出力（results/）
    stage0_single_own.csv      一本の走行：見た感覚の数 n ごとの度合いと log₁₀ オッズ
    stage0_single_r2.csv       一本の走行：時間の区切りごとの、自分の模型の予測の良さ（R²）
    stage0_single_weights.csv  一本の走行：最後の重みの事後平均（チャンネル × 遅れ）
    stage0_single.png
    stage0_noise.csv, stage0_delay.csv, stage0_sweep.png
    stage0_log.txt             計算時間など
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
from selfworld import Agent, Sensorium, run  # noqa: E402

RESULTS = os.path.join(HERE, "results")
N = config.U5_N_CHANNELS
L = config.U5_LAGS
LN10 = np.log(10.0)


def channel_labels():
    out = []
    for i in range(N):
        if config.U5_SELF_GAINS[i] != 0:
            out.append(f"ch{i + 1}: self (delay {config.U5_SELF_DELAYS[i]})")
        else:
            out.append(f"ch{i + 1}: world")
    return out


def single_run():
    rng = np.random.default_rng(config.U5_S0_SEED)
    T = config.U5_S0_T_SINGLE
    rec = np.unique(np.concatenate([[0], np.logspace(0, np.log10(T), config.U5_S0_RECORD_POINTS).astype(int)]))
    sens = Sensorium(1, rng)
    ag = Agent(1)
    out = run(sens, ag, T, rng, record_at=rec, keep_series=True)
    # 予測の良さ：時間の区切りごとに R² = 1 − Σ(s − ŝ)² / Σ s²（ŝ は感覚を見る前の予測）
    edges = np.unique(np.logspace(0, np.log10(T), config.U5_S0_R2_BINS + 1).astype(int))
    s, sh = out["s"][:, 0, :], out["s_hat"][:, 0, :]
    r2 = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        num = ((s[lo:hi] - sh[lo:hi]) ** 2).sum(axis=0)
        den = (s[lo:hi] ** 2).sum(axis=0)
        r2.append(1.0 - num / den)
    out["r2_edges"] = edges
    out["r2"] = np.array(r2)
    return out


def sweep(param_runs, T, seed, make_sens):
    """param_runs：条件ごとの値の並び。make_sens(values_per_run, rng) が Sensorium を返す。"""
    rng = np.random.default_rng(seed)
    R = config.U5_S0_RUNS
    vals = np.repeat(np.asarray(param_runs), R)
    sens = make_sens(vals, rng)
    ag = Agent(len(vals))
    out = run(sens, ag, T, rng)
    return vals, out


def make_noise_sens(vals, rng):
    return Sensorium(len(vals), rng, noise_sd=vals[:, None])


def make_delay_sens(vals, rng):
    is_self = np.asarray(config.U5_SELF_GAINS) != 0
    delays = np.where(is_self[None, :], vals[:, None], np.asarray(config.U5_SELF_DELAYS)[None, :])
    return Sensorium(len(vals), rng, self_delays=delays)


def summarize(vals, out, T):
    """条件ごと・群（自分 / そうでない）ごとに、保たれるようになった歩数の四分位と、T までに保たれた割合。"""
    rows = []
    for v in np.unique(vals):
        sel = vals == v
        for grp, mask in (("self", out["is_self"][sel]), ("not_self", ~out["is_self"][sel])):
            st = out["settle"][sel][mask]
            ok = st >= 0
            frac = ok.mean()
            if ok.any():
                # 保たれなかった走行は「T より大きい」として四分位に入れる（分位点が T を超えれば nan）
                tt = np.where(ok, st, np.inf).astype(float)
                q = np.percentile(tt, [25, 50, 75], method="inverted_cdf")   # 補間しない分位点
                q = np.where(np.isfinite(q), q, np.nan)
            else:
                q = np.array([np.nan] * 3)
            rows.append((v, grp, int(mask.sum()), frac, *q))
    return rows


def write_rows(path, name, rows):
    with open(path, "w") as fh:
        fh.write(f"{name},group,n_channels,frac_settled_by_T,q25,median,q75\n")
        for v, g, n, f, q1, q2, q3 in rows:
            fh.write(f"{v},{g},{n},{f:.4f},{q1:.1f},{q2:.1f},{q3:.1f}\n")


def plot_single(single, path):
    C = config.SERIES_COLORS_6
    labels = channel_labels()
    col = [C[0] if config.U5_SELF_GAINS[i] != 0 else C[1] for i in range(N)]
    ls = ["-", "--", ":"] * 2
    n = single["rec_n"]
    own = single["rec_own"][:, 0, :]
    lo10 = (np.log(config.U5_PRIOR_SELF / (1 - config.U5_PRIOR_SELF)) + single["rec_logbf"][:, 0, :]) / LN10
    x = np.maximum(n, 0.7)    # n = 0（事前）を対数軸の左端に置く

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    a1, a2, a3, a4 = axs.ravel()
    for i in range(N):
        a1.plot(x, own[:, i], color=col[i], ls=ls[i], lw=1.6, label=labels[i])
        a2.plot(x, lo10[:, i], color=col[i], ls=ls[i], lw=1.6, label=labels[i])
    a1.axhline(config.U5_DECIDE, color="0.6", lw=0.8, ls="-.")
    a1.axhline(1 - config.U5_DECIDE, color="0.6", lw=0.8, ls="-.")
    a1.set_ylabel("ownership  P(moved by my commands)")
    a1.set_ylim(-0.02, 1.02)
    a1.legend(frameon=False, fontsize=8)
    a2.set_yscale("symlog", linthresh=1.0)
    a2.axhline(0, color="0.6", lw=0.8)
    a2.set_ylabel("evidence  log10 odds (self : not self)")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("n (sensations seen; leftmost point = prior)")
        ax.grid(color="0.92", lw=0.8)

    e = single["r2_edges"]
    mid = np.sqrt(e[:-1] * e[1:])
    for i in range(N):
        a3.plot(mid, single["r2"][:, i], color=col[i], ls=ls[i], lw=1.6, marker="o", ms=3, label=labels[i])
    best = np.array(config.U5_SELF_GAINS) ** 2 / (np.array(config.U5_SELF_GAINS) ** 2 + config.U5_NOISE_SD ** 2)
    for b in np.unique(best[best > 0]):
        a3.axhline(b, color=C[0], lw=0.8, ls="-.", alpha=0.7)
    a3.text(mid[0], best.max() + 0.03, "best possible for self channels (a² / (a² + σ²))", fontsize=7, color="0.35")
    a3.axhline(0, color="0.6", lw=0.8)
    a3.set_xscale("log")
    a3.set_xlabel("t (window of steps, log-spaced)")
    a3.set_ylabel("prediction quality R² of my model (before seeing)")
    a3.set_ylim(-0.6, 1.0)
    a3.grid(color="0.92", lw=0.8)
    a3.legend(frameon=False, fontsize=7, loc="lower right")

    m = single["m_T"][0]   # (k, N)
    im = a4.imshow(m.T, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    for i in range(N):
        for lag in range(L + 1):
            a4.text(lag, i, f"{m[lag, i]:.2f}", ha="center", va="center", fontsize=7, color="0.1")
    a4.set_xticks(range(L + 1))
    a4.set_xlabel("lag l of efference copy u(t − l)")
    a4.set_yticks(range(N))
    a4.set_yticklabels(labels, fontsize=8)
    a4.set_title(f"learned weights (posterior mean, H_self) at n = {config.U5_S0_T_SINGLE}", fontsize=9)
    fig.colorbar(im, ax=a4, fraction=0.04)
    fig.suptitle(f"stage 0: one run (noise σ = {config.U5_NOISE_SD}, world ρ = {config.U5_WORLD_RHO}, "
                 f"window L = {L})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(noise_rows, delay_rows, path):
    C = config.SERIES_COLORS_6
    fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(15, 4.4))
    for grp, c, lab in (("self", C[0], f"self channels: P ≥ {config.U5_DECIDE} for good"),
                        ("not_self", C[1], f"world channels: P ≤ {1 - config.U5_DECIDE:.2f} for good")):
        rr = [r for r in noise_rows if r[1] == grp]
        v = np.array([r[0] for r in rr])
        q1, q2, q3 = (np.array([r[k] for r in rr]) for k in (4, 5, 6))
        b1.plot(v, q2, color=c, lw=1.8, marker="o", ms=5, label=lab)
        b1.fill_between(v, q1, q3, color=c, alpha=0.18, lw=0)
    b1.set_xscale("log")
    b1.set_yscale("log")
    b1.set_xlabel("noise σ")
    b1.set_ylabel("steps until told apart for good (median, IQR)")
    b1.set_title(f"sweep 1: noise ({config.U5_S0_RUNS} runs each, T = {config.U5_S0_T_NOISE})", fontsize=9)
    b1.legend(frameon=False, fontsize=8)

    for grp, c, lab in (("self", C[0], "self channels"), ("not_self", C[1], "world channels")):
        rr = [r for r in delay_rows if r[1] == grp]
        v = np.array([r[0] for r in rr])
        q1, q2, q3 = (np.array([r[k] for r in rr]) for k in (4, 5, 6))
        f = np.array([r[3] for r in rr])
        b2.plot(v, q2, color=c, lw=1.8, marker="o", ms=5, label=lab)
        b2.fill_between(v, q1, q3, color=c, alpha=0.18, lw=0)
        b3.plot(v, f, color=c, lw=1.8, marker="o", ms=5, label=lab)
    for ax in (b2, b3):
        ax.axvline(L + 0.5, color="0.5", lw=0.8, ls="-.")
        ax.set_xlabel("delay d of the self channels (steps)")
        ax.set_xticks(config.U5_S0_DELAYS)
    b2.text(L + 0.6, 0.98, f"window edge\n(L = {L})", transform=b2.get_xaxis_transform(), fontsize=7,
            color="0.35", va="top")
    b2.set_yscale("log")
    b2.set_ylabel("steps until told apart for good (median, IQR)")
    b2.set_title(f"sweep 2: delay (σ = {config.U5_NOISE_SD}, T = {config.U5_S0_T_DELAY})", fontsize=9)
    b2.legend(frameon=False, fontsize=8)
    b3.set_ylim(-0.03, 1.03)
    b3.set_ylabel(f"fraction told apart correctly at T = {config.U5_S0_T_DELAY}")
    b3.legend(frameon=False, fontsize=8)
    for ax in (b1, b2, b3):
        ax.grid(color="0.92", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def estimate():
    """短く走らせて、本番の時間とメモリを見積もる。結果の統計は出さない。"""
    R = config.U5_S0_RUNS
    jobs = [("一本の走行", 1, config.U5_S0_T_SINGLE),
            ("掃引一（雑音）", R * len(config.U5_S0_NOISE_SDS), config.U5_S0_T_NOISE),
            ("掃引二（遅れ）", R * len(config.U5_S0_DELAYS), config.U5_S0_T_DELAY)]
    total = 0.0
    rng = np.random.default_rng(0)
    for name, n_runs, T in jobs:
        steps = 300
        sens = Sensorium(n_runs, rng)
        ag = Agent(n_runs)
        t0 = time.time()
        run(sens, ag, steps, rng)
        per = (time.time() - t0) / steps
        total += per * T
        k = L + 1
        mem = n_runs * (k * k + k * N + 6 * N) * 8 * 4 / 1e6
        print(f"  {name}：{n_runs} 本 × {T} 歩、一歩 {per * 1e3:.2f} ms → 約 {per * T:.0f} 秒、配列 約 {mem:.0f} MB")
    print(f"  合計 約 {total / 60:.1f} 分")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    t0 = time.time()
    single = single_run()
    t1 = time.time()
    nv, nout = sweep(config.U5_S0_NOISE_SDS, config.U5_S0_T_NOISE, config.U5_S0_SWEEP_SEED, make_noise_sens)
    t2 = time.time()
    dv, dout = sweep(config.U5_S0_DELAYS, config.U5_S0_T_DELAY, config.U5_S0_SWEEP_SEED + 1, make_delay_sens)
    t3 = time.time()

    # 一本の走行
    n = single["rec_n"]
    own = single["rec_own"][:, 0, :]
    lo10 = (np.log(config.U5_PRIOR_SELF / (1 - config.U5_PRIOR_SELF)) + single["rec_logbf"][:, 0, :]) / LN10
    hdr = "n," + ",".join(f"own_ch{i + 1}" for i in range(N)) + "," + ",".join(f"log10odds_ch{i + 1}" for i in range(N))
    np.savetxt(os.path.join(RESULTS, "stage0_single_own.csv"), np.column_stack([n, own, lo10]),
               delimiter=",", comments="", header=hdr, fmt=["%d"] + ["%.6g"] * (2 * N))
    e = single["r2_edges"]
    np.savetxt(os.path.join(RESULTS, "stage0_single_r2.csv"), np.column_stack([e[:-1], e[1:], single["r2"]]),
               delimiter=",", comments="", header="t_from,t_to," + ",".join(f"r2_ch{i + 1}" for i in range(N)),
               fmt=["%d", "%d"] + ["%.5f"] * N)
    np.savetxt(os.path.join(RESULTS, "stage0_single_weights.csv"),
               np.column_stack([np.arange(1, N + 1), single["m_T"][0].T]), delimiter=",", comments="",
               header="channel," + ",".join(f"w_lag{l}" for l in range(L + 1)), fmt=["%d"] + ["%.5f"] * (L + 1))
    plot_single(single, os.path.join(RESULTS, "stage0_single.png"))

    # 掃引
    noise_rows = summarize(nv, nout, config.U5_S0_T_NOISE)
    delay_rows = summarize(dv, dout, config.U5_S0_T_DELAY)
    write_rows(os.path.join(RESULTS, "stage0_noise.csv"), "noise_sd", noise_rows)
    write_rows(os.path.join(RESULTS, "stage0_delay.csv"), "delay", delay_rows)
    plot_sweep(noise_rows, delay_rows, os.path.join(RESULTS, "stage0_sweep.png"))

    log = [f"一本の走行 {t1 - t0:.1f} s、掃引一（雑音）{t2 - t1:.1f} s、掃引二（遅れ）{t3 - t2:.1f} s",
           "", "一本の走行：最初に正しく見分けた n ／ 最後まで保たれるようになった n"]
    for i, lab in enumerate(channel_labels()):
        log.append(f"  {lab:<24s} {single['first'][0, i]:>6d} ／ {single['settle'][0, i]:>6d}")
    log += ["", "掃引一（雑音）：σ、群、チャンネル数、T までに保たれた割合、四分位（25%、中央値、75%）"]
    log += [f"  {r[0]:>5}  {r[1]:<8s} {r[2]:>4d}  {r[3]:.3f}  {r[4]:.0f} {r[5]:.0f} {r[6]:.0f}" for r in noise_rows]
    log += ["", "掃引二（遅れ）：d、群、チャンネル数、T までに保たれた割合、四分位（25%、中央値、75%）"]
    log += [f"  {r[0]:>5}  {r[1]:<8s} {r[2]:>4d}  {r[3]:.3f}  {r[4]:.0f} {r[5]:.0f} {r[6]:.0f}" for r in delay_rows]
    text = "\n".join(log)
    with open(os.path.join(RESULTS, "stage0_log.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\n保存：{RESULTS}/stage0_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    if ap.parse_args().estimate:
        estimate()
    else:
        main()

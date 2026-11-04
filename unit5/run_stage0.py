"""単元五 段階0：既知の再現（随意運動と世界だけ）。

使い方（sync-lab/unit5 で）:
    python3 run_stage0.py

主な条件（config の U5_* の世界）：U5_S0_RUNS 本を U5_S0_T ステップ。
    チャンネルごとの P(自分)、順モデルによる予測の良さ R²、自他の区別（log BF：M2「自分と世界」対 M1「自分だけ」）
掃引：三本の自分のチャンネルの遅れ d（U5_S0_SWEEP_DELAYS）× 雑音 σ（U5_S0_SWEEP_NOISES）、各 U5_S0_SWEEP_RUNS 本。
    「見分けた時刻」（ASSUMPTIONS.md の 12）

出力（results/）
    stage0_main.png, stage0_main.csv     主な条件の推移
    stage0_detect.csv                     主な条件の、走行ごと・チャンネルごとの見分けた時刻
    stage0_sweep.png, stage0_sweep.csv    遅れ × 雑音の掃引
    stage0_log.txt                        値と計算時間
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
from selfworld import (Agent, default_env, first_settled, pad_for, prediction_r2,  # noqa: E402
                       record_times, simulate)

RESULTS = os.path.join(HERE, "results")
CHUNK = 25            # 一つのプロセスに渡す走行の数


def _chunk(args):
    """一かたまりの走行を、世界の生成から信念まで。"""
    seed, n_runs, T, env, want_theta = args
    rng = np.random.default_rng(seed)
    ag = Agent(env["n_ch"])
    pad = pad_for(env, ag.L)
    rec = record_times(T)
    u, s = simulate(rng, n_runs, T, env, pad)
    b = ag.run(u, s, pad, rec, want_theta=want_theta)
    out = {k: b[k] for k in ("p_self", "log_bf", "p_two", "phi2", "grad_max")}
    if want_theta:
        out["r2"], _ = prediction_r2(b["theta"], env, ag.L)
    return out


def run_condition(pool, env, n_runs, T, seed, want_theta):
    seeds = np.random.SeedSequence(seed).spawn((n_runs + CHUNK - 1) // CHUNK)
    jobs = [(sq, min(CHUNK, n_runs - k * CHUNK), T, env, want_theta) for k, sq in enumerate(seeds)]
    parts = list(pool.map(_chunk, jobs))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def settle_times(p_self, env, rec):
    """自分のチャンネルは P ≥ 閾値、そうでないものは P ≤ 1 − 閾値 に最後までとどまり始めた時刻。(R, N)"""
    thr = config.U5_DETECT_P
    is_self = np.isin(np.arange(env["n_ch"]), env["self_ch"])
    p = np.moveaxis(p_self, 1, 2)                      # (R, N, n_rec)
    t_self = first_settled(p, rec, thr, above=True)
    t_not = first_settled(p, rec, 1 - thr, above=False)
    return np.where(is_self[None, :], t_self, t_not), is_self


def qstats(t, T):
    """見分けた時刻の中央値と四分位（T までに見分けられなかったものは ∞ として数える）と、見分けた割合。"""
    x = np.where(np.isnan(t), np.inf, t).ravel()
    q = np.quantile(x, [0.25, 0.5, 0.75], method="inverted_cdf")   # ∞ をまたいで補間しない
    return q, np.isfinite(x).mean()


def style():
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35",
                                "axes.spines.top": False, "axes.spines.right": False})


def channel_label(env, i):
    if i in env["self_ch"]:
        return f"ch{i}: self (d = {env['delays'][env['self_ch'].index(i)]})"
    return f"ch{i}: world only"


def plot_main(env, rec, res, r2_ceil, path):
    style()
    cols = config.SERIES_COLORS_6
    N = env["n_ch"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    (a1, a2), (a3, a4) = axs
    for i in range(N):
        ls = "-" if i in env["self_ch"] else "--"
        a1.plot(rec, res["p_self"][0, :, i], color=cols[i], lw=2, ls=ls, label=channel_label(env, i))
        med = np.median(res["p_self"][:, :, i], axis=0)
        lo, hi = np.quantile(res["p_self"][:, :, i], [0.1, 0.9], axis=0)
        a2.fill_between(rec, lo, hi, color=cols[i], alpha=0.15, lw=0)
        a2.plot(rec, med, color=cols[i], lw=2, ls=ls)
        a3.plot(rec, np.median(res["r2"][:, :, i], axis=0), color=cols[i], lw=2, ls=ls)
        if i in env["self_ch"]:
            a3.axhline(r2_ceil[i], color=cols[i], lw=1, ls=":")
    thr = config.U5_DETECT_P
    for ax in (a1, a2):
        ax.axhline(thr, color="0.6", lw=0.8, ls=":")
        ax.axhline(1 - thr, color="0.6", lw=0.8, ls=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(self | data)")
    a1.set_title("one run", fontsize=10, loc="left")
    a2.set_title(f"{res['p_self'].shape[0]} runs: median and 10–90%", fontsize=10, loc="left")
    a1.legend(frameon=False, fontsize=8, loc="center right")
    a3.set_ylabel("R² of the forward model (median)")
    a3.set_title("prediction from own commands (dotted: ceiling)", fontsize=10, loc="left")
    lb = res["log_bf"]
    a4.fill_between(rec, *np.quantile(lb, [0.1, 0.9], axis=0), color=cols[0], alpha=0.15, lw=0)
    a4.plot(rec, np.median(lb, axis=0), color=cols[0], lw=2)
    a4.axhline(0, color="0.5", lw=0.8)
    a4.set_yscale("symlog", linthresh=1.0)
    a4.set_ylabel("log BF  (self + world  vs  self only)")
    a4.set_title("self / world distinction: median and 10–90%", fontsize=10, loc="left")
    for ax in axs.ravel():
        ax.set_xscale("log")
        ax.grid(color="0.92", lw=0.8)
    for ax in (a3, a4):
        ax.set_xlabel("n (steps of motor babbling)")
    fig.suptitle(f"Stage 0: c = {env['world_amp']}, τ = {env['world_tau']}, σ = {env['noise_std']}, "
                 f"L = {config.U5_LAGS}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(rows, T, path):
    style()
    cols = config.SERIES_COLORS_6
    delays = config.U5_S0_SWEEP_DELAYS
    noises = config.U5_S0_SWEEP_NOISES
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, sg in enumerate(noises):
        r = [row for row in rows if row["noise"] == sg]
        d = np.array([row["delay"] for row in r])
        for ax, key in ((a1, "self"), (a3, "not")):
            med = np.array([row[f"{key}_q50"] for row in r])
            lo = np.array([row[f"{key}_q25"] for row in r])
            hi = np.array([row[f"{key}_q75"] for row in r])
            fin = np.isfinite(med)
            ax.errorbar(d[fin] + (k - 1.5) * 0.08, med[fin],
                        yerr=[med[fin] - lo[fin], np.where(np.isfinite(hi[fin]), hi[fin], T) - med[fin]],
                        color=cols[k], lw=2, marker="o", ms=6, capsize=3, label=f"σ = {sg}")
        a2.plot(d, [row["self_frac"] for row in r], color=cols[k], lw=2, marker="o", ms=6, label=f"σ = {sg}")
    for ax in (a1, a2, a3):
        ax.axvline(config.U5_LAGS + 0.5, color="0.5", lw=1, ls=":")
        ax.set_xlabel("delay d of the self channels (steps)")
        ax.set_xticks(delays)
        ax.grid(color="0.92", lw=0.8)
    a1.text(config.U5_LAGS + 0.6, 0.98, "end of the agent's\nwindow (L)", transform=a1.get_xaxis_transform(),
            fontsize=8, color="0.4", va="top")
    a1.set_yscale("log")
    a3.set_yscale("log")
    a1.set_ylabel(f"steps until P(self) ≥ {config.U5_DETECT_P} for good")
    a1.set_title("self channels: median and 25–75%", fontsize=10, loc="left")
    a2.set_ylabel(f"fraction detected by T = {T}")
    a2.set_ylim(-0.03, 1.03)
    a2.set_title("self channels", fontsize=10, loc="left")
    a3.set_ylabel(f"steps until P(self) ≤ {1 - config.U5_DETECT_P:.2f} for good")
    a3.set_title("world-only channels: median and 25–75%", fontsize=10, loc="left")
    a1.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Stage 0 sweep: c = {config.U5_WORLD_AMP}, τ = {config.U5_WORLD_TAU}, L = {config.U5_LAGS}, "
                 f"{config.U5_S0_SWEEP_RUNS} runs × 3 channels each", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, "stage0_log.txt"), "w")

    def say(msg):
        print(msg)
        log.write(msg + "\n")

    T = config.U5_S0_T
    rec = record_times(T)
    env = default_env()
    say(f"段階0  世界：{env}")
    say(f"エージェント：L = {config.U5_LAGS}、θ の事前の分散 {config.U5_THETA_PRIOR_VAR}、"
        f"φ の格子 {config.U5_PHI_GRID_N} 点、log10 v ~ N({config.U5_LOG10_VAR_PRIOR_MEAN}, "
        f"{config.U5_LOG10_VAR_PRIOR_SD}²)、事前 P(self) = {config.U5_PRIOR_SELF}、P(M2) = {config.U5_PRIOR_TWO}")
    t0 = time.time()
    with ProcessPoolExecutor(config.U5_WORKERS) as pool:
        res = run_condition(pool, env, config.U5_S0_RUNS, T, config.U5_S0_SEED, want_theta=True)
        t1 = time.time()
        say(f"主な条件：{config.U5_S0_RUNS} 本 × {T} ステップ、{t1 - t0:.0f} 秒")
        rows = []
        for d in config.U5_S0_SWEEP_DELAYS:
            for k, sg in enumerate(config.U5_S0_SWEEP_NOISES):
                e = dict(env, delays=[d] * len(env["self_ch"]), noise_std=sg)
                seed = config.U5_S0_SWEEP_SEED * 1000 + 10 * config.U5_S0_SWEEP_DELAYS.index(d) + k
                r = run_condition(pool, e, config.U5_S0_SWEEP_RUNS, T, seed, want_theta=False)
                ts, is_self = settle_times(r["p_self"], e, rec)
                qs, fs = qstats(ts[:, is_self], T)
                qn, fn = qstats(ts[:, ~is_self], T)
                rows.append(dict(delay=d, noise=sg, self_q25=qs[0], self_q50=qs[1], self_q75=qs[2], self_frac=fs,
                                 not_q25=qn[0], not_q50=qn[1], not_q75=qn[2], not_frac=fn,
                                 grad_max=r["grad_max"].max()))
        say(f"掃引：{len(rows)} 条件 × {config.U5_S0_SWEEP_RUNS} 本、{time.time() - t1:.0f} 秒")

    # 主な条件の保存
    _, r2_ceil = prediction_r2(np.zeros((env["n_ch"], config.U5_LAGS + 1)), env, config.U5_LAGS)
    cols = ["n"]
    data = [rec]
    for i in range(env["n_ch"]):
        q = np.quantile(res["p_self"][:, :, i], [0.1, 0.5, 0.9], axis=0)
        cols += [f"p_self_ch{i}_q10", f"p_self_ch{i}_med", f"p_self_ch{i}_q90", f"r2_ch{i}_med"]
        data += [q[0], q[1], q[2], np.median(res["r2"][:, :, i], axis=0)]
    qb = np.quantile(res["log_bf"], [0.1, 0.5, 0.9], axis=0)
    cols += ["log_bf_q10", "log_bf_med", "log_bf_q90", "p_two_med"]
    data += [qb[0], qb[1], qb[2], np.median(res["p_two"], axis=0)]
    np.savetxt(os.path.join(RESULTS, "stage0_main.csv"), np.column_stack(data), delimiter=",",
               header=",".join(cols), comments="", fmt="%.6g")
    ts, is_self = settle_times(res["p_self"], env, rec)
    with open(os.path.join(RESULTS, "stage0_detect.csv"), "w") as fh:
        fh.write("run,channel,is_self,delay,t_settled\n")
        for r_ in range(ts.shape[0]):
            for i in range(env["n_ch"]):
                d = env["delays"][env["self_ch"].index(i)] if is_self[i] else -1
                fh.write(f"{r_},{i},{int(is_self[i])},{d},{ts[r_, i]:.0f}\n")
    plot_main(env, rec, res, r2_ceil, os.path.join(RESULTS, "stage0_main.png"))

    # 掃引の保存
    keys = list(rows[0].keys())
    with open(os.path.join(RESULTS, "stage0_sweep.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(f"{row[k]:.6g}" for k in keys) + "\n")
    plot_sweep(rows, T, os.path.join(RESULTS, "stage0_sweep.png"))

    say("\n主な条件：見分けた時刻（中央値 [25%, 75%]、T までに見分けた割合）")
    for i in range(env["n_ch"]):
        q, f = qstats(ts[:, i], T)
        say(f"  {channel_label(env, i):<24s} {q[1]:>7.0f} [{q[0]:.0f}, {q[2]:.0f}]  {f:.2f}")
    say(f"  ニュートン法の頂上での傾きの最大：{res['grad_max'].max():.1e}")
    say("\n掃引：自分のチャンネル（中央値 [25%, 75%]、割合） ／ 世界だけのチャンネル")
    for row in rows:
        say(f"  d = {row['delay']:>2d}, σ = {row['noise']:<4}: "
            f"{row['self_q50']:>7.0f} [{row['self_q25']:.0f}, {row['self_q75']:.0f}] {row['self_frac']:.2f}  ／  "
            f"{row['not_q50']:>5.0f} [{row['not_q25']:.0f}, {row['not_q75']:.0f}] {row['not_frac']:.2f}")
    say(f"  ニュートン法の頂上での傾きの最大：{max(r['grad_max'] for r in rows):.1e}")
    log.close()


if __name__ == "__main__":
    main()

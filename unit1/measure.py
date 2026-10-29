"""段階二：雑音なしで、ω のすぐ下の a について一周の時間 T を測る。

使い方（sync-lab/unit1 で）:
    python measure.py

一周の時間 T：θ = STAGE2_THETA0 から出発して θ0 + 2π に達するまでの時間。
越えたステップの前後を直線でつないで、達した時刻を dt より細かく読む。
すべての ε を numpy の配列でまとめて進める。dt ごとに一度ずつ走らせる。

出力（results/）
    stage2_T.csv      dt, a, eps, T, steps
    stage2_fit.csv    dt ごとの log T = slope·log ε + intercept の最小二乗（全点と、ε の小さい半分）
    stage2_T.png      T と ε の両対数図
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
from model import f  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def period(a, dt):
    """各 a について一周の時間 T とステップ数を返す。"""
    a = np.asarray(a, dtype=float)
    theta = np.full(a.shape, config.STAGE2_THETA0)
    target = config.STAGE2_THETA0 + 2.0 * np.pi
    T = np.full(a.shape, np.nan)
    steps = np.zeros(a.shape, dtype=np.int64)
    alive = np.ones(a.shape, dtype=bool)
    n = 0
    t = 0.0
    while alive.any() and t < config.STAGE2_T_MAX:
        old = theta[alive]
        new = old + f(old, config.OMEGA, a[alive]) * dt
        theta[alive] = new
        n += 1
        t = n * dt
        done = new >= target
        if done.any():
            idx = np.flatnonzero(alive)[done]
            frac = (target - old[done]) / (new[done] - old[done])
            T[idx] = (n - 1 + frac) * dt
            steps[idx] = n
            alive[idx] = False
    return T, steps


def fit(eps, T):
    slope, intercept = np.polyfit(np.log10(eps), np.log10(T), 1)
    return slope, intercept


def main():
    os.makedirs(RESULTS, exist_ok=True)
    eps = np.logspace(np.log10(config.STAGE2_EPS_MAX), np.log10(config.STAGE2_EPS_MIN),
                      config.STAGE2_N_EPS)
    a = config.OMEGA - eps

    rows, fits, res = [], [], {}
    for dt in config.STAGE2_DTS:
        t0 = time.time()
        T, steps = period(a, dt)
        res[dt] = T
        print(f"dt = {dt:g}: {time.time() - t0:.1f} s, 打ち切り {np.isnan(T).sum()} 個")
        rows += [(dt, ai, ei, Ti, si) for ai, ei, Ti, si in zip(a, eps, T, steps)]
        ok = ~np.isnan(T)
        small = ok & (eps <= np.median(eps))
        for name, m in (("all", ok), ("small_half", small)):
            s, c = fit(eps[m], T[m])
            fits.append((dt, name, eps[m].min(), eps[m].max(), m.sum(), s, c))

    np.savetxt(os.path.join(RESULTS, "stage2_T.csv"), np.array(rows), delimiter=",",
               header="dt,a,eps,T,steps", comments="", fmt=["%g", "%.10g", "%.6g", "%.10g", "%d"])
    with open(os.path.join(RESULTS, "stage2_fit.csv"), "w") as fh:
        fh.write("dt,range,eps_min,eps_max,n,slope,intercept\n")
        for r in fits:
            fh.write("%g,%s,%.6g,%.6g,%d,%.6f,%.6f\n" % r)

    # 報告：傾き、dt による違い
    print("\n傾き（log T を log ε に最小二乗）")
    for dt, name, e0, e1, n, s, c in fits:
        print(f"  dt = {dt:<6g} {name:<10s} ε ∈ [{e0:.1e}, {e1:.1e}]  n = {n:2d}  slope = {s:+.4f}")
    ref = res[min(config.STAGE2_DTS)]
    print(f"\nT の dt による違い（最小の dt = {min(config.STAGE2_DTS):g} との相対差の最大）")
    for dt in config.STAGE2_DTS:
        if dt != min(config.STAGE2_DTS):
            print(f"  dt = {dt:g}: {np.nanmax(np.abs(res[dt] / ref - 1)):.2e}")

    # 図
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    fig, ax = plt.subplots(figsize=(6.5, 5))
    markers = ["o", "s", "^"]
    for i, dt in enumerate(config.STAGE2_DTS):
        ax.plot(eps, res[dt], markers[i % 3], ms=8 - 1.5 * i, mfc="none", mew=1.5,
                color=config.SERIES_COLORS[i % 3], label=f"dt = {dt:g}", zorder=3)
    s, c = [(r[5], r[6]) for r in fits if r[0] == min(config.STAGE2_DTS) and r[1] == "all"][0]
    xx = np.array([eps.min(), eps.max()])
    ax.plot(xx, 10 ** (c + s * np.log10(xx)), color="0.3", lw=1, ls="--",
            label=f"fit (dt = {min(config.STAGE2_DTS):g}, all): slope {s:+.3f}", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ε = ω − a")
    ax.set_ylabel("T  (one revolution)")
    ax.grid(which="major", color="0.9", lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "stage2_T.png"), dpi=150)
    print(f"\n保存：{RESULTS}/stage2_T.csv, stage2_fit.csv, stage2_T.png")


if __name__ == "__main__":
    main()

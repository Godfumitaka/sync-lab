"""単元三：相手（双方向）と影（一方向）で、滑るまでの平均時間を測り、単元一の予想と比べる。

使い方（sync-lab/unit3 で）:
    python3 measure_pair.py            本番
    python3 measure_pair.py --pilot    軌道数を減らした試し走り（所要時間の見積もり用。保存しない）

位相の差の式に縮めずに、二つの振動子そのもの（pair.py の velocities と、二人それぞれの独立な雑音）を動かす。
出発点：二人の位相の差 φ = θ₁ − θ₂ を、雑音なしのときのそろった値 φ* に置く（φ* は数値で求める）。
滑り：φ が φ* から ±2π 離れること。打ち切りは U3M_T_MAX。

比べる予想（利用者が単元一から組み立てたもの）
    φ̇ = Δω − a sin φ、a = 2K（相手）、a = K（影）、φ に入る雑音は 2D
    平均時間 ∝ exp(ΔV / 2D)、ΔV = 2Δω(tan ψ − ψ)、cos ψ = Δω / a
したがって、ln(平均時間) を 1/D に対して描いた傾きの予想は ΔV / 2。

出力（results/）
    pair_mean.csv   system, dt, D, n, n_done, n_back, mean_done
    pair_fit.csv    system, dt, 使った点の数, 傾き, 予想の傾き ΔV/2
    pair_mean.png   ln(平均時間) と 1/D
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
from pair import noise_step, velocities  # noqa: E402

RESULTS = os.path.join(HERE, "results")
TWO_PI = 2.0 * np.pi
SYSTEMS = [("partner", True), ("shadow", False)]


def locked_phase(dw, K, mutual):
    """雑音なしのときのそろった位相の差 φ*（ゆっくり時間発展させて落ち着いた値）。"""
    w1, w2 = config.U3_OMEGA_MEAN + dw / 2, config.U3_OMEGA_MEAN - dw / 2
    t1, t2 = 0.0, 0.0
    for _ in range(200000):
        v1, v2 = velocities(t1, t2, w1, w2, K, mutual)
        t1 += v1 * 0.01
        t2 += v2 * 0.01
    return (t1 - t2 + np.pi) % TWO_PI - np.pi


def predicted_barrier(dw, a):
    psi = np.arccos(dw / a)
    return 2.0 * dw * (np.tan(psi) - psi)


def slip_times(mutual, phi0, D_list, n_traj, dt, rng, t_max):
    dw, K = config.U3M_DW, config.U3M_K
    w1, w2 = config.U3_OMEGA_MEAN + dw / 2, config.U3_OMEGA_MEAN - dw / 2
    n_cells = len(D_list)
    cell = np.repeat(np.arange(n_cells), n_traj)
    slot = np.tile(np.arange(n_traj), n_cells)
    D = np.asarray(D_list, float)[cell]
    th1 = np.full(cell.size, phi0)
    th2 = np.zeros(cell.size)
    times = np.full((n_cells, n_traj), np.nan)
    back = np.zeros((n_cells, n_traj), dtype=bool)
    for n in range(1, int(round(t_max / dt)) + 1):
        v1, v2 = velocities(th1, th2, w1, w2, K, mutual)
        th1 += v1 * dt + noise_step(D, dt, rng, size=th1.size)
        th2 += v2 * dt + noise_step(D, dt, rng, size=th2.size)
        disp = (th1 - th2) - phi0
        done = np.abs(disp) >= TWO_PI
        if done.any():
            times[cell[done], slot[done]] = n * dt
            back[cell[done], slot[done]] = disp[done] < 0
            keep = ~done
            cell, slot, D, th1, th2 = cell[keep], slot[keep], D[keep], th1[keep], th2[keep]
            if th1.size == 0:
                break
    return times, back


def main(pilot=False):
    n_traj = config.U3M_N_TRAJ // (20 if pilot else 1)
    Ds = np.logspace(config.U3M_LOG10_D_MIN, config.U3M_LOG10_D_MAX, config.U3M_N_D)
    rng = np.random.default_rng(config.U3M_SEED)
    dw, K = config.U3M_DW, config.U3M_K

    res = {}
    for name, mutual in SYSTEMS:
        phi0 = locked_phase(dw, K, mutual)
        for dt in config.U3M_DTS:
            t0 = time.time()
            res[(name, dt)] = slip_times(mutual, phi0, Ds, n_traj, dt, rng, config.U3M_T_MAX)
            print(f"{name}, dt = {dt:g}: {time.time() - t0:.1f} s（φ* = {phi0:.4f}）")
    if pilot:
        return

    os.makedirs(RESULTS, exist_ok=True)
    rows, fits = [], []
    for (name, dt), (times, back) in res.items():
        n = times.shape[1]
        nd = (~np.isnan(times)).sum(axis=1)
        md = np.where(nd > 0, np.nansum(times, axis=1) / np.maximum(nd, 1), np.nan)
        for D, k, b, m in zip(Ds, nd, back.sum(axis=1), md):
            rows.append((name, dt, D, n, k, b, m))
        use = (nd / n >= config.U3M_DONE_FRAC) & (md >= config.U3M_MIN_MEAN)
        a = 2 * K if name == "partner" else K
        pred = predicted_barrier(dw, a) / 2
        if use.sum() >= 2:
            s, c = np.polyfit(1 / Ds[use], np.log(md[use]), 1)
        else:
            s, c = np.nan, np.nan
        fits.append((name, dt, int(use.sum()), s, c, pred))

    with open(os.path.join(RESULTS, "pair_mean.csv"), "w") as fh:
        fh.write("system,dt,D,n,n_done,n_back,mean_done\n")
        for r in rows:
            fh.write("%s,%g,%.6g,%d,%d,%d,%.6g\n" % r)
    with open(os.path.join(RESULTS, "pair_fit.csv"), "w") as fh:
        fh.write("system,dt,n_points,slope,intercept,predicted_slope\n")
        for r in fits:
            fh.write("%s,%g,%d,%.6g,%.6g,%.6g\n" % r)

    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {0.05: "o", 0.01: "s"}
    for i, (name, _) in enumerate(SYSTEMS):
        col = config.SERIES_COLORS[i]
        for dt in config.U3M_DTS:
            rr = [r for r in rows if r[0] == name and r[1] == dt]
            D = np.array([r[2] for r in rr])
            ok = np.array([r[4] / r[3] >= config.U3M_DONE_FRAC for r in rr])
            m = np.array([r[6] for r in rr])
            ax.plot(1 / D[ok], m[ok], markers.get(dt, "o"), mfc="none", mew=1.5, ms=7, color=col,
                    label=f"{name}, dt = {dt:g}")
        f = [x for x in fits if x[0] == name and x[1] == min(config.U3M_DTS)][0]
        if not np.isnan(f[3]):
            xx = np.linspace(1 / Ds.max(), 1 / Ds.min(), 50)
            ax.plot(xx, np.exp(f[4] + f[3] * xx), color=col, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(1, 3 * config.U3M_T_MAX)
    ax.set_xlabel("1/D")
    ax.set_ylabel("mean time to slip")
    ax.set_title(f"Δω = {dw:g}, K = {K:g}  (dashed: fit on points with mean ≥ {config.U3M_MIN_MEAN:g})",
                 fontsize=9)
    ax.grid(color="0.92", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "pair_mean.png"), dpi=150)

    print("\n平均時間（全軌道のうち滑り終えた割合が 99% 以上の点。それ以外は「—」）")
    print("  D          " + "  ".join(f"{n}(dt={dt:g})" for n, _ in SYSTEMS for dt in config.U3M_DTS))
    for D in Ds:
        cells = []
        for name, _ in SYSTEMS:
            for dt in config.U3M_DTS:
                r = [x for x in rows if x[0] == name and x[1] == dt and x[2] == D][0]
                cells.append(f"{r[6]:>10.1f}" if r[4] / r[3] >= config.U3M_DONE_FRAC else f"{'—':>10s}")
        print(f"  {D:<9.4g}" + "  ".join(f"{c:>14s}" for c in cells))
    print("\nln(平均時間) と 1/D の傾き（平均時間 ≥ %g の点で当てはめ）と、予想 ΔV/2" % config.U3M_MIN_MEAN)
    for name, dt, k, s, c, pred in fits:
        print(f"  {name:<8s} dt = {dt:<5g} 点 {k}  傾き {s:.4f}  予想 {pred:.4f}")
    nb = sum(int(r[5]) for r in rows)
    print(f"\n後ろ向きの滑りの総数：{nb}")
    print(f"保存：{RESULTS}/pair_mean.csv, pair_fit.csv, pair_mean.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    main(pilot=ap.parse_args().pilot)

"""単元四 寄り道：合図を無視する状態の近くで、状態 2 の玉の割合 y が y ∝ (玉の数)^c で減るときの指数 c を確かめる。

二つの読みを比べる。
    ノートの読み：c = λ = 1 − 2π（時間 t = log M とした場合）
    補正の読み  ：c = λ/π       （一回ごとの時間 Δt = 1/M、玉は勝ったときだけ増えるので log M ≈ ψ t ≈ π t）
λ = (1−π)Σ q²/a − π は q = a のとき 1 − 2π。

模型は signaling.py と同じ規則。受け手の壺は送り手と同じ数で始めるので、二人の壺はずっと一致し、
四つの数 C[i][x]（状態 i、合図 x）だけで追える：
    送り手が状態 i で合図 x を送る確率 = C[i][x] / Σ_x C[i][x]
    受け手が合図 x を見て行為 i を選ぶ確率 = C[i][x] / Σ_i C[i][x]
    勝ったら C[i][x] += 1
初期：玉の総数 M0、状態 1 は a、状態 2 は y0·M0 個を q = a で分ける（数は config.py の U4_PX_*）。

出力（results/）：pooling_exponent.csv（記録した時刻ごとの各走行の n, M, y, q_A）、pooling_exponent_fit.csv、pooling_exponent.png
使い方（sync-lab/unit4 で）:  python3 pooling_exponent.py   （--pilot で回数を 1/100 にした試し走り）
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

RESULTS = os.path.join(HERE, "results")


def run(pi, n_steps, rng):
    R = config.U4_PX_RUNS
    a = np.asarray(config.U4_PX_A, float)
    M0, y0 = config.U4_PX_M0, config.U4_PX_Y0
    C = np.empty((R, 2, 2))
    C[:, 0, :] = (1 - y0) * M0 * a
    C[:, 1, :] = y0 * M0 * a
    r = np.arange(R)
    rec = set(np.unique(np.logspace(0, np.log10(n_steps), config.U4_PX_RECORD_POINTS).astype(int)))
    out = []
    for n in range(1, n_steps + 1):
        u1, u2, u3 = rng.random((3, R))
        state = (u1 >= pi).astype(int)                         # 0：状態 1（確率 π）、1：状態 2
        row = C[r, state]                                      # 送り手：状態の壺
        signal = (u2 * row.sum(axis=1) >= row[:, 0]).astype(int)
        col = C[r, :, signal]                                  # 受け手：合図の壺（行為 = 状態の添字）
        act = (u3 * col.sum(axis=1) >= col[:, 0]).astype(int)
        win = act == state
        C[r[win], state[win], signal[win]] += 1.0
        if n in rec:
            M = C.sum(axis=(1, 2))
            y = C[:, 1, :].sum(axis=1) / M
            qA = C[:, 1, 0] / C[:, 1, :].sum(axis=1)
            for k in range(R):
                out.append((pi, n, k, M[k], y[k], qA[k]))
    return np.array(out)


def main(pilot=False):
    os.makedirs(RESULTS, exist_ok=True)
    n_steps = config.U4_PX_N // (100 if pilot else 1)
    rng = np.random.default_rng(config.U4_PX_SEED)
    data = []
    for pi in config.U4_PX_PIS:
        t0 = time.time()
        data.append(run(pi, n_steps, rng))
        print(f"π = {pi}: {time.time() - t0:.1f} s（{n_steps} 回）", flush=True)
    if pilot:
        return
    D = np.vstack(data)
    np.savetxt(os.path.join(RESULTS, "pooling_exponent.csv"), D, delimiter=",", comments="",
               header="pi,n,run,M,y,q_A", fmt=["%g", "%d", "%d", "%.1f", "%.6g", "%.6g"])

    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    fig, axes = plt.subplots(2, len(config.U4_PX_PIS), figsize=(11, 8), squeeze=False)
    fits = []
    M0 = config.U4_PX_M0
    for j, pi in enumerate(config.U4_PX_PIS):
        d = D[D[:, 0] == pi]
        ns = np.unique(d[:, 1])
        lam = 1 - 2 * pi
        for row, (xcol, xlabel) in enumerate([(3, "M (balls)"), (1, "n (plays)")]):
            ax = axes[row, j]
            for k in range(config.U4_PX_RUNS):
                dk = d[d[:, 2] == k]
                ax.plot(dk[:, xcol], dk[:, 4], color="#2a78d6", lw=0.5, alpha=0.12)
            med_x = np.array([np.median(d[d[:, 1] == n][:, xcol]) for n in ns])
            med_y = np.array([np.median(d[d[:, 1] == n][:, 4]) for n in ns])
            ax.plot(med_x, med_y, color="0.1", lw=2.2, label="median")
            sel = (np.array([np.median(d[d[:, 1] == n][:, 3]) for n in ns]) >= config.U4_PX_FIT_FROM * M0)
            if sel.sum() >= 2:
                slope, icpt = np.polyfit(np.log(med_x[sel]), np.log(med_y[sel]), 1)
                fits.append((pi, xlabel.split()[0], int(sel.sum()), slope, lam, lam / pi))
            x0, y0m = med_x[sel][0], med_y[sel][0]
            xx = np.array([x0, med_x[-1]])
            for c, ls, name in [(lam, "--", f"slope λ = {lam:+.3f}"), (lam / pi, ":", f"slope λ/π = {lam / pi:+.3f}")]:
                ax.plot(xx, y0m * (xx / x0) ** c, color="#eb6834", lw=1.8, ls=ls, label=name)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("y (fraction of state-2 balls)")
            ax.set_title(f"π = {pi}", fontsize=10)
            ax.grid(color="0.92", lw=0.8)
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "pooling_exponent.png"), dpi=150)

    with open(os.path.join(RESULTS, "pooling_exponent_fit.csv"), "w") as fh:
        fh.write("pi,x_axis,n_points,fitted_slope,lambda,lambda_over_pi\n")
        for f in fits:
            fh.write("%g,%s,%d,%.4f,%.4f,%.4f\n" % f)
    print("\n傾き（中央値の log y を log M または log n に最小二乗、M ≥ %g·M0 の範囲）" % config.U4_PX_FIT_FROM)
    for pi, xa, k, s, lam, lp in fits:
        print(f"  π = {pi}  横軸 {xa:<2s}  点 {k:2d}  傾き {s:+.4f}   λ = {lam:+.4f}   λ/π = {lp:+.4f}")
    print("\ny が初期より増えた走行と、そのときの q_A（最後の時刻）")
    for pi in config.U4_PX_PIS:
        d = D[D[:, 0] == pi]
        last = d[d[:, 1] == d[:, 1].max()]
        grew = last[last[:, 4] > config.U4_PX_Y0]
        qA = last[:, 5]
        print(f"  π = {pi}: {len(grew)} 本  q_A = {np.round(grew[:, 5], 3).tolist()}")
        print(f"          全走行の最後の q_A：中央値 {np.median(qA):.3f}、5%〜95% [{np.quantile(qA, 0.05):.3f}, {np.quantile(qA, 0.95):.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    main(pilot=ap.parse_args().pilot)

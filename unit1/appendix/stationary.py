"""付録 手順一：定常分布。定常の FP 方程式の数値解と、長時間のシミュレーションのヒストグラムを重ねる。

使い方（sync-lab/unit1/appendix で）:
    python stationary.py            本番
    python stationary.py --pilot    軌道数と時間を減らした試し走り（所要時間の見積もり用。保存しない）

定常の FP 方程式（利用者が立てたもの）
    f(θ)·p − D·dp/dθ = J（定数）、周期境界 p(0) = p(2π)、正規化 ∫p dθ = 1
を、周期的な格子の上の有限体積法で解く。格子の境目での流れは、ドリフトが強く D が小さいときにも
崩れないように、指数の重みをつけた差分（Scharfetter–Gummel 型）で書く。すべての境目で流れが
等しいことを連立一次方程式にし、一行を正規化に置き換えて解く。J は境目の流れの平均。

シミュレーション：θ を一様に置いて T_BURN だけ捨て、T_RUN のあいだ SAMPLE_EVERY ステップごとに
θ mod 2π を数える。あわせて、T_RUN のあいだに θ が進んだ量（折り返さない）の平均を T_RUN で割り、
平均の回転の速さとする。

出力（results/）
    apx1_fp.csv       a, D, theta, p（FP の数値解）
    apx1_hist.csv     a, D, dt, theta, density（シミュレーションのヒストグラム）
    apx1_J.csv        a, D, J（FP）, 各 dt での平均の回転の速さとその標準誤差
    apx1_stationary.png
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
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import config  # noqa: E402
from model import f, noise_step  # noqa: E402

RESULTS = os.path.join(HERE, "results")
TWO_PI = 2.0 * np.pi


def _bern(x):
    """x / (e^x − 1)。x が 0 に近いところも安定に。"""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    out[small] = 1.0 - x[small] / 2.0
    out[~small] = x[~small] / np.expm1(x[~small])
    return out


def solve_fp(a, D, n):
    """定常の FP 方程式の数値解。セル中心の θ、p、J を返す。"""
    h = TWO_PI / n
    theta = (np.arange(n) + 0.5) * h
    face = (np.arange(n) + 1.0) * h            # セル i と i+1 の境目
    pe = f(face, config.OMEGA, a) * h / D
    cl = D / h * _bern(-pe)                     # 境目 i+1/2 の流れ = cl[i]·p[i] − cr[i]·p[i+1]
    cr = D / h * _bern(pe)
    ip = (np.arange(n) + 1) % n
    im = (np.arange(n) - 1) % n
    A = np.zeros((n, n))
    idx = np.arange(n)
    # セル i：流れ(i+1/2) − 流れ(i−1/2) = 0
    A[idx, idx] += cl
    A[idx, ip] -= cr
    A[idx, im] -= cl[im]
    A[idx, idx] += cr[im]
    b = np.zeros(n)
    A[0, :] = h
    b[0] = 1.0
    p = np.linalg.solve(A, b)
    flux = cl * p - cr * p[ip]
    return theta, p, flux.mean(), flux.std()


def simulate(a, D, dt, n_traj, t_burn, t_run, every, n_bins, rng):
    theta = rng.uniform(0.0, TWO_PI, n_traj)
    for _ in range(int(round(t_burn / dt))):
        theta += f(theta, config.OMEGA, a) * dt + noise_step(D, dt, rng, size=n_traj)
    start = theta.copy()
    counts = np.zeros(n_bins)
    n_run = int(round(t_run / dt))
    for k in range(1, n_run + 1):
        theta += f(theta, config.OMEGA, a) * dt + noise_step(D, dt, rng, size=n_traj)
        if k % every == 0:
            b = np.minimum((np.mod(theta, TWO_PI) / TWO_PI * n_bins).astype(int), n_bins - 1)
            counts += np.bincount(b, minlength=n_bins)
    density = counts / counts.sum() / (TWO_PI / n_bins)
    speed = (theta - start) / (n_run * dt)
    return density, speed.mean(), speed.std() / np.sqrt(n_traj)


def main(pilot=False):
    n_traj = config.APX1_N_TRAJ // (20 if pilot else 1)
    t_run = config.APX1_T_RUN / (10 if pilot else 1)
    rng = np.random.default_rng(config.APX1_SEED)
    nb = config.APX1_N_BINS
    centers = (np.arange(nb) + 0.5) * TWO_PI / nb

    fp, hist, rows = {}, {}, []
    for a in config.APX1_A_LIST:
        for D in config.APX1_D_LIST:
            th, p, J, Jsd = solve_fp(a, D, config.APX1_N_GRID)
            fp[(a, D)] = (th, p)
            row = [a, D, J, Jsd]
            for dt in config.APX1_DTS:
                t0 = time.time()
                dens, v, verr = simulate(a, D, dt, n_traj, config.APX1_T_BURN, t_run,
                                         config.APX1_SAMPLE_EVERY, nb, rng)
                hist[(a, D, dt)] = dens
                row += [v, verr]
                print(f"a = {a:g}, D = {D:g}, dt = {dt:g}: {time.time() - t0:.1f} s")
            rows.append(row)
    if pilot:
        return

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "apx1_fp.csv"), "w") as fh:
        fh.write("a,D,theta,p\n")
        for (a, D), (th, p) in fp.items():
            for x, y in zip(th, p):
                fh.write(f"{a:g},{D:g},{x:.6f},{y:.8g}\n")
    with open(os.path.join(RESULTS, "apx1_hist.csv"), "w") as fh:
        fh.write("a,D,dt,theta,density\n")
        for (a, D, dt), dens in hist.items():
            for x, y in zip(centers, dens):
                fh.write(f"{a:g},{D:g},{dt:g},{x:.6f},{y:.8g}\n")
    with open(os.path.join(RESULTS, "apx1_J.csv"), "w") as fh:
        head = "a,D,J_fp,J_fp_spread," + ",".join(f"speed_dt{dt:g},speed_err_dt{dt:g}"
                                                   for dt in config.APX1_DTS)
        fh.write(head + "\n")
        for r in rows:
            fh.write(",".join(f"{v:.8g}" for v in r) + "\n")

    # 図：行が a、列が D
    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    A_, D_ = config.APX1_A_LIST, config.APX1_D_LIST
    fig, axes = plt.subplots(len(A_), len(D_), figsize=(3.6 * len(D_), 3.2 * len(A_)),
                             squeeze=False)
    for i, a in enumerate(A_):
        for j, D in enumerate(D_):
            ax = axes[i, j]
            for k, dt in enumerate(config.APX1_DTS):
                ax.step(centers, hist[(a, D, dt)], where="mid", lw=1.5,
                        color=config.SERIES_COLORS[k % 3], label=f"simulation, dt = {dt:g}")
            th, p = fp[(a, D)]
            ax.plot(th, p, color="0.15", lw=1.2, ls="--", label="stationary FP (numerical)")
            ax.set_xlim(0, TWO_PI)
            ax.set_xticks(np.arange(5) * np.pi / 2)
            ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
            ax.set_title(f"a = {a:g}, D = {D:g}", fontsize=10)
            ax.grid(color="0.92", lw=0.8)
            if j == 0:
                ax.set_ylabel("p(θ)")
            if i == len(A_) - 1:
                ax.set_xlabel("θ")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "apx1_stationary.png"), dpi=150)

    print("\nJ（FP の数値解）と、シミュレーションで測った平均の回転の速さ ⟨dθ/dt⟩")
    print("  a      D       J_fp          " + "  ".join(f"speed (dt={dt:g})      " for dt in config.APX1_DTS))
    for r in rows:
        a, D, J = r[0], r[1], r[2]
        sp = "  ".join(f"{r[4 + 2 * k]:+.5f} ± {r[5 + 2 * k]:.5f}" for k in range(len(config.APX1_DTS)))
        print(f"  {a:<5g}  {D:<6g}  {J:.6e}  {sp}")
    print(f"\n保存：{RESULTS}/apx1_*.csv, apx1_stationary.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    main(pilot=ap.parse_args().pilot)

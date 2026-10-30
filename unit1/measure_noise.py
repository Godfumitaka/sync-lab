"""段階三：雑音ありで、2π の滑りが起きるまでの平均時間を測る。

使い方（sync-lab/unit1 で）:
    python measure_noise.py            本番（config.py の数のとおり）
    python measure_noise.py --pilot    軌道数を減らした試し走り（所要時間の見積もり用。保存しない）

滑り：θ が出発点 θs から ±2π 離れること（前向き・後ろ向きの両方を数え、後ろ向きの数も残す）。
出発点 θs：a > ω なら安定点 arcsin(ω/a)、a ≤ ω なら π/2。
滑るまでの時間は、越えたステップ数 × dt。STAGE3_T_MAX までに滑らなかった軌道は打ち切る。
すべての格子点・すべての軌道を一つの numpy の配列にまとめて進め、滑った軌道は配列から外す。

平均の二通りの書き方
    mean_done  ：滑り終えた軌道だけの平均（打ち切りがあると短めに偏る）
    mean_capped：打ち切った軌道を T_MAX として数えた平均（真の平均の下からの見積もり）
    打ち切りが一つもなければ、二つは一致する。

出力（results/）
    stage3_map.csv, stage3_map.png          出力1：平均時間の色の図（横 a − ω、縦 D）
    stage3_lock.csv                          ロック側の各 (δ, D) の平均時間
    stage3_views.csv, stage3_views.png       出力2：四通りの描き方と、直線への乗り（R²）
    stage3_slope.csv, stage3_slope.png       出力3：横軸 1/D・縦軸 ln(平均時間) の傾きを δ に対して両対数で
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
from model import f, noise_step  # noqa: E402

RESULTS = os.path.join(HERE, "results")
TWO_PI = 2.0 * np.pi


def start_theta(a):
    return np.arcsin(np.minimum(config.OMEGA / a, 1.0))


def slip_times(a_cells, D_cells, n_traj, dt, rng, t_max):
    """各セル (a, D) に n_traj 本の軌道を置き、滑るまでの時間を返す。

    返り値：times (n_cells, n_traj)、打ち切りは nan。back (n_cells, n_traj) 後ろ向きなら True。
    """
    n_cells = len(a_cells)
    cell = np.repeat(np.arange(n_cells), n_traj)
    slot = np.tile(np.arange(n_traj), n_cells)
    a = np.asarray(a_cells, float)[cell]
    D = np.asarray(D_cells, float)[cell]
    th0 = start_theta(a)
    th = th0.copy()

    times = np.full((n_cells, n_traj), np.nan)
    back = np.zeros((n_cells, n_traj), dtype=bool)
    n_max = int(round(t_max / dt))
    for n in range(1, n_max + 1):
        th += f(th, config.OMEGA, a) * dt + noise_step(D, dt, rng, size=th.size)
        disp = th - th0
        done = np.abs(disp) >= TWO_PI
        if done.any():
            times[cell[done], slot[done]] = n * dt
            back[cell[done], slot[done]] = disp[done] < 0
            keep = ~done
            cell, slot, a, D, th0, th = cell[keep], slot[keep], a[keep], D[keep], th0[keep], th[keep]
            if th.size == 0:
                break
    return times, back


def summarize(times, t_max):
    n = times.shape[1]
    ok = ~np.isnan(times)
    n_done = ok.sum(axis=1)
    with np.errstate(invalid="ignore"):
        mean_done = np.where(n_done > 0, np.nansum(times, axis=1) / np.maximum(n_done, 1), np.nan)
    mean_capped = np.where(ok, times, t_max).mean(axis=1)
    return n, n_done, mean_done, mean_capped


def linfit(x, y):
    p = np.polyfit(x, y, 1)
    resid = y - np.polyval(p, x)
    r2 = 1.0 - resid.var() / y.var()
    return p[0], p[1], r2


VIEWS = [  # 名前、横軸、縦軸
    ("linear", lambda D: D, lambda T: T, "D", "mean time"),
    ("semilog", lambda D: D, lambda T: np.log(T), "D", "ln(mean time)"),
    ("loglog", lambda D: np.log(D), lambda T: np.log(T), "ln D", "ln(mean time)"),
    ("inverse", lambda D: 1.0 / D, lambda T: np.log(T), "1/D", "ln(mean time)"),
]


def run(pilot=False):
    scale = 0.05 if pilot else 1.0
    n_map = max(4, int(config.STAGE3_N_TRAJ_MAP * scale))
    n_lock = max(4, int(config.STAGE3_N_TRAJ_LOCK * scale))
    t_max = config.STAGE3_T_MAX

    xs = np.linspace(config.STAGE3_X_MIN, config.STAGE3_X_MAX, config.STAGE3_N_X)
    Ds = np.logspace(config.STAGE3_LOG10_D_MIN, config.STAGE3_LOG10_D_MAX, config.STAGE3_N_D)
    X, DD = np.meshgrid(xs, Ds, indexing="ij")
    deltas = np.array(config.STAGE3_DELTAS, float)
    Dl = np.logspace(config.STAGE3_LOCK_LOG10_D_MIN, config.STAGE3_LOCK_LOG10_D_MAX,
                     config.STAGE3_LOCK_N_D)
    L, DL = np.meshgrid(deltas, Dl, indexing="ij")

    rng = np.random.default_rng(config.STAGE3_SEED)
    out = {}
    for dt in config.STAGE3_DTS:
        t0 = time.time()
        tm, bm = slip_times(config.OMEGA + X.ravel(), DD.ravel(), n_map, dt, rng, t_max)
        t1 = time.time()
        tl, bl = slip_times(config.OMEGA + L.ravel(), DL.ravel(), n_lock, dt, rng, t_max)
        t2 = time.time()
        out[dt] = dict(map=(tm, bm), lock=(tl, bl))
        print(f"dt = {dt:g}: 出力1の格子 {t1 - t0:.1f} s（{tm.size} 軌道）、"
              f"ロック側 {t2 - t1:.1f} s（{tl.size} 軌道）")
    if pilot:
        return None
    return dict(xs=xs, Ds=Ds, X=X, DD=DD, deltas=deltas, Dl=Dl, L=L, DL=DL,
                n_map=n_map, n_lock=n_lock, out=out)


def report(r):
    os.makedirs(RESULTS, exist_ok=True)
    t_max = config.STAGE3_T_MAX
    dts = config.STAGE3_DTS
    dt_fine = min(dts)

    # ---- CSV：格子とロック側 ----
    for key, A, B, name in (("map", r["X"], r["DD"], "a_minus_omega"),
                            ("lock", r["L"], r["DL"], "delta")):
        with open(os.path.join(RESULTS, f"stage3_{key}.csv"), "w") as fh:
            fh.write(f"dt,{name},D,n,n_done,n_back,mean_done,mean_capped\n")
            for dt in dts:
                times, back = r["out"][dt][key]
                n, nd, md, mc = summarize(times, t_max)
                nb = back.sum(axis=1)
                for i, (u, v) in enumerate(zip(A.ravel(), B.ravel())):
                    fh.write(f"{dt:g},{u:.6g},{v:.6g},{n},{nd[i]},{nb[i]},{md[i]:.6g},{mc[i]:.6g}\n")

    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})

    # ---- 出力1：色の図 ----
    fig, axes = plt.subplots(1, len(dts), figsize=(5.2 * len(dts) + 1, 4.6), squeeze=False)
    xs, Ds = r["xs"], r["Ds"]
    dx = xs[1] - xs[0]
    xe = np.concatenate([xs - dx / 2, [xs[-1] + dx / 2]])
    lD = np.log10(Ds)
    dl = lD[1] - lD[0]
    De = 10 ** np.concatenate([lD - dl / 2, [lD[-1] + dl / 2]])
    norm = LogNorm(vmin=1.0, vmax=t_max)
    for ax, dt in zip(axes[0], dts):
        times, _ = r["out"][dt]["map"]
        n, nd, md, mc = summarize(times, t_max)
        Z = mc.reshape(r["X"].shape)
        cens = (nd < n).reshape(r["X"].shape)
        im = ax.pcolormesh(xe, De, Z.T, norm=norm, cmap="viridis", shading="flat")
        cx, cy = np.nonzero(cens)
        ax.plot(xs[cx], Ds[cy], "x", color="white", ms=4, mew=1)
        ax.axvline(0.0, color="white", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("a − ω")
        ax.set_ylabel("D")
        ax.set_title(f"dt = {dt:g}", fontsize=10)
    fig.subplots_adjust(bottom=0.2)
    fig.colorbar(im, ax=axes[0].tolist(), label=f"mean time to 2π slip (capped at {t_max:g})")
    fig.text(0.01, 0.01, "×: some trajectories did not slip by T_MAX (value is a lower bound)",
             fontsize=8, color="0.35")
    fig.savefig(os.path.join(RESULTS, "stage3_map.png"), dpi=150)
    plt.close(fig)

    # ---- 出力2：四通りの描き方 ----
    deltas, Dl = r["deltas"], r["Dl"]
    colors = config.SERIES_COLORS_6
    view_rows = []
    usable = {}
    for dt in dts:
        times, _ = r["out"][dt]["lock"]
        n, nd, md, mc = summarize(times, t_max)
        md = md.reshape(r["L"].shape)
        okf = (nd / n >= config.STAGE3_DONE_FRAC).reshape(r["L"].shape)
        for i, d in enumerate(deltas):
            m = okf[i]
            usable[(dt, d)] = (Dl[m], md[i][m])
            if m.sum() < 3:
                continue
            for name, fx, fy, _, _ in VIEWS:
                s, c, r2 = linfit(fx(Dl[m]), fy(md[i][m]))
                view_rows.append((dt, d, name, m.sum(), s, c, r2))
    with open(os.path.join(RESULTS, "stage3_views.csv"), "w") as fh:
        fh.write("dt,delta,view,npts,slope,intercept,r2\n")
        for row in view_rows:
            fh.write("%g,%g,%s,%d,%.6g,%.6g,%.6f\n" % row)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, fx, fy, xl, yl) in zip(axes.ravel(), VIEWS):
        for i, d in enumerate(deltas):
            Dm, Tm = usable[(dt_fine, d)]
            if Dm.size == 0:
                continue
            ax.plot(fx(Dm), fy(Tm), "o-", ms=5, lw=1.5, color=colors[i % len(colors)],
                    label=f"δ = {d:g}")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(name, fontsize=10)
        ax.grid(color="0.9", lw=0.8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"lock side, dt = {dt_fine:g} (points with ≥{config.STAGE3_DONE_FRAC:.0%} slipped)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "stage3_views.png"), dpi=150)
    plt.close(fig)

    # ---- 出力3：横軸 1/D の傾きを δ に対して ----
    slope_rows = []
    fig, ax = plt.subplots(figsize=(6, 4.8))
    markers = ["o", "s", "^"]
    for j, dt in enumerate(dts):
        rows = [(d, s) for (dd, d, name, k, s, c, r2) in view_rows if dd == dt and name == "inverse"]
        if len(rows) < 2:
            continue
        d_arr = np.array([u for u, _ in rows])
        s_arr = np.array([v for _, v in rows])
        p, q, r2 = linfit(np.log10(d_arr), np.log10(s_arr))
        slope_rows.append((dt, len(rows), d_arr.min(), d_arr.max(), p, q, r2))
        ax.plot(d_arr, s_arr, markers[j % 3], ms=8 - 1.5 * j, mfc="none", mew=1.5,
                color=config.SERIES_COLORS[j % 3], label=f"dt = {dt:g}: fit slope {p:+.3f}")
        xx = np.array([d_arr.min(), d_arr.max()])
        ax.plot(xx, 10 ** (q + p * np.log10(xx)), color=config.SERIES_COLORS[j % 3], lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("δ = a − ω")
    ax.set_ylabel("slope of ln(mean time) vs 1/D")
    ax.grid(which="major", color="0.9", lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "stage3_slope.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(RESULTS, "stage3_slope.csv"), "w") as fh:
        fh.write("dt,n_delta,delta_min,delta_max,slope,intercept,r2\n")
        for row in slope_rows:
            fh.write("%g,%d,%g,%g,%.6f,%.6f,%.6f\n" % row)

    # ---- 報告 ----
    print("\n出力2：各 δ の、四通りの描き方での直線への乗り（R²）")
    for dt in dts:
        print(f"  dt = {dt:g}")
        for d in deltas:
            rr = {name: (k, r2) for (dd, de, name, k, s, c, r2) in view_rows if dd == dt and de == d}
            if not rr:
                print(f"    δ = {d:<5g}  使える点が 3 未満")
                continue
            k = next(iter(rr.values()))[0]
            print(f"    δ = {d:<5g} 点 {k:2d}  " +
                  "  ".join(f"{name} {rr[name][1]:.4f}" for name, *_ in VIEWS))
    print("\n出力3：横軸 1/D・縦軸 ln(平均時間) の傾きの、δ に対する両対数の傾き")
    for dt, k, d0, d1, p, q, r2 in slope_rows:
        print(f"  dt = {dt:g}: δ ∈ [{d0:g}, {d1:g}]（{k} 点）  slope = {p:+.4f}  R² = {r2:.4f}")
    print("\ndt による違い（両方の dt で全軌道が滑り終えた点での、平均時間の比 粗い dt / 細かい dt）")
    for key in ("map", "lock"):
        ms = []
        for dt in dts:
            times, _ = r["out"][dt][key]
            n, nd, md, mc = summarize(times, t_max)
            ms.append(np.where(nd == n, md, np.nan))
        ratio = ms[0] / ms[-1]
        ratio = ratio[~np.isnan(ratio)]
        print(f"  {key}: {ratio.size} 点  中央値 {np.median(ratio):.3f}  "
              f"範囲 [{ratio.min():.3f}, {ratio.max():.3f}]")
    nb = sum(r["out"][dt][k][1].sum() for dt in dts for k in ("map", "lock"))
    print(f"\n後ろ向きの滑りの総数：{nb}")
    print(f"\n保存：{RESULTS}/stage3_*.csv, stage3_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    res = run(pilot=args.pilot)
    if res is not None:
        report(res)

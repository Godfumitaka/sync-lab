"""位相の模型 段階0：二つのリズム（アドラー方程式）。

使い方（sync-lab/phase で）:
    python3 rhythm.py --selftest

二人の身体の位相 θ₁, θ₂（円の上の一点。一周 2π でもとに戻る）が、互いに合わせ合う。

    θ̇₁ = ω₁ − (K/2) sin(θ₁ − θ₂)
    θ̇₂ = ω₂ + (K/2) sin(θ₁ − θ₂)

ずれ φ = θ₁ − θ₂ は dφ/dt = Δω − K sin φ（Δω = ω₁ − ω₂）に従う。
ここの K は「ずれの式の係数」で、単元一の a と同じ、単元三の K の 2 倍。数は config.py の PH0_*。
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402


def omegas(d_omega, omega_mean=None):
    """Δω から二人の固有のリズム (ω₁, ω₂)。平均は ω̄。"""
    om = config.PH0_OMEGA_MEAN if omega_mean is None else omega_mean
    d = np.asarray(d_omega, dtype=float)
    return om + d / 2, om - d / 2


def velocity(theta1, theta2, omega1, omega2, K):
    """二人の速さ (θ̇₁, θ̇₂)。"""
    s = np.sin(theta1 - theta2)
    return omega1 - 0.5 * K * s, omega2 + 0.5 * K * s


def run_pair(d_omega, K, T, dt, phi0=None, omega_mean=None, every=1):
    """二人を時間 T だけ動かす（ルンゲ＝クッタ四次、雑音なし）。

    d_omega は数でも配列でもよい（配列なら、その数だけの組を同時に動かす）。
    返り値：t (n,)、θ₁, θ₂ (n, ...)。θ は円の上に折り返さない（何周したかを残す）。every ステップごとに記録する。
    """
    phi0 = config.PH0_PHI0 if phi0 is None else phi0
    w1, w2 = omegas(d_omega, omega_mean)
    th1 = np.full(np.shape(w1), phi0 / 2, dtype=float)
    th2 = np.full(np.shape(w2), -phi0 / 2, dtype=float)
    n = int(round(T / dt))
    keep = list(range(0, n + 1, every))
    out1 = np.empty((len(keep),) + th1.shape)
    out2 = np.empty_like(out1)
    k = 0
    for i in range(n + 1):
        if i == keep[k]:
            out1[k], out2[k] = th1, th2
            k += 1
            if k == len(keep):
                break
        a1, a2 = velocity(th1, th2, w1, w2, K)
        b1, b2 = velocity(th1 + 0.5 * dt * a1, th2 + 0.5 * dt * a2, w1, w2, K)
        c1, c2 = velocity(th1 + 0.5 * dt * b1, th2 + 0.5 * dt * b2, w1, w2, K)
        d1, d2 = velocity(th1 + dt * c1, th2 + dt * c2, w1, w2, K)
        th1 = th1 + dt / 6 * (a1 + 2 * b1 + 2 * c1 + d1)
        th2 = th2 + dt / 6 * (a2 + 2 * b2 + 2 * c2 + d2)
    return np.array(keep) * dt, out1, out2


def phi_rate(phi, d_omega, K):
    """ずれの速さ dφ/dt。"""
    return d_omega - K * np.sin(phi)


def beat(d_omega, K, T, skip, dt):
    """長く動かしたときの、平均のずれの速さ（滑りの頻度 × 2π）= (φ(T) − φ(skip)) / (T − skip)。"""
    every = max(1, int(round(0.05 / dt)))
    t, th1, th2 = run_pair(d_omega, K, T, dt, every=every)
    phi = th1 - th2
    sel = t >= skip
    return (phi[-1] - phi[sel][0]) / (t[-1] - t[sel][0])


# ---------------------------------------------------------------- 点検

def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'NG'}] {name} {detail}")

    K = 1.0
    # 1. 二人の式から作ったずれ φ が、dφ/dt = Δω − K sin φ を満たす（差分で）
    t, a, b = run_pair(1.3, K, 20.0, 1e-3)
    phi = a - b
    dphi = np.gradient(phi, t)
    err = np.abs(dphi[5:-5] - phi_rate(phi[5:-5], 1.3, K)).max()
    check("ずれの式 dφ/dt = Δω − K sin φ", err < 1e-5, f"(最大の差 {err:.1e})")

    # 2. 刻みを半分にしても変わらない（ルンゲ＝クッタ四次の誤差）
    _, a1, b1 = run_pair(np.array([0.5, 1.01, 2.0]), K, 100.0, config.PH0_DT)
    _, a2, b2 = run_pair(np.array([0.5, 1.01, 2.0]), K, 100.0, config.PH0_DT / 2)
    e = np.abs((a1[-1] - b1[-1]) - (a2[-1] - b2[-1])).max()
    check("刻みを半分にしたときの φ(100) の差", e < 1e-6, f"({e:.1e})")

    # 3. 揃う側：ずれは sin φ* = Δω/K の安定な点に落ち着き、二人は平均のリズム ω̄ で一緒に回る
    dw = 0.5
    t, a, b = run_pair(dw, K, 200.0, config.PH0_DT)
    phi_end = a[-1] - b[-1]
    v1, v2 = velocity(a[-1], b[-1], *omegas(dw), K)
    check("揃う側：sin φ* = Δω/K", abs(np.sin(phi_end) - dw / K) < 1e-9 and np.cos(phi_end) > 0,
          f"(φ* = {phi_end:.6f}, asin = {np.arcsin(dw / K):.6f})")
    check("揃う側：二人の速さ = ω̄", abs(v1 - config.PH0_OMEGA_MEAN) < 1e-9 and abs(v2 - config.PH0_OMEGA_MEAN) < 1e-9)

    # 4. 滑る側：一周の時間 2π/√(Δω² − K²)（単元一の段階二と同じ式。ここでは数値の点検にだけ使う）
    for dw in (1.01, 1.5, 3.0):
        Tp = 2 * np.pi / np.sqrt(dw ** 2 - K ** 2)
        t, a, b = run_pair(dw, K, 6 * Tp, 1e-3)
        phi = a - b
        laps = np.floor((phi - np.pi / 2) / (2 * np.pi))       # φ = π/2 + 2πm を通った回数
        cross = t[1:][np.diff(laps) > 0]
        meas = np.diff(cross).mean()
        check(f"滑る側の一周の時間（Δω = {dw}）", abs(meas / Tp - 1) < 1e-4, f"({meas:.4f} / {Tp:.4f})")

    # 5. 平均のずれの速さ（掃引の関数）：揃う側で 0、滑る側で 2π/一周の時間
    rate = beat(np.array([0.5, 0.99, 1.5, 2.5]), K, 600.0, 100.0, config.PH0_DT)
    exp = np.sqrt(np.clip(np.array([0.5, 0.99, 1.5, 2.5]) ** 2 - K ** 2, 0, None))
    check("平均のずれの速さ", np.abs(rate - exp).max() < 2 * np.pi / 500 + 1e-3, f"({np.round(rate, 4)})")

    print("selftest:", "すべて通った" if ok else "通らないものがある")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)

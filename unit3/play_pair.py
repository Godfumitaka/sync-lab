"""単元三：相手との協調（双方向）と、影との協調（一方向）を上下に並べて見る。

使い方（sync-lab/unit3 で）:
    python3 play_pair.py

上の段：相手（双方向）。二人とも相手に合わせる
下の段：影（一方向）。θ₁ だけが θ₂ に合わせ、θ₂ は自分のリズムのまま
二つの段で、ω₁、ω₂、K、D、dt はいつも同じ。雑音も同じ乱数を使う（違うのは合わせ方だけ）。

各段
    左  ：円の上の二つの点（青が θ₁、オレンジが θ₂）
    右  ：位相の差 φ = θ₁ − θ₂ を 2π で折り返さずに描いた時系列。点線は 2π の倍数
    文字：滑った回数（φ が 2π 進んだ回数）と、直近 FREQ_WINDOW の時間で測った二人の実際の速さ
下のスライダー
    Δω（ω₁ = 平均 + Δω/2、ω₂ = 平均 − Δω/2）、K、log10 D と noise のチェック、log10 dt、speed、window
    reset：二人を同じ位相（φ = 0）から出発させ直す
"""

import os
import sys
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.widgets import Button, CheckButtons, Slider

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
from pair import noise_step, velocities  # noqa: E402

TWO_PI = 2.0 * np.pi
SYSTEMS = [("partner (mutual)", True), ("shadow (one-way)", False)]


def _two_pi_label(y, _pos):
    k = int(round(y / TWO_PI))
    return {0: "0", 1: "2π", -1: "−2π"}.get(k, f"{k}·2π".replace("-", "−"))


class PairPlayer:
    def __init__(self, seed=config.U3_SEED):
        self.seed = seed
        self._build_figure()
        self.reset()

    # ---- パラメータ ----
    @property
    def dw(self):
        return self.s_dw.val

    @property
    def omegas(self):
        return config.U3_OMEGA_MEAN + self.dw / 2, config.U3_OMEGA_MEAN - self.dw / 2

    @property
    def K(self):
        return self.s_K.val

    @property
    def D(self):
        return 10.0 ** self.s_logD.val if self.chk_noise.get_status()[0] else 0.0

    @property
    def dt(self):
        return 10.0 ** self.s_logdt.val

    @property
    def speed(self):
        return 10.0 ** self.s_speed.val

    @property
    def window(self):
        return 10.0 ** self.s_win.val

    # ---- 状態 ----
    def reset(self, _event=None):
        self.rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.t_target = 0.0
        self.th = np.zeros((2, 2))                 # [段, 振動子]
        self.hist_t = deque([0.0], maxlen=50000)
        self.hist_th = deque([tuple(self.th.ravel())], maxlen=50000)
        self._redraw()

    def step(self):
        dt, D = self.dt, self.D
        w1, w2 = self.omegas
        xi = noise_step(D, dt, self.rng, size=2)   # 二人の雑音。二つの段で同じものを使う
        for s, (_, mutual) in enumerate(SYSTEMS):
            v1, v2 = velocities(self.th[s, 0], self.th[s, 1], w1, w2, self.K, mutual)
            self.th[s, 0] += v1 * dt + xi[0]
            self.th[s, 1] += v2 * dt + xi[1]
        self.t += dt

    def advance_frame(self):
        self.t_target += self.speed
        stride = max(1, int(self.speed / self.dt / 20))
        n = 0
        while self.t + self.dt <= self.t_target + 1e-12:
            self.step()
            n += 1
            if n % stride == 0:
                self.hist_t.append(self.t)
                self.hist_th.append(tuple(self.th.ravel()))
        if n % stride:
            self.hist_t.append(self.t)
            self.hist_th.append(tuple(self.th.ravel()))
        self._redraw()

    # ---- 図 ----
    def _build_figure(self):
        fig = plt.figure(figsize=(12, 8.5))
        self.fig = fig
        gs = fig.add_gridspec(2, 2, left=0.04, right=0.97, top=0.95, bottom=0.37,
                              width_ratios=[1, 2.6], hspace=0.35, wspace=0.12)
        ph = np.linspace(0, TWO_PI, 361)
        self.rows = []
        for s, (name, _) in enumerate(SYSTEMS):
            ax_c = fig.add_subplot(gs[s, 0])
            ax_c.plot(np.cos(ph), np.sin(ph), color="0.75", lw=1.5)
            (p1,) = ax_c.plot([], [], "o", ms=12, color=config.SERIES_COLORS[0])
            (p2,) = ax_c.plot([], [], "o", ms=12, color=config.SERIES_COLORS[1])
            ax_c.set_xlim(-1.25, 1.25)
            ax_c.set_ylim(-1.25, 1.25)
            ax_c.set_aspect("equal")
            ax_c.axis("off")
            ax_c.set_title(name, fontsize=10)
            ax_t = fig.add_subplot(gs[s, 1])
            (line,) = ax_t.plot([], [], color="0.2", lw=1)
            ax_t.yaxis.set_major_locator(MultipleLocator(TWO_PI))
            ax_t.yaxis.set_major_formatter(FuncFormatter(_two_pi_label))
            ax_t.grid(axis="y", ls=":", color="0.6")
            ax_t.set_ylabel("φ = θ₁ − θ₂")
            if s == 1:
                ax_t.set_xlabel("t")
            txt = ax_t.text(0.01, 0.97, "", transform=ax_t.transAxes, va="top",
                            fontsize=9, family="monospace")
            self.rows.append(dict(p1=p1, p2=p2, line=line, ax=ax_t, txt=txt))
        self.txt_top = fig.text(0.04, 0.305, "", fontsize=9, family="monospace", va="top")

        def slider(row, label, lo, hi, init, fmt=None):
            ax = fig.add_axes([0.12, 0.245 - 0.04 * row, 0.53, 0.025])
            return Slider(ax, label, lo, hi, valinit=init, valfmt=fmt)

        self.s_dw = slider(0, "Δω", config.U3_DW_MIN, config.U3_DW_MAX, config.U3_DW_INIT, "%.3f")
        self.s_K = slider(1, "K", config.U3_K_MIN, config.U3_K_MAX, config.U3_K_INIT, "%.3f")
        self.s_logD = slider(2, "log10 D", config.U3_LOG10_D_MIN, config.U3_LOG10_D_MAX,
                             config.U3_LOG10_D_INIT)
        self.s_logdt = slider(3, "log10 dt", np.log10(config.U3_DT_MIN), np.log10(config.U3_DT_MAX),
                              np.log10(config.U3_DT_INIT))
        self.s_speed = slider(4, "log10 speed", np.log10(config.U3_SPEED_MIN),
                              np.log10(config.U3_SPEED_MAX), np.log10(config.U3_SPEED_INIT))
        self.s_win = slider(5, "log10 window", np.log10(config.U3_WINDOW_MIN),
                            np.log10(config.U3_WINDOW_MAX), np.log10(config.U3_WINDOW_INIT))
        self.chk_noise = CheckButtons(fig.add_axes([0.74, 0.17, 0.09, 0.06]),
                                      ["noise"], [config.U3_NOISE_ON_INIT])
        self.btn_reset = Button(fig.add_axes([0.86, 0.175, 0.08, 0.05]), "reset")
        self.btn_reset.on_clicked(self.reset)

    def _redraw(self):
        ts = np.fromiter(self.hist_t, float)
        ths = np.array(self.hist_th).reshape(-1, 2, 2)   # (時刻, 段, 振動子)
        win = self.window
        t1 = max(self.t, win)
        t0 = t1 - win
        keep = ts >= self.t - win
        keep[:-1] |= keep[1:]
        # 実際の速さ：直近 FREQ_WINDOW の時間で平均
        fw = min(config.U3_FREQ_WINDOW, self.t)
        j = np.searchsorted(ts, self.t - fw) if fw > 0 else len(ts) - 1
        w1, w2 = self.omegas
        for s, row in enumerate(self.rows):
            a1, a2 = self.th[s]
            row["p1"].set_data([np.cos(a1)], [np.sin(a1)])
            row["p2"].set_data([np.cos(a2)], [np.sin(a2)])
            phi = ths[:, s, 0] - ths[:, s, 1]
            row["line"].set_data(ts[keep], phi[keep])
            row["ax"].set_xlim(t0, t1)
            vis = phi[keep] if keep.any() else phi[-1:]
            mid, half = 0.5 * (vis.min() + vis.max()), max(0.5 * (vis.max() - vis.min()), np.pi) * 1.15
            row["ax"].set_ylim(mid - half, mid + half)
            slips = int(np.floor((phi[-1] + np.pi) / TWO_PI))
            if fw > 0 and j < len(ts) - 1:
                span = ts[-1] - ts[j]
                f1 = (ths[-1, s, 0] - ths[j, s, 0]) / span
                f2 = (ths[-1, s, 1] - ths[j, s, 1]) / span
                freq = f"actual speed θ₁ {f1:.3f}, θ₂ {f2:.3f}  (last {span:.0f})"
            else:
                freq = ""
            row["txt"].set_text(f"slips {slips:+d}   {freq}")
        self.txt_top.set_text(f"t = {self.t:9.1f}   ω₁ = {w1:.3f}  ω₂ = {w2:.3f}  "
                              f"Δω = {self.dw:.3f}  K = {self.K:.3f}  D = {self.D:.2e}  dt = {self.dt:.1e}")

    def animate(self):
        self.anim = FuncAnimation(self.fig, lambda _i: self.advance_frame(),
                                  interval=30, cache_frame_data=False)
        return self.anim


if __name__ == "__main__":
    p = PairPlayer()
    p.animate()
    plt.show()

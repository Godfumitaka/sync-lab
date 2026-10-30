"""付録 手順三：雑音なしで a > ω に置いた点に、一時的な一押しを加えて遊ぶ。

使い方（sync-lab/unit1/appendix で）:
    python3 push.py

一ステップの更新は  θ ← θ + (f(θ, ω, a) + u(t))·dt。u(t) は押しているあいだだけ P、ほかは 0。
雑音は入れない。

画面
    左    ：円の上の点。灰色の小さな印が安定点（●）と不安定点（○）
    右上  ：今の θ のまわりのポテンシャル V(θ)（f = −dV/dθ）と、その上の玉。点線が谷と山の位置
    右下  ：θ を折り返さずに描いた時系列。押していたあいだを薄く塗る
    下    ：a、押しの強さ P、押しの長さ τ（対数目盛）、speed のスライダー
            push ボタン：今の P と τ で一回押す。reset ボタン：安定点に戻す
"""

import os
import sys
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.widgets import Button, Slider

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import config  # noqa: E402
from model import f  # noqa: E402
from play import _two_pi_label  # noqa: E402

TWO_PI = 2.0 * np.pi


def V(theta, omega, a):
    return -omega * theta - a * np.cos(theta)


class Pusher:
    def __init__(self):
        self.omega = config.OMEGA
        self._build_figure()
        self.reset()

    @property
    def a(self):
        return self.s_a.val

    @property
    def P(self):
        return self.s_P.val

    @property
    def tau(self):
        return 10.0 ** self.s_tau.val

    @property
    def speed(self):
        return 10.0 ** self.s_speed.val

    def stable_point(self):
        return np.arcsin(min(self.omega / self.a, 1.0))

    # ---- 状態 ----
    def reset(self, _event=None):
        self.theta = self.stable_point()
        self.t = 0.0
        self.t_target = 0.0
        self.push_end = -1.0
        self.push_P = 0.0
        self.pushes = []                      # (始まり, 終わり, P)
        self.base = self.theta                # 一周の数を数える基準
        self.hist_t = deque([0.0], maxlen=50000)
        self.hist_th = deque([self.theta], maxlen=50000)
        self._redraw()

    def push(self, _event=None):
        self.push_P = self.P
        self.push_end = self.t + self.tau
        self.pushes.append([self.t, self.push_end, self.push_P])

    def step(self):
        dt = config.APX3_DT
        u = self.push_P if self.t < self.push_end else 0.0
        self.theta += (f(self.theta, self.omega, self.a) + u) * dt
        self.t += dt

    def advance_frame(self):
        self.t_target += self.speed
        n = 0
        while self.t + config.APX3_DT <= self.t_target + 1e-12:
            self.step()
            n += 1
            if n % 20 == 0:
                self.hist_t.append(self.t)
                self.hist_th.append(self.theta)
        self.hist_t.append(self.t)
        self.hist_th.append(self.theta)
        self._redraw()

    # ---- 図 ----
    def _build_figure(self):
        fig = plt.figure(figsize=(12, 7.5))
        self.fig = fig
        gs = fig.add_gridspec(2, 2, left=0.05, right=0.97, top=0.95, bottom=0.33,
                              width_ratios=[1, 1.4], hspace=0.35, wspace=0.18)

        ax_c = fig.add_subplot(gs[:, 0])
        ph = np.linspace(0, TWO_PI, 361)
        ax_c.plot(np.cos(ph), np.sin(ph), color="0.75", lw=1.5)
        (self.mk_s,) = ax_c.plot([], [], "o", ms=6, color="0.5")
        (self.mk_u,) = ax_c.plot([], [], "o", ms=6, mfc="white", mec="0.5")
        (self.pt,) = ax_c.plot([], [], "o", ms=14, color="C0")
        ax_c.set_xlim(-1.3, 1.3)
        ax_c.set_ylim(-1.3, 1.3)
        ax_c.set_aspect("equal")
        ax_c.axis("off")
        self.txt = ax_c.text(-1.3, 1.3, "", fontsize=9, family="monospace", va="top")

        ax_v = fig.add_subplot(gs[0, 1])
        (self.vline,) = ax_v.plot([], [], color="C1")
        (self.ball,) = ax_v.plot([], [], "o", ms=12, color="C0")
        self.marks = []
        ax_v.set_xlabel("θ (unwrapped)")
        ax_v.set_ylabel("V(θ)")
        ax_v.set_yticks([])
        self.ax_v = ax_v

        ax_t = fig.add_subplot(gs[1, 1])
        (self.tline,) = ax_t.plot([], [], color="C0", lw=1)
        ax_t.yaxis.set_major_locator(MultipleLocator(TWO_PI))
        ax_t.yaxis.set_major_formatter(FuncFormatter(_two_pi_label))
        ax_t.grid(axis="y", ls=":", color="0.6")
        ax_t.set_xlabel("t")
        ax_t.set_ylabel("θ (unwrapped)")
        self.ax_t = ax_t
        self.spans = []

        def slider(row, label, lo, hi, init, fmt=None):
            ax = fig.add_axes([0.12, 0.25 - 0.055 * row, 0.53, 0.03])
            return Slider(ax, label, lo, hi, valinit=init, valfmt=fmt)

        self.s_a = slider(0, "a", config.APX3_A_MIN, config.APX3_A_MAX, config.APX3_A_INIT, "%.3f")
        self.s_P = slider(1, "push P", config.APX3_P_MIN, config.APX3_P_MAX, config.APX3_P_INIT, "%.3f")
        self.s_tau = slider(2, "log10 τ", np.log10(config.APX3_TAU_MIN), np.log10(config.APX3_TAU_MAX),
                            np.log10(config.APX3_TAU_INIT))
        self.s_speed = slider(3, "log10 speed", np.log10(config.APX3_SPEED_MIN),
                              np.log10(config.APX3_SPEED_MAX), np.log10(config.APX3_SPEED_INIT))
        self.btn_push = Button(fig.add_axes([0.74, 0.18, 0.09, 0.06]), "push")
        self.btn_push.on_clicked(self.push)
        self.btn_reset = Button(fig.add_axes([0.86, 0.18, 0.08, 0.06]), "reset")
        self.btn_reset.on_clicked(self.reset)

    def _redraw(self):
        th, a, w = self.theta, self.a, self.omega
        th_s = self.stable_point()
        th_u = np.pi - th_s

        self.pt.set_data([np.cos(th)], [np.sin(th)])
        self.mk_s.set_data([np.cos(th_s)], [np.sin(th_s)])
        self.mk_u.set_data([np.cos(th_u)], [np.sin(th_u)])

        hw = config.APX3_V_HALF_WIDTH
        xs = np.linspace(th - hw, th + hw, 400)
        vs = V(xs, w, a)
        self.vline.set_data(xs, vs)
        self.ball.set_data([th], [V(th, w, a)])
        self.ax_v.set_xlim(xs[0], xs[-1])
        pad = 0.08 * (vs.max() - vs.min() + 1e-9)
        self.ax_v.set_ylim(vs.min() - pad, vs.max() + pad)
        for m in self.marks:
            m.remove()
        self.marks = []
        for base in (th_s, th_u):
            k0 = np.ceil((xs[0] - base) / TWO_PI)
            k1 = np.floor((xs[-1] - base) / TWO_PI)
            for k in np.arange(k0, k1 + 1):
                self.marks.append(self.ax_v.axvline(base + k * TWO_PI, color="0.6", lw=0.8, ls=":"))

        ts = np.fromiter(self.hist_t, float)
        ths = np.fromiter(self.hist_th, float)
        win = config.APX3_TIME_WINDOW
        t1 = max(self.t, win)
        t0 = t1 - win
        self.tline.set_data(ts, ths)
        self.ax_t.set_xlim(t0, t1)
        vis = ths[ts >= t0 - 1e-9]
        if vis.size == 0:
            vis = ths[-1:]
        mid, half = 0.5 * (vis.min() + vis.max()), max(0.5 * (vis.max() - vis.min()), np.pi) * 1.15
        self.ax_t.set_ylim(mid - half, mid + half)
        for sp in self.spans:
            sp.remove()
        self.spans = [self.ax_t.axvspan(s, min(e, self.t), color="C1", alpha=0.15, lw=0)
                      for s, e, _ in self.pushes if s <= self.t and e >= t0]

        slips = int(np.floor((th - self.base + np.pi) / TWO_PI))
        pushing = self.t < self.push_end
        self.txt.set_text(f"t      = {self.t:8.2f}\na      = {a:.3f}\n"
                          f"P      = {self.P:+.3f}\nτ      = {self.tau:.3g}\n"
                          f"P·τ    = {self.P * self.tau:+.3g}\n"
                          f"slips  = {slips:+d}\n" + ("PUSHING" if pushing else ""))

    def animate(self):
        self.anim = FuncAnimation(self.fig, lambda _i: self.advance_frame(),
                                  interval=30, cache_frame_data=False)
        return self.anim


if __name__ == "__main__":
    p = Pusher()
    p.animate()
    plt.show()

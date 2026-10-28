"""段階一：θ の動きを見ながら a・D・dt を動かす。

使い方（sync-lab/unit1 で）:
    python play.py

画面
    左    ：円の上を動く点
    右上  ：横軸 θ、縦軸 f(θ)。縦線が今の θ（0〜2π に折り返した位置）
    右下  ：θ を 2π で折り返さずに描いた時系列。横の点線は 2π の倍数
    下    ：a・D・dt のスライダー、「noise」のチェック（外すと D = 0）、reset ボタン
            a は右の欄に数を打ち込んで Enter でも決められる（0.999 など、指で合わせにくい値のため）
            speed：一コマで進める模型の時間。window：右下に見せる模型の時間の幅

一ステップの更新は  θ ← θ + f(θ, ω, a)·dt + noise_step(D, dt, rng)  だけ。
f と noise_step は既定では model.py のものを使う。run(f=..., noise_step=...) で差し替えられる。
"""

import os
import sys
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

TWO_PI = 2.0 * np.pi


def _f_on_grid(f, grid, omega, a):
    """右上のグラフ用。f が配列を受け付けなければ一点ずつ呼ぶ。"""
    try:
        y = np.asarray(f(grid, omega, a), dtype=float)
        if y.shape == grid.shape:
            return y
    except Exception:
        pass
    return np.array([float(f(x, omega, a)) for x in grid])


def _two_pi_label(y, _pos):
    k = int(round(y / TWO_PI))
    if k == 0:
        return "0"
    if k == 1:
        return "2π"
    if k == -1:
        return "−2π"
    return f"{k}·2π".replace("-", "−")


class Player:
    def __init__(self, f, noise_step, seed=config.SEED):
        self.f = f
        self.noise_step = noise_step
        self.omega = config.OMEGA
        self.seed = seed

        self._build_figure()
        self.reset()

    # ---- 状態 ----
    def reset(self, _event=None):
        self.rng = np.random.default_rng(self.seed)
        self.theta = float(config.THETA_INIT)
        self.t = 0.0
        self.t_target = 0.0
        self.hist_t = deque([self.t], maxlen=config.MAX_HISTORY)
        self.hist_theta = deque([self.theta], maxlen=config.MAX_HISTORY)
        self._redraw()

    @property
    def a(self):
        return self.s_a.val

    @property
    def D(self):
        return 10.0 ** self.s_logD.val if self.chk_noise.get_status()[0] else 0.0

    @property
    def dt(self):
        return 10.0 ** self.s_logdt.val

    @property
    def sim_per_frame(self):
        return 10.0 ** self.s_logspeed.val

    @property
    def window(self):
        return 10.0 ** self.s_logwin.val

    def step(self):
        """一ステップ。f と noise_step を呼ぶだけ。"""
        dt = self.dt
        self.theta = float(self.theta + self.f(self.theta, self.omega, self.a) * dt
                           + self.noise_step(self.D, dt, self.rng))
        self.t += dt

    def advance_frame(self):
        """一コマ分（模型の時間 speed）進める。dt が大きいと何コマかに一回だけ進む。"""
        self.t_target += self.sim_per_frame
        # 時系列に残す点は一コマあたり POINTS_PER_FRAME 個ほどに間引く
        stride = max(1, int(self.sim_per_frame / self.dt / config.POINTS_PER_FRAME))
        n = 0
        while self.t + self.dt <= self.t_target + 1e-12:
            self.step()
            n += 1
            if n % stride == 0:
                self.hist_t.append(self.t)
                self.hist_theta.append(self.theta)
        if n % stride:
            self.hist_t.append(self.t)
            self.hist_theta.append(self.theta)
        self._redraw()
        return self.artists

    # ---- 図 ----
    def _build_figure(self):
        fig = plt.figure(figsize=(12, 7.5))
        self.fig = fig
        gs = fig.add_gridspec(2, 2, left=0.05, right=0.97, top=0.95, bottom=0.36,
                              width_ratios=[1, 1.4], hspace=0.35, wspace=0.18)

        # 左：円
        ax_c = fig.add_subplot(gs[:, 0])
        ph = np.linspace(0, TWO_PI, 361)
        ax_c.plot(np.cos(ph), np.sin(ph), color="0.75", lw=1.5)
        ax_c.plot([1.0, 1.12], [0, 0], color="0.4", lw=1)
        ax_c.text(1.16, 0, "θ=0", va="center", fontsize=9, color="0.4")
        (self.pt,) = ax_c.plot([], [], "o", ms=14, color="C0")
        (self.spoke,) = ax_c.plot([], [], color="C0", lw=1, alpha=0.5)
        ax_c.set_xlim(-1.3, 1.45)
        ax_c.set_ylim(-1.3, 1.3)
        ax_c.set_aspect("equal")
        ax_c.axis("off")
        self.txt = ax_c.text(-1.3, 1.3, "", fontsize=9, family="monospace", va="top")

        # 右上：f(θ)
        ax_f = fig.add_subplot(gs[0, 1])
        self.grid = np.linspace(0.0, TWO_PI, config.N_THETA_GRID)
        (self.fline,) = ax_f.plot([], [], color="C1")
        ax_f.axhline(0.0, color="0.6", lw=0.8)
        self.vline = ax_f.axvline(0.0, color="C0", lw=1.5)
        ax_f.set_xlim(0, TWO_PI)
        ax_f.xaxis.set_major_locator(MultipleLocator(np.pi / 2))
        ax_f.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _p: ["0", "π/2", "π", "3π/2", "2π"][int(round(x / (np.pi / 2)))]
            if 0 <= int(round(x / (np.pi / 2))) <= 4 else ""))
        ax_f.set_xlabel("θ")
        ax_f.set_ylabel("f(θ)")
        self.ax_f = ax_f

        # 右下：θ(t)（折り返さない）
        ax_t = fig.add_subplot(gs[1, 1])
        (self.tline,) = ax_t.plot([], [], color="C0", lw=1)
        ax_t.yaxis.set_major_locator(MultipleLocator(TWO_PI))
        ax_t.yaxis.set_major_formatter(FuncFormatter(_two_pi_label))
        ax_t.grid(axis="y", ls=":", color="0.6")
        ax_t.set_xlabel("t")
        ax_t.set_ylabel("θ (unwrapped)")
        self.ax_t = ax_t

        # スライダーなど
        def slider(row, label, lo, hi, init, fmt=None):
            ax = fig.add_axes([0.12, 0.26 - 0.055 * row, 0.53, 0.03])
            return Slider(ax, label, lo, hi, valinit=init, valfmt=fmt)

        self.s_a = slider(0, "a", config.A_MIN, config.A_MAX, config.A_INIT, "%.4f")
        self.s_logD = slider(1, "log10 D", config.LOG10_D_MIN, config.LOG10_D_MAX,
                             config.LOG10_D_INIT)
        self.s_logdt = slider(2, "log10 dt", np.log10(config.DT_MIN), np.log10(config.DT_MAX),
                              np.log10(config.DT_INIT))
        self.s_logspeed = slider(3, "log10 speed", np.log10(config.SPEED_MIN),
                                 np.log10(config.SPEED_MAX), np.log10(config.SPEED_INIT))
        self.s_logwin = slider(4, "log10 window", np.log10(config.WINDOW_MIN),
                               np.log10(config.WINDOW_MAX), np.log10(config.WINDOW_INIT))
        self.tb_a = TextBox(fig.add_axes([0.76, 0.255, 0.08, 0.04]), "a = ",
                            initial=f"{config.A_INIT}")
        self.tb_a.on_submit(self._set_a_from_text)
        self.chk_noise = CheckButtons(fig.add_axes([0.74, 0.13, 0.09, 0.07]),
                                      ["noise"], [config.NOISE_ON_INIT])
        self.btn_reset = Button(fig.add_axes([0.86, 0.14, 0.08, 0.05]), "reset")
        self.btn_reset.on_clicked(self.reset)
        self.s_a.on_changed(lambda _v: self._update_f_curve())

        self.artists = [self.pt, self.spoke, self.txt, self.fline, self.vline, self.tline]
        self._update_f_curve()

    def _set_a_from_text(self, text):
        try:
            v = float(text)
        except ValueError:
            return
        self.s_a.set_val(min(max(v, config.A_MIN), config.A_MAX))

    def _update_f_curve(self):
        y = _f_on_grid(self.f, self.grid, self.omega, self.a)
        self.fline.set_data(self.grid, y)
        lo, hi = float(min(y.min(), 0.0)), float(max(y.max(), 0.0))
        pad = 0.1 * max(hi - lo, 1e-3)
        self.ax_f.set_ylim(lo - pad, hi + pad)
        self.fig.canvas.draw_idle()

    def _redraw(self):
        th = self.theta
        x, y = np.cos(th), np.sin(th)
        self.pt.set_data([x], [y])
        self.spoke.set_data([0, x], [0, y])
        self.vline.set_xdata([th % TWO_PI] * 2)

        ts = np.fromiter(self.hist_t, float)
        ths = np.fromiter(self.hist_theta, float)
        keep = ts >= self.t - self.window
        keep[:-1] |= keep[1:]          # 窓の左端の外の一点も描いて線を端までつなぐ
        self.tline.set_data(ts[keep], ths[keep])
        t1 = max(self.t, self.window)
        t0 = t1 - self.window
        self.ax_t.set_xlim(t0, t1)
        vis = ths[ts >= t0]
        if vis.size == 0:
            vis = ths[-1:]
        lo, hi = vis.min(), vis.max()
        mid, half = 0.5 * (lo + hi), max(0.5 * (hi - lo), np.pi) * 1.1
        self.ax_t.set_ylim(mid - half, mid + half)

        self.txt.set_text(f"t = {self.t:9.2f}\na = {self.a:.3f}\n"
                          f"D = {self.D:.2e}\ndt = {self.dt:.2e}\n"
                          f"turns = {th / TWO_PI:+.2f}")

    def animate(self):
        # blit を使わない：軸の範囲を毎コマ動かすため
        self.anim = FuncAnimation(self.fig, lambda _i: self.advance_frame(),
                                  interval=config.FRAME_INTERVAL_MS, cache_frame_data=False)
        return self.anim


def _check_callable(f, noise_step):
    """TODO が未記入なら、窓を開く前に知らせて止まる。"""
    rng = np.random.default_rng(0)
    for name, call in (("f", lambda: f(config.THETA_INIT, config.OMEGA, config.A_INIT)),
                       ("noise_step", lambda: noise_step(0.0, config.DT_INIT, rng))):
        try:
            call()
        except NotImplementedError:
            sys.exit(f"model.py の {name} の TODO の一行がまだ書かれていません。")


def run(f=None, noise_step=None):
    if f is None or noise_step is None:
        import model
        f = f or model.f
        noise_step = noise_step or model.noise_step
    _check_callable(f, noise_step)
    player = Player(f, noise_step)
    player.animate()
    plt.show()
    return player


if __name__ == "__main__":
    run()

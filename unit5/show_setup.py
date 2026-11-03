"""単元五：設定を目で見る図。世界の生成だけを使い、エージェントの学習は走らせない。

使い方（sync-lab/unit5 で）:
    python3 show_setup.py

指令 u(t) と 6 本のチャンネル s_i(t) を U5_SHOW_T ステップ分（config の世界のまま、種 U5_SHOW_SEED）。
各チャンネルは、感覚（太線）に、その中身の「糸の分」g u(t − d) と「風の分」c w(t) を細線で重ねる
（残りが毎回独立な雑音）。エージェントに見えるのは、指令と太線だけ。

出力：results/setup_waveforms.png
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
from selfworld import default_env, pad_for, simulate  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    env = default_env()
    T = config.U5_SHOW_T
    rng = np.random.default_rng(config.U5_SHOW_SEED)
    pad = pad_for(env, config.U5_LAGS)
    u, s, world, noise = simulate(rng, 1, T, env, pad, parts=True)
    t = np.arange(T)
    u_t = u[0, pad:pad + T]
    s_t = s[0, 1:]
    w_t = world[0, 1:]
    string_t = s_t - w_t - noise[0, 1:]

    matplotlib.rcParams.update({"font.family": "IPAGothic", "axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35",
                                "axes.spines.top": False, "axes.spines.right": False,
                                "axes.unicode_minus": False})
    c_self, c_wind, c_u = "#2a78d6", "#eb6834", "0.15"
    N = env["n_ch"]
    fig, axs = plt.subplots(N + 1, 1, figsize=(11, 13), sharex=True,
                            gridspec_kw={"height_ratios": [1.0] + [1.0] * N})

    # 指令：見どころとして、20〜70 の中でいちばん大きな指令の時刻に縦線を引き、糸つきのチャンネルでは d だけ後ろにずらして引く
    lo, hi = 20, 70
    t_mark = lo + int(np.argmax(u_t[lo:hi]))
    a0 = axs[0]
    a0.plot(t, u_t, color=c_u, lw=1.5, marker="o", ms=2.5)
    a0.axvline(t_mark, color="0.45", lw=1, ls=":")
    a0.set_ylabel("u(t)")
    a0.set_title("指令（でたらめ）：エージェントは、この控えを持っている", loc="left", fontsize=10, pad=3)

    ylim = np.max(np.abs(s_t)) * 1.1
    for i in range(N):
        ax = axs[i + 1]
        is_self = i in env["self_ch"]
        col = c_self if is_self else c_wind
        ax.plot(t, s_t[:, i], color=col, lw=2.0)
        if is_self:
            d = env["delays"][env["self_ch"].index(i)]
            ax.plot(t, string_t[:, i], color="0.1", lw=0.9, ls="--")
            ax.axvline(t_mark + d, color="0.45", lw=1, ls=":")
            name = f"ch{i}　糸つき（遅れ d = {d}）＋ 風 ＋ 雑音"
        else:
            name = f"ch{i}　風だけ ＋ 雑音（指令は届かない）"
        ax.plot(t, w_t[:, i], color="0.6", lw=0.9)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel(f"s{i}(t)")
        ax.set_title(name, loc="left", fontsize=10, color=col, pad=3)
    axs[-1].set_xlabel("t（ステップ）")
    for ax in axs:
        ax.grid(color="0.93", lw=0.7, axis="x")
    fig.suptitle(f"段階0の世界（学習なし）：糸の大きさ g = {env['gain']}、風の大きさ c = {env['world_amp']}"
                 f"（相関時間 τ = {env['world_tau']:g} ステップ）、雑音 σ = {env['noise_std']}\n"
                 f"点線：t = {t_mark} のいちばん大きな指令と、それが糸つきのチャンネルに届く時刻 t + d。"
                 f"縦軸の目盛りは6本で共通\n"
                 f"エージェントに見えるのは、指令と太線だけ（細線と点線は、こちらが中身を分けて見せたもの）",
                 fontsize=10.5, y=0.997)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c_self, lw=2.0, label="感覚 s(t)：糸つきのチャンネル"),
               Line2D([], [], color=c_wind, lw=2.0, label="感覚 s(t)：風だけのチャンネル"),
               Line2D([], [], color="0.1", lw=0.9, ls="--", label="糸の分 g·u(t − d)"),
               Line2D([], [], color="0.6", lw=0.9, label="風の分 c·w(t)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=4, frameon=False,
               fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(os.path.join(RESULTS, "setup_waveforms.png"), dpi=200)
    plt.close(fig)
    print(f"保存しました：results/setup_waveforms.png（点線の時刻 t = {t_mark}）")


if __name__ == "__main__":
    main()

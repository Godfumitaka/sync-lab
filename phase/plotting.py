"""位相の模型の図の共通の設定（字体と線の色）。"""

import os
import sys

import matplotlib
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402


def setup():
    """日本語の字体（config.PH_FONTS のうち入っているもの）と、軸の見た目。"""
    have = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in config.PH_FONTS if f in have] + ["DejaVu Sans"]
    matplotlib.rcParams.update({"font.family": fonts, "axes.unicode_minus": False,
                                "axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35",
                                "axes.spines.top": False, "axes.spines.right": False,
                                "axes.grid": True, "grid.color": "0.92", "grid.linewidth": 0.8})

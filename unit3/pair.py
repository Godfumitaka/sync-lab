"""単元三のモデル：二つの振動子。

    θ̇₁ = ω₁ + K sin(θ₂ − θ₁)
    θ̇₂ = ω₂ + K sin(θ₁ − θ₂)   （双方向：相手）
    θ̇₂ = ω₂                     （一方向：影。θ₂ は誰にも合わせない）

雑音は二人それぞれに独立に入る。一ステップの雑音は単元一の noise_step を使う。
"""

import importlib.util
import os

import numpy as np

# 単元一の model.py（noise_step）を、名前がぶつからないように読み込む
_spec = importlib.util.spec_from_file_location(
    "unit1_model", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unit1", "model.py"))
_unit1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_unit1)
noise_step = _unit1.noise_step


def velocities(theta1, theta2, omega1, omega2, K, mutual):
    """二人の決定論的な速さ。mutual が False なら θ₂ は結合を受けない。"""
    v1 = omega1 + K * np.sin(theta2 - theta1)
    v2 = omega2 + (K * np.sin(theta1 - theta2) if mutual else 0.0 * theta2)
    return v1, v2

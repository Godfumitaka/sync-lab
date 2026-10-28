"""単元一のモデル。

play.py・measure.py・measure_noise.py は、この二つの関数だけを通してモデルを使う。
各関数の TODO の一行は利用者が書く。
"""


def f(theta, omega, a):
    """決定論的な速さ。

    theta には数のほか、numpy の配列が渡されることもある
    （play.py の右上のグラフ、段階三のまとめての計算）。
    """
    raise NotImplementedError("TODO")  # TODO: この一行を書く


def noise_step(D, dt, rng, size=None):
    """一ステップ分の雑音による変位。

    rng は numpy.random.Generator。
    size は返す変位の形。None のときは一つの数を、
    たとえば size=1000 のときは 1000 本の独立な軌道の分を返す（段階三で使う）。
    """
    raise NotImplementedError("TODO")  # TODO: この一行を書く

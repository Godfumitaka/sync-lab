"""sync-lab の数値はすべてここに書く。"""

# ---- モデル ----
OMEGA = 1.0

# ---- 段階一：play.py ----
# a のスライダー
A_MIN = 0.0
A_MAX = 1.5
A_INIT = 0.5

# D のスライダー（対数目盛：log10 D）。「雑音」のチェックを外すと D = 0
LOG10_D_MIN = -4.0
LOG10_D_MAX = -1.0
LOG10_D_INIT = -3.0
NOISE_ON_INIT = False

# dt のスライダー（対数目盛：log10 dt）
DT_MIN = 1e-3
DT_MAX = 0.5
DT_INIT = 1e-2

THETA_INIT = 0.0
SEED = 0

FRAME_INTERVAL_MS = 30      # 画面の一コマの間隔（実時間）

# speed のスライダー（対数目盛）：一コマで進める模型の時間
SPEED_MIN = 0.01
SPEED_MAX = 10.0
SPEED_INIT = 0.1

# window のスライダー（対数目盛）：右下の時系列に見せる模型の時間の幅
WINDOW_MIN = 10.0
WINDOW_MAX = 1000.0
WINDOW_INIT = 100.0

POINTS_PER_FRAME = 20       # 時系列に残す点の、一コマあたりのおよその数
MAX_HISTORY = 100000        # 時系列に保持する点の数の上限
N_THETA_GRID = 400          # 右上のグラフの θ の刻み数

# ---- 段階二：measure.py（雑音なし） ----
STAGE2_EPS_MIN = 1e-4       # ε = ω − a の最小
STAGE2_EPS_MAX = 1e-1       # ε の最大
STAGE2_N_EPS = 13           # ε を対数等間隔にとる個数
STAGE2_DTS = [1e-1, 1e-2, 1e-3]
STAGE2_THETA0 = 0.0         # 一周を測り始める θ
STAGE2_T_MAX = 1e5          # 一周がこれを超えたら打ち切る（模型の時間）

# 図の色（一系列ずつ、決まった順に使う）
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

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

# ---- 段階三：measure_noise.py（雑音あり） ----
STAGE3_DTS = [0.05, 0.01]
STAGE3_T_MAX = 1e4          # これまでに滑らなかった軌道は打ち切り（模型の時間）
STAGE3_SEED = 1

# 出力1：(a − ω, D) の格子
STAGE3_X_MIN = -0.1         # a − ω の範囲（閾値 0 の両側）
STAGE3_X_MAX = 0.2
STAGE3_N_X = 16
STAGE3_LOG10_D_MIN = -4.0
STAGE3_LOG10_D_MAX = -1.0
STAGE3_N_D = 13
STAGE3_N_TRAJ_MAP = 200     # 一つの格子点あたりの軌道数

# 出力2・3：ロック側（a − ω = δ > 0）
STAGE3_DELTAS = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
STAGE3_LOCK_LOG10_D_MIN = -3.5
STAGE3_LOCK_LOG10_D_MAX = -0.75
STAGE3_LOCK_N_D = 12
STAGE3_N_TRAJ_LOCK = 400
STAGE3_DONE_FRAC = 0.99     # 当てはめに使うのは、この割合以上の軌道が滑り終えた点だけ

# 図の色（決まった順に使う。六系列まで）
SERIES_COLORS_6 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

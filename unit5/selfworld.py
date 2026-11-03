"""単元五：自分の指令の発見。感覚の生成（世界）と、エージェントの推論。

使い方（sync-lab/unit5 で）:
    python3 selfworld.py --selftest

--selftest は式と実装の点検だけを行い、見分けるまでの時間などの結果の統計は出さない。
記号と、論文外の選択はすべて ASSUMPTIONS.md にある。数は config.py の U5_*。

世界（感覚の生成）
    s_i(t) = g u(t − d_i) [自分のチャンネルだけ] + c w_i(t) + σ ε_i(t)
    u(t)：指令。毎回独立な正規分布（でたらめ）
    w_i(t)：世界の揺れ。チャンネルごとに独立な OU 過程（分散 1、一歩の相関 a = exp(−1/τ)）
    ε_i(t)：毎回独立な雑音

エージェント（チャンネル i ごとの仮説）
    s_i(t) = θ_i・ũ(t) [「自分」のときだけ] + x_i(t),   x_i(t) = φ_i x_i(t−1) + η_i(t),   η_i ~ N(0, v_i)
    ũ(t) = (u(t), u(t−1), ..., u(t−L))：指令の控え
    「自分」なら θ_i ~ N(0, θ0 I)、そうでなければ θ_i = 0（事前は五分五分）
    「自分だけ」の模型 M1：φ_i = 0（指令で説明できない残りは、毎回独立）
    「自分と世界」の模型 M2：φ_i は (0, 1) の上で一様（指令で説明できない残りに、記憶がある）
    v_i は log10 v ~ N(m, s²)。
    積分：θ_i は式で厳密に、φ_i は (0, 1) を等分した区間の中点の和で、v_i は ln v のまわりのラプラス近似で。
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402

LOG2PI = np.log(2.0 * np.pi)
LN10 = np.log(10.0)


# ---------------------------------------------------------------- 世界

def default_env():
    """config の値で世界を決める。"""
    return dict(n_ch=config.U5_N_CH, self_ch=list(config.U5_SELF_CH), delays=list(config.U5_SELF_DELAYS),
                gain=config.U5_SELF_GAIN, u_std=config.U5_U_STD, world_amp=config.U5_WORLD_AMP,
                world_tau=config.U5_WORLD_TAU, noise_std=config.U5_NOISE_STD)


def simulate(rng, n_runs, T, env, pad, parts=False):
    """n_runs 本を T ステップ。

    返り値
        u : (R, pad + T)      u[:, pad + t] = u(t)、t = −pad, ..., T − 1
        s : (R, T + 1, N)     s[:, 1 + t]   = s(t)、t = −1, ..., T − 1
        parts=True なら、(u, s, 世界の項 c w, 雑音の項 σ ε) を返す
    """
    n = env["n_ch"]
    if pad < max(env["delays"], default=0) + 1:
        raise ValueError("pad が遅れより短い")
    u = rng.normal(0.0, env["u_std"], size=(n_runs, pad + T))
    a = np.exp(-1.0 / env["world_tau"])
    b = np.sqrt(1.0 - a * a)
    w = np.empty((n_runs, T + 1, n))
    w[:, 0] = rng.normal(size=(n_runs, n))            # 定常分布から始める
    xi = rng.normal(size=(n_runs, T, n))
    for t in range(T):
        w[:, t + 1] = a * w[:, t] + b * xi[:, t]
    world = env["world_amp"] * w
    noise = rng.normal(0.0, env["noise_std"], size=(n_runs, T + 1, n))
    s = world + noise
    times = np.arange(-1, T)
    for ch, d in zip(env["self_ch"], env["delays"]):
        s[:, :, ch] += env["gain"] * u[:, pad + times - d]
    if parts:
        return u, s, world, noise
    return u, s


def true_theta(env, L):
    """各チャンネルの、窓の中の真の係数 (N, L + 1) と、窓の外に出た係数の二乗和 (N,)。"""
    th = np.zeros((env["n_ch"], L + 1))
    out = np.zeros(env["n_ch"])
    for ch, d in zip(env["self_ch"], env["delays"]):
        if d <= L:
            th[ch, d] += env["gain"]
        else:
            out[ch] += env["gain"] ** 2
    return th, out


def prediction_r2(theta_hat, env, L):
    """自分の模型による予測の良さ（実験者の目から見た量。真の分布での期待値）。

    R² = 1 − E[(s_i − θ̂_i・ũ)²] / Var(s_i)。指令は毎回独立なので
    E[(s_i − θ̂_i・ũ)²] = u_std² (|θ̂_i − θ_i|² + 窓の外の係数²) + c² + σ²。
    返り値：R² (..., N) と、上限（窓の中の自分の効果が分散に占める割合）(N,)
    """
    th, out = true_theta(env, L)
    us2 = env["u_std"] ** 2
    other = env["world_amp"] ** 2 + env["noise_std"] ** 2
    var_s = us2 * ((th ** 2).sum(axis=1) + out) + other
    err = us2 * (((theta_hat - th) ** 2).sum(axis=-1) + out) + other
    return 1.0 - err / var_s, us2 * (th ** 2).sum(axis=1) / var_s


def pad_for(env, L):
    return max(L + 1, max(env["delays"], default=0) + 1)


# ---------------------------------------------------------------- エージェント

def record_times(T, dense=None, n_log=None):
    """記録する時刻（それまでに使った観測の数 n）。"""
    dense = config.U5_REC_DENSE if dense is None else dense
    n_log = config.U5_REC_LOG if n_log is None else n_log
    a = np.arange(1, min(dense, T) + 1)
    b = np.round(np.logspace(np.log10(min(dense, T)), np.log10(T), n_log)).astype(int)
    return np.unique(np.concatenate([a, b]))


class Agent:
    def __init__(self, n_ch, L=None, phi_grid=None, theta_var=None, prior_self=None, prior_two=None,
                 var_mean_log10=None, var_sd_log10=None, newton_iter=None, batch=None):
        self.n_ch = n_ch
        self.L = config.U5_LAGS if L is None else L
        self.p = self.L + 1
        if phi_grid is None:
            m = config.U5_PHI_GRID_N
            phi_grid = (np.arange(m) + 0.5) / m
        self.phi_grid = np.asarray(phi_grid, dtype=float)
        self.phis = np.concatenate([[0.0], self.phi_grid])     # 0 番目が M1（φ = 0）
        self.theta_var = config.U5_THETA_PRIOR_VAR if theta_var is None else theta_var
        self.prior_self = config.U5_PRIOR_SELF if prior_self is None else prior_self
        self.prior_two = config.U5_PRIOR_TWO if prior_two is None else prior_two
        m10 = config.U5_LOG10_VAR_PRIOR_MEAN if var_mean_log10 is None else var_mean_log10
        s10 = config.U5_LOG10_VAR_PRIOR_SD if var_sd_log10 is None else var_sd_log10
        self.xmu, self.xsd = m10 * LN10, s10 * LN10             # x = ln v の事前の平均と標準偏差
        self.newton_iter = config.U5_NEWTON_ITER if newton_iter is None else newton_iter
        self.batch = config.U5_BATCH if batch is None else batch

    # -- 十分統計量
    def features(self, u, s, pad):
        """各時刻 t = 0..T−1 の z(t) = (u(t), ..., u(t−L−1), s(t), s(t−1))。(R, T, L + 2 + 2N)"""
        T = s.shape[1] - 1
        if pad < self.L + 1:
            raise ValueError("pad が窓より短い")
        idx = pad + np.arange(T)[:, None] - np.arange(self.L + 2)[None, :]
        v = u[:, idx]
        return np.concatenate([v, s[:, 1:], s[:, :-1]], axis=2)

    @staticmethod
    def grams(Z, rec_n):
        """記録の時刻 n ごとの G = Σ_{t<n} z(t) z(t)ᵀ。(R, len(rec_n), D, D)"""
        R, _, D = Z.shape
        out = np.empty((R, len(rec_n), D, D))
        G = np.zeros((R, D, D))
        prev = 0
        for k, n in enumerate(rec_n):
            seg = Z[:, prev:n]
            G += np.einsum("rtd,rte->rde", seg, seg)
            out[:, k] = G
            prev = n
        return out

    def _transformed(self, G):
        """φ ごとに、y = s(t) − φ s(t−1)、X = ũ(t) − φ ũ(t−1) の XᵀX, Xᵀy, yᵀy。"""
        p, N = self.p, self.n_ch
        q = self.L + 2
        iS = q + np.arange(N)
        iS1 = q + N + np.arange(N)
        ph = self.phis[None, :, None, None]
        A = G[:, None, 0:p, 0:p]
        B = G[:, None, 0:p, 1:p + 1]
        C = G[:, None, 1:p + 1, 1:p + 1]
        XtX = A - ph * (B + np.swapaxes(B, -1, -2)) + ph ** 2 * C
        a = G[:, None, 0:p][..., iS]
        b = G[:, None, 0:p][..., iS1]
        c = G[:, None, 1:p + 1][..., iS]
        d = G[:, None, 1:p + 1][..., iS1]
        Xty = a - ph * (b + c) + ph ** 2 * d
        ph2 = self.phis[None, :, None]
        yy = (G[:, None, iS, iS] - 2 * ph2 * G[:, None, iS, iS1] + ph2 ** 2 * G[:, None, iS1, iS1])
        return XtX, Xty, yy

    def _spectral(self, G):
        """固有分解：XᵀX = V diag(λ) Vᵀ、c = Vᵀ Xᵀy。

        「自分」の対数尤度を x = ln v の関数として書くための量を返す：
            ln a_j = ln(θ0 λ_j)（λ_j = 0 なら −∞）、b_j = c_j² / λ_j、R = yᵀy − Σ_j b_j（最小二乗の残り）
        """
        XtX, Xty, yy = self._transformed(G)
        lam, V = np.linalg.eigh(XtX)
        lam = np.clip(lam, 0.0, None)
        c = np.einsum("bjqp,bjqn->bjpn", V, Xty)                  # (B, J, p, N)
        tol = 1e-10 * np.maximum(lam.max(axis=-1, keepdims=True), 1.0)
        pos = lam > tol
        safe = np.where(pos, lam, 1.0)
        b = np.where(pos[..., None], c ** 2 / safe[..., None], 0.0)      # (B, J, p, N)
        b = np.moveaxis(b, 2, 3)                                         # (B, J, N, p)
        with np.errstate(divide="ignore"):
            la = np.where(pos, np.log(self.theta_var * safe), -np.inf)  # (B, J, p)
        R = np.clip(yy - b.sum(axis=-1), 0.0, None)                      # (B, J, N)
        return dict(lam=lam, V=V, c=c, yy=yy, la=la[:, :, None, :], b=b, R=R)

    def _logpost_x(self, x, n, R, la, b, p):
        """x = ln v の関数として、ln[ p(y | v) p(x) ] とその一階・二階の微分。

        ln p(y | v) = −½[ n ln 2π + (n − p) x + R e^{−x} + Σ_j ( ln(e^x + a_j) + b_j / (e^x + a_j) ) ]
        （「自分」の仮説。θ を積分した N(y; 0, v I + θ0 X Xᵀ) を固有分解で書いたもの。
          「自分でない」は p = 0、R = yᵀy とした同じ式。）
        """
        ex = np.exp(-x)
        if p > 0:
            w = _sigmoid(x[..., None] - la)                 # v / (v + a_j)
            bt = b * ex[..., None] * w                       # b_j / (v + a_j)
            s0 = (np.logaddexp(x[..., None], la) + bt).sum(axis=-1)
            s1 = (w - bt * w).sum(axis=-1)
            s2 = (w * (1.0 - w) - bt * w * (1.0 - 2.0 * w)).sum(axis=-1)
        else:
            s0 = s1 = s2 = 0.0
        ll = -0.5 * (n * LOG2PI + (n - p) * x + R * ex + s0)
        g = -0.5 * ((n - p) - R * ex + s1)
        h = -0.5 * (R * ex + s2)
        z = (x - self.xmu) / self.xsd
        lp = -0.5 * z * z - 0.5 * np.log(2 * np.pi * self.xsd ** 2)
        return ll + lp, g - z / self.xsd, h - 1.0 / self.xsd ** 2

    def _laplace(self, n, R, la, b, p, x0):
        """ラプラス近似：ln ∫ p(y | v) p(x) dx ≈ f(x*) + ½ ln 2π − ½ ln(−f''(x*))。返り値 (近似の値, x*, |f'(x*)|)。"""
        lo, hi = self.xmu - 12 * self.xsd, self.xmu + 12 * self.xsd
        x = np.clip(x0, lo, hi)
        for _ in range(self.newton_iter):
            f, g, h = self._logpost_x(x, n, R, la, b, p)
            newton = -g / np.where(h < 0, h, -1.0)
            step = np.where(h < 0, newton, np.sign(g))
            x = np.clip(x + np.clip(step, -2.0, 2.0), lo, hi)
        f, g, h = self._logpost_x(x, n, R, la, b, p)
        h = np.minimum(h, -1e-12)
        return f + 0.5 * LOG2PI - 0.5 * np.log(-h), x, np.abs(g)

    def _evidence_v(self, n, R, la, b, p, yy):
        """v を積分した対数の証拠。二つの出発点（最小二乗の残りから／全体の分散から）でニュートン法を走らせ、
        別々の山に着いたら二つを足し合わせ、同じ山なら一つを使う。"""
        nb = np.broadcast_to(n, R.shape)
        xb = np.log(np.maximum(yy, 1e-300) / nb)
        fit = (nb > p) & (R > 0)          # 観測が係数より多く、残りがあるときだけ、残りから始める
        xa = np.where(fit, np.log(np.maximum(R, 1e-300) / np.maximum(nb - p, 1.0)), xb)
        xa = np.clip(xa, xb - 30.0, xb)
        fa, xsa, ga = self._laplace(n, R, la, b, p, xa)
        if p == 0:
            return fa, xsa, ga
        fb, xsb, gb = self._laplace(n, R, la, b, p, xb)
        same = np.abs(xsa - xsb) < 1e-3
        conv = (ga < 1e-6) & (gb < 1e-6)
        both = np.logaddexp(fa, fb)
        best = np.where(fa >= fb, fa, fb)
        ev = np.where(same | ~conv, best, both)
        xs = np.where(fa >= fb, xsa, xsb)
        gmax = np.where(fa >= fb, ga, gb)
        return ev, xs, gmax

    def _evaluate_batch(self, G, n, want_theta=True):
        sp = self._spectral(G)
        nn = n[:, None, None]
        en, _, gn = self._evidence_v(nn, sp["yy"], None, None, 0, sp["yy"])        # (B, J+1, N)
        es, xs, gs = self._evidence_v(nn, sp["R"], sp["la"], sp["b"], self.p, sp["yy"])
        J = len(self.phi_grid)
        lse = _logsumexp
        out = {"none1": en[:, 0], "self1": es[:, 0],
               "none2": lse(en[:, 1:], axis=1) - np.log(J),
               "self2": lse(es[:, 1:], axis=1) - np.log(J),
               "grad_max": np.maximum(gn.max(axis=(1, 2)), gs.max(axis=(1, 2)))}
        # M2 のもとでの φ の事後平均（自分かどうかも積分する）
        ps = self.prior_self
        ech = np.logaddexp(np.log1p(-ps) + en[:, 1:], np.log(ps) + es[:, 1:])     # (B, J, N)
        wj = np.exp(ech - lse(ech, axis=1)[:, None])
        out["phi2"] = np.einsum("bjn,j->bn", wj, self.phi_grid)
        if not want_theta:
            return out
        # 「自分」の仮説のもとでの θ の事後平均：v は山の頂上 v* に置き、φ は事後の重みで混ぜる
        vstar = np.exp(xs)                                                             # (B, J+1, N)
        den = sp["lam"][:, :, :, None] + vstar[:, :, None, :] / self.theta_var          # (B, J+1, p, N)
        th_all = np.einsum("bjqp,bjpn->bjnq", sp["V"], sp["c"] / den)                  # (B, J+1, N, p)
        out["theta1"] = th_all[:, 0]
        w2 = np.exp(es[:, 1:] - lse(es[:, 1:], axis=1)[:, None])                        # (B, J, N)
        out["theta2"] = np.einsum("bjn,bjnq->bnq", w2, th_all[:, 1:])
        return out

    def evaluate(self, Gs, rec_n, want_theta=True):
        """Gs (R, n_rec, D, D) → 各量 (R, n_rec, ...)。want_theta=False なら θ の事後平均を省く。"""
        R, nr, D, _ = Gs.shape
        Gf = Gs.reshape(R * nr, D, D)
        nf = np.tile(np.asarray(rec_n, dtype=float), R)
        parts = []
        for i in range(0, R * nr, self.batch):
            parts.append(self._evaluate_batch(Gf[i:i + self.batch], nf[i:i + self.batch], want_theta))
        return {k: np.concatenate([p[k] for p in parts]).reshape((R, nr) + parts[0][k].shape[1:])
                for k in parts[0]}

    def beliefs(self, ev):
        """証拠から、エージェントの信念をつくる。

        log_bf  ：「自分と世界」M2 と「自分だけ」M1 の証拠の比の対数（自他の区別の強さ）
        p_two   ：P(M2 | データ)
        p_self  ：P(チャンネル i が自分の指令で動く | データ)（M1 と M2 を事後の重みで混ぜる）
        theta   ：θ_i の事後平均（自分でない場合の θ = 0 も混ぜる）
        """
        ps, pt = self.prior_self, self.prior_two
        lo_s = np.log(ps) - np.log1p(-ps)
        ev1 = np.logaddexp(np.log1p(-ps) + ev["none1"], np.log(ps) + ev["self1"])
        ev2 = np.logaddexp(np.log1p(-ps) + ev["none2"], np.log(ps) + ev["self2"])
        log_bf = (ev2 - ev1).sum(axis=-1)
        p_two = _sigmoid(log_bf + np.log(pt) - np.log1p(-pt))
        ps1 = _sigmoid(lo_s + ev["self1"] - ev["none1"])
        ps2 = _sigmoid(lo_s + ev["self2"] - ev["none2"])
        pt_ = p_two[..., None]
        p_self = (1 - pt_) * ps1 + pt_ * ps2
        out = dict(log_bf=log_bf, p_two=p_two, p_self=p_self, p_self1=ps1, p_self2=ps2,
                   phi2=ev["phi2"], grad_max=ev["grad_max"])
        if "theta1" in ev:
            out["theta"] = ((1 - pt_) * ps1)[..., None] * ev["theta1"] + (pt_ * ps2)[..., None] * ev["theta2"]
        return out

    def run(self, u, s, pad, rec_n, want_theta=True):
        Z = self.features(u, s, pad)
        ev = self.evaluate(self.grams(Z, rec_n), rec_n, want_theta)
        return self.beliefs(ev)


def _logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def first_settled(p, rec_n, thresh, above=True):
    """p (..., n_rec) が、ある記録の時刻から最後まで thresh の正しい側にとどまる、その最初の時刻。

    とどまらなければ nan（T までに見分けられなかった）。
    """
    ok = p >= thresh if above else p <= thresh
    tail = np.flip(np.logical_and.accumulate(np.flip(ok, axis=-1), axis=-1), axis=-1)
    has = tail[..., -1]
    first = np.argmax(tail, axis=-1)
    out = np.asarray(rec_n, dtype=float)[first]
    return np.where(has, out, np.nan)


# ---------------------------------------------------------------- 点検

def _direct_loglik(X, y, v, t0, self_model):
    n = len(y)
    cov = v * np.eye(n) + (t0 * X @ X.T if self_model else 0.0)
    _, ld = np.linalg.slogdet(cov)
    return -0.5 * (n * LOG2PI + ld + y @ np.linalg.solve(cov, y))


def selftest():
    rng = np.random.default_rng(config.U5_SELFTEST_SEED)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'NG'}] {name} {detail}")

    # 1. 世界の生成：自分のチャンネル = g u(t−d) + 世界 + 雑音、ほかは 世界 + 雑音
    env = default_env()
    L = 3
    pad = pad_for(env, L)
    u, s, world, noise = simulate(rng, 2, 50, env, pad, parts=True)
    rest = s - world - noise
    exp_rest = np.zeros_like(rest)
    for ch, d in zip(env["self_ch"], env["delays"]):
        exp_rest[:, :, ch] = env["gain"] * u[:, pad + np.arange(-1, 50) - d]
    check("感覚 = 指令の効果 + 世界 + 雑音", np.allclose(rest, exp_rest))

    # 2. 世界の揺れ：OU（分散 1、一歩の相関 exp(−1/τ)）
    env_w = dict(env, self_ch=[], delays=[], noise_std=0.0, world_amp=1.0)
    _, sw = simulate(rng, 50, 4000, env_w, 1)
    x = sw[:, 1:, :]
    v = x.var()
    r1 = (x[:, 1:] * x[:, :-1]).mean() / (x * x).mean()
    a = np.exp(-1 / env["world_tau"])
    check("世界の揺れの分散 ≈ 1", abs(v - 1) < 0.05, f"({v:.3f})")
    check("世界の揺れの一歩の相関 ≈ exp(−1/τ)", abs(r1 - a) < 0.01, f"({r1:.4f} / {a:.4f})")

    # 3. 十分統計量から書いた、v を固定した対数尤度 = 直接の多変量正規の対数密度（いくつかの φ と v で）
    phis = [0.0, 0.3, 0.9]
    ag = Agent(env["n_ch"], L=L, phi_grid=phis[1:], theta_var=0.7)
    T = 23
    u, s = simulate(rng, 1, T, env, pad)
    Z = ag.features(u, s, pad)
    rec = np.array([3, 12, T])
    Gs = ag.grams(Z, rec)
    N = env["n_ch"]
    for k, n in enumerate(rec):
        sp = ag._spectral(Gs[:, k])
        maxerr = 0.0
        for xv in np.log([1e-3, 0.2, 3.0]):
            xx = np.full((1, len(phis), N), xv)
            f_none, _, _ = ag._logpost_x(xx, float(n), sp["yy"], None, None, 0)
            f_self, _, _ = ag._logpost_x(xx, float(n), sp["R"], sp["la"], sp["b"], ag.p)
            z = (xv - ag.xmu) / ag.xsd
            lp = -0.5 * z * z - 0.5 * np.log(2 * np.pi * ag.xsd ** 2)
            for j, ph in enumerate(phis):
                X = Z[0, :n, 0:L + 1] - ph * Z[0, :n, 1:L + 2]
                for i in range(N):
                    y = Z[0, :n, L + 2 + i] - ph * Z[0, :n, L + 2 + N + i]
                    for sm, f in ((False, f_none), (True, f_self)):
                        d = _direct_loglik(X, y, np.exp(xv), ag.theta_var, sm)
                        maxerr = max(maxerr, abs(d - (f[0, j, i] - lp)))
        check(f"v を固定した対数尤度（n = {n}）が直接の計算と一致", maxerr < 1e-6, f"(最大の差 {maxerr:.1e})")

    # 4. x = ln v についての一階・二階の微分（差分と比べる）
    sp = ag._spectral(Gs[:, 2])
    xx = np.full((1, len(phis), N), np.log(0.4))
    e = 1e-5
    f0, g0, h0 = ag._logpost_x(xx, float(T), sp["R"], sp["la"], sp["b"], ag.p)
    fp, gp, _ = ag._logpost_x(xx + e, float(T), sp["R"], sp["la"], sp["b"], ag.p)
    fm, gm, _ = ag._logpost_x(xx - e, float(T), sp["R"], sp["la"], sp["b"], ag.p)
    check("一階の微分", np.allclose(g0, (fp - fm) / (2 * e), rtol=1e-5, atol=1e-5))
    check("二階の微分", np.allclose(h0, (gp - gm) / (2 * e), rtol=1e-5, atol=1e-5))

    # 4'. 証拠 = 予測の当たりの積：v を固定すると、n 個の証拠 − (n−1) 個の証拠 = n 番目の一歩先の予測の対数密度
    ph, vv, i = 0.3, 0.2, 1
    agp = Agent(N, L=L, phi_grid=[ph], theta_var=0.7)
    G2 = agp.grams(Z, np.array([T - 1, T]))
    xx = np.full((1, 2, N), np.log(vv))
    lls = []
    for k, n in enumerate([T - 1, T]):
        sp = agp._spectral(G2[:, k])
        f, _, _ = agp._logpost_x(xx, float(n), sp["R"], sp["la"], sp["b"], agp.p)
        lls.append(f[0, 1, i])                       # 事前 p(x) は両方に同じだけ入るので、差で消える
    X = Z[0, :T, 0:L + 1] - ph * Z[0, :T, 1:L + 2]
    y = Z[0, :T, L + 2 + i] - ph * Z[0, :T, L + 2 + N + i]
    Xp, yp = X[:-1], y[:-1]
    S = np.linalg.inv(Xp.T @ Xp / vv + np.eye(L + 1) / agp.theta_var)
    m = S @ Xp.T @ yp / vv
    pv = vv + X[-1] @ S @ X[-1]
    lpred = -0.5 * (LOG2PI + np.log(pv) + (y[-1] - X[-1] @ m) ** 2 / pv)
    check("証拠の増分 = 一歩先の予測の対数密度（v を固定）", abs((lls[1] - lls[0]) - lpred) < 1e-8,
          f"(差 {abs((lls[1] - lls[0]) - lpred):.1e})")

    # 5. v の積分：ラプラス近似と、細かい格子での数値積分の差
    xs_fine = np.linspace(ag.xmu - 12 * ag.xsd, ag.xmu + 12 * ag.xsd, 40001)
    dx = xs_fine[1] - xs_fine[0]
    T5 = 600
    for noise_std in (0.5, 0.01):
        env5 = dict(env, noise_std=noise_std)
        u, s = simulate(rng, 1, T5, env5, pad)
        Z5 = ag.features(u, s, pad)
        rec5 = np.array([5, 10, 30, 100, T5])
        G5 = ag.grams(Z5, rec5)
        worst = {}
        for k, n in enumerate(rec5):
            sp = ag._spectral(G5[:, k])
            ev = ag._evaluate_batch(G5[:, k], np.array([float(n)]), want_theta=False)
            nn = np.array([float(n)])[:, None, None]
            es, _, _ = ag._evidence_v(nn, sp["R"], sp["la"], sp["b"], ag.p, sp["yy"])
            en, _, _ = ag._evidence_v(nn, sp["yy"], None, None, 0, sp["yy"])
            err = 0.0
            for j in range(len(phis)):
                for i in range(N):
                    xx = xs_fine
                    fs, _, _ = ag._logpost_x(xx, float(n), np.full_like(xx, sp["R"][0, j, i]),
                                             np.broadcast_to(sp["la"][0, j, 0], xx.shape + (ag.p,)),
                                             np.broadcast_to(sp["b"][0, j, i], xx.shape + (ag.p,)), ag.p)
                    fn, _, _ = ag._logpost_x(xx, float(n), np.full_like(xx, sp["yy"][0, j, i]), None, None, 0)
                    qs = _logsumexp(fs, 0) + np.log(dx)
                    qn = _logsumexp(fn, 0) + np.log(dx)
                    err = max(err, abs(qs - es[0, j, i]), abs(qn - en[0, j, i]))
            worst[int(n)] = err
            del ev
        txt = "、".join(f"n={k}: {v_:.3f}" for k, v_ in worst.items())
        check(f"ラプラス近似の誤差（雑音 {noise_std}、対数の証拠の差の最大）",
              all(v_ < 0.1 for k, v_ in worst.items() if k >= 30), f"({txt})")

    # 6. θ の事後平均（φ の格子が一点、v = v* のとき）= (XᵀX + v*/θ0 I)⁻¹ Xᵀy
    ag1 = Agent(N, L=L, phi_grid=[0.3], theta_var=0.7)
    G6 = ag1.grams(Z, np.array([T]))
    ev = ag1.evaluate(G6, np.array([T]))
    sp = ag1._spectral(G6[:, 0])
    _, xs, _ = ag1._evidence_v(np.array([float(T)])[:, None, None], sp["R"], sp["la"], sp["b"], ag1.p, sp["yy"])
    i = 1
    for j, key in ((1, "theta2"), (0, "theta1")):
        ph = ag1.phis[j]
        X = Z[0, :T, 0:L + 1] - ph * Z[0, :T, 1:L + 2]
        y = Z[0, :T, L + 2 + i] - ph * Z[0, :T, L + 2 + N + i]
        vs = np.exp(xs[0, j, i])
        th = np.linalg.solve(X.T @ X + vs / ag1.theta_var * np.eye(L + 1), X.T @ y)
        check(f"θ の事後平均（{'M2' if j else 'M1'}、一点の格子）", np.allclose(ev[key][0, 0, i], th))

    # 7. 信念：確率は [0, 1]、P(self) は M1 と M2 のあいだ、ニュートン法は収束している
    ag2 = Agent(N, L=L)
    u, s = simulate(rng, 3, 300, env, pad)
    rec3 = record_times(300, dense=10, n_log=10)
    b = ag2.run(u, s, pad, rec3)
    inr = all(((b[k] >= 0) & (b[k] <= 1)).all() for k in ("p_two", "p_self", "p_self1", "p_self2"))
    check("確率はどれも [0, 1]", inr)
    lo = np.minimum(b["p_self1"], b["p_self2"]) - 1e-12
    hi = np.maximum(b["p_self1"], b["p_self2"]) + 1e-12
    check("P(self) は M1 と M2 のあいだ", ((b["p_self"] >= lo) & (b["p_self"] <= hi)).all())
    check("ニュートン法の収束（頂上での傾きの最大）", b["grad_max"].max() < 1e-6, f"({b['grad_max'].max():.1e})")

    # 8. 予測の良さ：真の θ なら上限に一致、θ̂ = 0 なら 0
    th, _ = true_theta(env, L)
    r2, ceil = prediction_r2(th, env, L)
    r20, _ = prediction_r2(np.zeros_like(th), env, L)
    check("真の θ での R² = 上限", np.allclose(r2, ceil))
    check("θ̂ = 0 での R² = 0", np.allclose(r20, 0.0))

    # 9. 「最後までとどまる最初の時刻」
    p = np.array([[0.2, 0.96, 0.5, 0.97, 0.99], [0.2, 0.3, 0.4, 0.5, 0.96], [0.99, 0.99, 0.99, 0.99, 0.94]])
    got = first_settled(p, np.array([1, 2, 3, 4, 5]), 0.95)
    check("見分けた時刻の定義", np.allclose(got[:2], [4, 5]) and np.isnan(got[2]), str(got))

    print("selftest:", "すべて通った" if ok else "通らないものがある")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)

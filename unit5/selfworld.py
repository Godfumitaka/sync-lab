"""単元五：自分の指令の発見（模型の本体と点検）。

二つの部分からなる。

感覚を作る側（Sensorium、エージェントには見えない）
    チャンネル i の感覚  s_i(t) = a_i · u(t − d_i) ＋ g_i · z_i(t) ＋ σ · ε_i(t)
        u(t)   ：エージェントの指令。毎歩でたらめ（独立な正規分布、モーター・バブリング）
        a_i, d_i：指令の効きと遅れ（一部のチャンネルだけ a_i ≠ 0）
        z_i(t) ：世界の揺れ。チャンネルごとに独立な AR(1)、分散 1、一歩ごとの相関 ρ
        ε_i(t) ：雑音。独立な標準正規分布
    身体の揺れ（段階2）、まねる相手（段階3）は、その段階で書き足す。

エージェント（Agent）
    知っているのは、自分の指令の控え U(t) = (u(t), u(t−1), …, u(t−L)) と、感覚 s(t) だけ。
    チャンネルごとに二つの見方を比べる。
        H_自：s_i(t) = w_i · U(t) ＋ e(t)   （指令の控えから感覚を予測する、小さな線形の模型）
        H_0 ：s_i(t) = e(t)                 （指令とは関係がない）
        e(t) は平均 0、分散 σ_e² の独立な正規分布とみなす（σ_e² は未知）。
    事前分布：w_i | σ_e² ~ N(0, σ_e² τ² I)、σ_e² にはジェフリーズの事前分布（∝ 1/σ_e²）。
    このとき、証拠の比（ベイズ因子）は、十分統計量 XᵀX、Xᵀy、yᵀy、n から閉じた式で出る：
        log BF = −½ log|I + τ² XᵀX| + (n/2) [log yᵀy − log Q]
        Q = yᵀy − Xᵀyᵀ (XᵀX + τ⁻² I)⁻¹ Xᵀy
    「自分のものである度合い」= H_自 の事後確率 = σ(logit(事前) + log BF)、事前は 1/2。
    「自分の模型による予測」= 度合い × (w_i の事後平均) · U(t)（H_0 は 0 を予測する）。
    毎歩、感覚を見る前に予測し、見てから十分統計量を足す。

論文に書かれていない選択は ASSUMPTIONS.md にある。

使い方
    python3 selfworld.py --selftest    規則の点検（結果の統計は出さない）
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402


def _per_channel(x, n_runs, n_channels, dtype=float):
    """値・チャンネルごとの並び・(走行, チャンネル) の配列のどれでも、(走行, チャンネル) にそろえる。"""
    return np.array(np.broadcast_to(np.asarray(x, dtype), (n_runs, n_channels)))


def expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Sensorium:
    """感覚を作る側。多数の走行を配列でまとめて進める（走行どうしは独立）。"""

    def __init__(self, n_runs, rng,
                 self_gains=config.U5_SELF_GAINS, self_delays=config.U5_SELF_DELAYS,
                 world_gains=config.U5_WORLD_GAINS, noise_sd=config.U5_NOISE_SD,
                 world_rho=config.U5_WORLD_RHO, command_sd=config.U5_COMMAND_SD,
                 n_channels=config.U5_N_CHANNELS):
        R, N = n_runs, n_channels
        self.n_runs, self.n_channels = R, N
        self.a = _per_channel(self_gains, R, N)
        self.d = _per_channel(self_delays, R, N, dtype=np.int64)
        self.g = _per_channel(world_gains, R, N)
        self.sd = _per_channel(noise_sd, R, N)
        self.rho = float(world_rho)
        self.command_sd = float(command_sd)
        self.u_hist = np.zeros((R, int(self.d.max()) + 1))   # u_hist[:, k] = u(t − k)。始まる前の指令は 0
        self.z = rng.standard_normal((R, N))                  # 世界の揺れは定常な状態から始める

    def step(self, rng):
        """一歩。返り値は (指令 u(t)、感覚 s(t))。"""
        R, N = self.n_runs, self.n_channels
        u = self.command_sd * rng.standard_normal(R)
        self.u_hist[:, 1:] = self.u_hist[:, :-1]
        self.u_hist[:, 0] = u
        self.z = self.rho * self.z + np.sqrt(1.0 - self.rho ** 2) * rng.standard_normal((R, N))
        own = self.a * np.take_along_axis(self.u_hist, self.d, axis=1)
        s = own + self.g * self.z + self.sd * rng.standard_normal((R, N))
        return u, s


class Agent:
    """チャンネルごとに H_自（指令の控えから予測できる）と H_0（できない）を比べるエージェント。"""

    def __init__(self, n_runs, n_channels=config.U5_N_CHANNELS, lags=config.U5_LAGS,
                 tau2=config.U5_TAU2, prior_self=config.U5_PRIOR_SELF):
        R, N, k = n_runs, n_channels, lags + 1
        self.n_runs, self.n_channels, self.k = R, N, k
        self.tau2 = float(tau2)
        self.prior_logit = np.log(prior_self / (1.0 - prior_self))
        self.U = np.zeros((R, k))           # 指令の控え：U[:, l] = u(t − l)
        self.XtX = np.zeros((R, k, k))      # 指令の控えはどのチャンネルにも共通なので、XᵀX も共通
        self.Xty = np.zeros((R, k, N))
        self.yy = np.zeros((R, N))
        self.n = 0

    def efference(self, u):
        """指令の控えを一歩進め、U(t) を返す。"""
        self.U[:, 1:] = self.U[:, :-1]
        self.U[:, 0] = u
        return self.U.copy()

    def posterior(self):
        """今までの感覚から、(log BF、H_自 の重みの事後平均) を返す。形は (R, N) と (R, k, N)。"""
        R, N, k = self.n_runs, self.n_channels, self.k
        if self.n == 0:
            return np.zeros((R, N)), np.zeros((R, k, N))
        A = self.XtX + np.eye(k) / self.tau2
        m = np.linalg.solve(A, self.Xty)                         # (XᵀX + τ⁻² I)⁻¹ Xᵀy
        Q = self.yy - np.einsum("rkn,rkn->rn", self.Xty, m)
        Q = np.maximum(Q, 1e-300)
        _, logdetA = np.linalg.slogdet(A)
        logdet = logdetA + k * np.log(self.tau2)                 # log|I + τ² XᵀX|
        log_bf = -0.5 * logdet[:, None] + 0.5 * self.n * (np.log(self.yy) - np.log(Q))
        return log_bf, m

    def ownership(self, log_bf):
        """自分のものである度合い（H_自 の事後確率）。"""
        return expit(self.prior_logit + log_bf)

    @staticmethod
    def predict(U, own, m):
        """自分の模型による予測：度合い × (w の事後平均)·U(t)。H_0 は 0 を予測する。"""
        return own * np.einsum("rk,rkn->rn", U, m)

    def observe(self, U, s):
        self.XtX += U[:, :, None] * U[:, None, :]
        self.Xty += U[:, :, None] * s[:, None, :]
        self.yy += s * s
        self.n += 1


def run(sens, agent, T, rng, record_at=(), keep_series=False, decide=config.U5_DECIDE):
    """T 歩すすめる。

    record_at   ：度合いと log BF を記録する「見た感覚の数」n の集まり（0 ≤ n ≤ T）
    keep_series ：毎歩の感覚と予測を残す（一本の走行の図のため）
    返り値の dict
        rec_n, rec_own, rec_logbf ：記録した n と、そのときの度合い・log BF（形は (記録点, R, N)）
        settle    ：見分けが最後まで正しく保たれるようになった n（T までに保たれなければ −1）
        first     ：初めて正しく見分けた n（T までになければ −1）
        own_T, logbf_T, m_T ：T 歩のあとの度合い、log BF、重みの事後平均
        s, s_hat  ：（keep_series のとき）毎歩の感覚と、見る前の予測（形は (T, R, N)）
    「正しく見分けた」：a_i ≠ 0 のチャンネルは度合い ≥ decide、a_i = 0 のチャンネルは度合い ≤ 1 − decide。
    """
    R, N = agent.n_runs, agent.n_channels
    record_at = set(int(n) for n in record_at)
    is_self = sens.a != 0
    last_wrong = np.zeros((R, N), dtype=np.int64)
    first = np.full((R, N), -1, dtype=np.int64)
    rec_n, rec_own, rec_logbf = [], [], []
    if keep_series:
        s_all = np.empty((T, R, N))
        sh_all = np.empty((T, R, N))

    def book(n, log_bf):
        own = agent.ownership(log_bf)
        correct = np.where(is_self, own >= decide, own <= 1.0 - decide)
        last_wrong[~correct] = n
        first[(first < 0) & correct] = n
        if n in record_at:
            rec_n.append(n)
            rec_own.append(own)
            rec_logbf.append(log_bf)
        return own

    for t in range(T):
        u, s = sens.step(rng)
        U = agent.efference(u)
        log_bf, m = agent.posterior()          # t 個の感覚から（今の感覚はまだ見ていない）
        own = book(t, log_bf)
        if keep_series:
            s_all[t] = s
            sh_all[t] = agent.predict(U, own, m)
        agent.observe(U, s)
    log_bf, m = agent.posterior()
    own = book(T, log_bf)

    settle = np.where(last_wrong < T, last_wrong + 1, -1)
    out = dict(rec_n=np.array(rec_n), rec_own=np.array(rec_own), rec_logbf=np.array(rec_logbf),
               settle=settle, first=first, own_T=own, logbf_T=log_bf, m_T=m, is_self=is_self)
    if keep_series:
        out["s"] = s_all
        out["s_hat"] = sh_all
    return out


def selftest():
    """規則の点検。結果の統計（見分けるまでの時間など）は出さない。"""
    rng = np.random.default_rng(0)

    # 1. 雑音も世界の揺れもなければ、感覚は a_i · u(t − d_i) にちょうど等しい（始まる前の指令は 0）
    a, d = [1.0, -0.5, 2.0, 0.0], [0, 1, 3, 0]
    sens = Sensorium(3, rng, self_gains=a, self_delays=d, world_gains=0.0, noise_sd=0.0, n_channels=4)
    us, ss = [], []
    for _ in range(50):
        u, s = sens.step(rng)
        us.append(u)
        ss.append(s)
    us, ss = np.array(us), np.array(ss)                  # (T, R), (T, R, N)
    for i in range(4):
        lagged = np.vstack([np.zeros((d[i], 3)), us[:50 - d[i]]])
        assert np.allclose(ss[:, :, i], a[i] * lagged)

    # 2. 世界の揺れは分散 1、一歩ごとの相関 ρ の AR(1)（生成の点検。多数の走行の標本で許容幅つき）
    sens = Sensorium(4000, rng, self_gains=0.0, self_delays=0, world_gains=1.0, noise_sd=0.0, n_channels=2)
    z0 = sens.z.copy()
    _, s1 = sens.step(rng)
    assert abs(np.var(z0) - 1.0) < 0.05 and abs(np.var(s1) - 1.0) < 0.05
    rho_hat = np.mean(z0 * s1) / np.sqrt(np.var(z0) * np.var(s1))
    assert abs(rho_hat - config.U5_WORLD_RHO) < 0.02

    # 3. 指令の控え U(t) = (u(t), u(t−1), …, u(t−L))、始まる前は 0
    ag = Agent(2, n_channels=1, lags=3)
    seq = rng.standard_normal((6, 2))
    for t in range(6):
        U = ag.efference(seq[t])
        want = np.array([seq[t - l] if t - l >= 0 else np.zeros(2) for l in range(4)]).T
        assert np.allclose(U, want)

    # 4. 十分統計量の足し方と、閉じた式の log BF が、n×n の行列で直接計算した値と一致する
    R, N, T = 2, 3, 40
    sens = Sensorium(R, rng, self_gains=[1.0, 0.0, 0.3], self_delays=[1, 0, 2],
                     world_gains=[0.0, 1.0, 0.5], noise_sd=0.7, n_channels=N)
    ag = Agent(R, n_channels=N, lags=2, tau2=0.8)
    Xs, Ys = [], []
    for _ in range(T):
        u, s = sens.step(rng)
        U = ag.efference(u)
        ag.observe(U, s)
        Xs.append(U)
        Ys.append(s)
    X, Y = np.array(Xs), np.array(Ys)                    # (T, R, k), (T, R, N)
    log_bf, m = ag.posterior()
    for r in range(R):
        Xr = X[:, r, :]
        assert np.allclose(ag.XtX[r], Xr.T @ Xr)
        C = np.eye(T) + ag.tau2 * Xr @ Xr.T
        _, logdetC = np.linalg.slogdet(C)
        for i in range(N):
            y = Y[:, r, i]
            Qd = y @ np.linalg.solve(C, y)
            direct = -0.5 * logdetC + 0.5 * T * (np.log(y @ y) - np.log(Qd))
            assert np.isclose(log_bf[r, i], direct, rtol=1e-9, atol=1e-9)
            md = np.linalg.solve(Xr.T @ Xr + np.eye(Xr.shape[1]) / ag.tau2, Xr.T @ y)
            assert np.allclose(m[r, :, i], md)

    # 5. 閉じた式は「σ_e² をジェフリーズの事前分布で積分した周辺尤度の比」に等しい（log σ² の格子で数値積分）
    Xr, y = X[:, 0, :], Y[:, 0, 0]
    C1 = np.eye(T) + ag.tau2 * Xr @ Xr.T
    _, ld1 = np.linalg.slogdet(C1)
    q1, q0 = y @ np.linalg.solve(C1, y), y @ y
    logv = np.linspace(-12.0, 12.0, 200001)
    h = logv[1] - logv[0]

    def log_marg(ld, q):   # ∫ N(y; 0, v C) dv/v = ∫ N(y; 0, v C) d(log v)
        f = -0.5 * T * (np.log(2 * np.pi) + logv) - 0.5 * ld - 0.5 * q * np.exp(-logv)
        mx = f.max()
        return mx + np.log(np.sum(np.exp(f - mx)) * h)
    assert np.isclose(log_marg(ld1, q1) - log_marg(0.0, q0), log_bf[0, 0], atol=1e-6)

    # 6. 度合いは事前 1/2 から始まり、log BF = 0 なら 1/2 のまま
    ag = Agent(2, n_channels=2)
    lb, m = ag.posterior()
    assert np.allclose(ag.ownership(lb), 0.5) and np.allclose(m, 0.0)

    # 7. run の記録：n = 0 は事前（度合い 1/2）なので、どのチャンネルも「まだ見分けていない」
    sens = Sensorium(2, rng)
    ag = Agent(2)
    out = run(sens, ag, 20, rng, record_at=[0, 20], keep_series=True)
    assert list(out["rec_n"]) == [0, 20] and np.allclose(out["rec_own"][0], 0.5)
    assert out["s"].shape == (20, 2, config.U5_N_CHANNELS)
    assert np.all(out["settle"] != 0) and np.all(out["first"] != 0)
    assert ag.n == 20
    print("selftest: すべての点検を通った")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()

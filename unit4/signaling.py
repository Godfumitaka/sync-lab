"""単元四：合図ゲームの壺の模型（Argiento, Pemantle, Skyrms, Volkov 2009, §1.3–1.4）。

論文の規則とこのコードの対応は CORRESPONDENCE.md に、論文に書かれていない選択は ASSUMPTIONS.md にある。

壺の中身
    sender[r, i, x]   ：走行 r の送り手の、状態 i の壺に入っている合図 x の玉の数（論文の V(n, i, x)、i ∈ {1,2}, x ∈ {A,B}）
    receiver[r, x, j] ：走行 r の受け手の、合図 x の壺に入っている行為 j の玉の数（論文の V(n, x, j)）
添字は 0 から数える（論文の状態 1, 2 → 0, 1、合図 A, B → 0, 1、行為 1, 2 → 0, 1）。
多数の走行を配列でまとめて進めるが、走行どうしは互いに独立で、規則は一本ずつ走らせるのと同じ。

使い方
    python3 signaling.py --selftest    規則の点検（結果の統計は出さない）
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402


def draw(weights, u):
    """各行の重みに比例して添字を一つ選ぶ。u は一様乱数（行ごとに一つ）。

    u·(行の和) が累積和を初めて下回る添字を返す。二択なら「u < w0/(w0+w1) なら 0」と同じ（論文 §1.4 の S_n, R_n の作り方）。
    """
    cum = np.cumsum(weights, axis=1)
    return (u[:, None] * cum[:, -1:] >= cum).sum(axis=1)


class SignalingUrns:
    def __init__(self, n_runs, n_states=config.U4_N_STATES, n_signals=config.U4_N_SIGNALS,
                 n_acts=config.U4_N_ACTS, state_probs=config.U4_STATE_PROBS,
                 init_balls=config.U4_INIT_BALLS, reinforce=config.U4_REINFORCE):
        self.n_runs = n_runs
        self.state_cum = np.cumsum(np.asarray(state_probs, float))
        self.reinforce = reinforce
        self.sender = np.full((n_runs, n_states, n_signals), float(init_balls))
        self.receiver = np.full((n_runs, n_signals, n_acts), float(init_balls))
        self.wins = np.zeros(n_runs, dtype=np.int64)   # 論文の win_n
        self.n = 0

    def step(self, rng):
        """一回のやりとり。返り値は (状態, 合図, 行為, 勝ち)。"""
        r = np.arange(self.n_runs)
        u1, u2, u3 = rng.random((3, self.n_runs))                    # 論文の U_{n,1}, U_{n,2}, U_{n,3}
        state = np.searchsorted(self.state_cum, u1, side="right")    # Nature の手 N_n
        state = np.minimum(state, len(self.state_cum) - 1)
        signal = draw(self.sender[r, state], u2)                     # 送り手の手 S_n
        act = draw(self.receiver[r, signal], u3)                     # 受け手の手 R_n
        win = act == state                                           # 勝ち：R_n = N_n
        rw = r[win]
        self.sender[rw, state[win], signal[win]] += self.reinforce   # 勝ったときだけ、引いた玉の写しを足す
        self.receiver[rw, signal[win], act[win]] += self.reinforce
        self.wins += win
        self.n += 1
        return state, signal, act, win


def selftest():
    """規則の点検。結果の統計（成功率や落ち着き先）は出さない。"""
    rng = np.random.default_rng(0)
    m = SignalingUrns(n_runs=200)
    tot_s0, tot_r0 = m.sender.sum(), m.receiver.sum()
    total_wins = 0
    for _ in range(2000):
        s, x, a, w = m.step(rng)
        total_wins += w.sum()
        # 論文 §1.4 の観察：初期が対称なら V(n,i,x) = V(n,x,i)（送り手の 状態i・合図x と、受け手の 合図x・行為i）
        assert np.array_equal(m.sender, np.transpose(m.receiver, (0, 2, 1)))
    # 勝つたびに、送り手と受け手の玉がちょうど一つずつ増え、負けでは増えない
    assert m.sender.sum() - tot_s0 == total_wins
    assert m.receiver.sum() - tot_r0 == total_wins
    assert np.array_equal(m.wins.sum(), total_wins)
    # 二択の draw が「u < w0/(w0+w1) なら 0」と一致する
    w = rng.random((10000, 2)) + 0.01
    u = rng.random(10000)
    assert np.array_equal(draw(w, u), (u >= w[:, 0] / w.sum(axis=1)).astype(int))
    # 状態は 0, 1 を等確率で（N_n = 1 iff U_{n,1} < 1/2）
    st = np.searchsorted(np.cumsum([0.5, 0.5]), np.array([0.0, 0.4999, 0.5, 0.9999]), side="right")
    assert list(st) == [0, 0, 1, 1]
    print("selftest: すべての点検を通った")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()

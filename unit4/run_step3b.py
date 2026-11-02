"""単元四 手順三 (b)：三状態・二合図・三行為、状態は等確率（論文 §4 でシミュレーション扱いの場合）。

規則は手順二と同じ（どの壺も各色 1 個、勝ったら引いた玉の写しを一つずつ足す。状態 j には行為 j が正解）。
各判定時刻で、次の三つ（＋その他）に分ける（ASSUMPTIONS.md の 14'、閾値 U4_CLASS_PROP）：
    (ア) 合図を共有：三つの状態がどれも一つの合図に閾値以上かたより、二つの合図が両方使われ、一方は二つの状態で共有。
         専用の合図を受けた受け手は、その持ち主の状態の行為を閾値以上で選ぶ。
    (イ) 合図のない状態：二つの状態が別々の合図に閾値以上かたより、残りの一つはどちらにも閾値未満。
         受け手は、どちらの合図にも、その持ち主の状態の行為を閾値以上で選ぶ。
    全部同じ合図：三つの状態がどれも同じ合図に閾値以上かたよる。
    その他：どれでもない。
あわせて、勝つ確率 ψ と情報 I（状態 ; 行為）を、壺の中身から計算する。

出力（results/）：step3b_counts.csv、step3b_final.csv、step3b.png
使い方（sync-lab/unit4 で）:  python3 run_step3b.py
"""

import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
from signaling import SignalingUrns  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CLASSES = ["(a) shared signal", "(b) state without signal", "all one signal", "other"]


def channel(m):
    ps = m.sender / m.sender.sum(axis=2, keepdims=True)        # P(合図 | 状態)
    pr = m.receiver / m.receiver.sum(axis=2, keepdims=True)    # P(行為 | 合図)
    return ps, pr


def psi_and_info(m, state_probs):
    ps, pr = channel(m)
    p = np.asarray(state_probs)
    joint = p[None, :, None] * np.einsum("rix,rxj->rij", ps, pr)   # p(状態, 行為)
    psi = np.trace(joint, axis1=1, axis2=2)
    pi_ = joint.sum(axis=2, keepdims=True)
    pj_ = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(joint > 0, joint * np.log2(joint / (pi_ * pj_)), 0.0)
    return psi, term.sum(axis=(1, 2))


def classify(m):
    th = config.U4_CLASS_PROP
    ps, pr = channel(m)
    cls = np.full(m.n_runs, 3)
    for r in range(m.n_runs):
        dom = ps[r].max(axis=1) >= th
        sig = ps[r].argmax(axis=1)
        if dom.all():
            used = set(sig.tolist())
            if len(used) == 1:
                cls[r] = 2
                continue
            owners = [i for i in range(3) if (sig == sig[i]).sum() == 1]
            if len(owners) == 1:
                o = owners[0]
                if pr[r, sig[o], o] >= th:
                    cls[r] = 0
        elif dom.sum() == 2:
            two = np.flatnonzero(dom)
            if sig[two[0]] != sig[two[1]] and all(pr[r, sig[k], k] >= th for k in two):
                cls[r] = 1
    return cls


def main():
    os.makedirs(RESULTS, exist_ok=True)
    probs = [1.0 / config.U4_STEP3B_N_STATES] * config.U4_STEP3B_N_STATES
    rng = np.random.default_rng(config.U4_STEP3B_SEED)
    m = SignalingUrns(config.U4_STEP3B_RUNS, n_states=config.U4_STEP3B_N_STATES,
                      n_signals=config.U4_STEP3B_N_SIGNALS, n_acts=config.U4_STEP3B_N_ACTS, state_probs=probs)
    cps = set(config.U4_STEP3B_CHECKPOINTS)
    counts, t0 = [], time.time()
    for n in range(1, config.U4_STEP3B_CHECKPOINTS[-1] + 1):
        m.step(rng)
        if n in cps:
            cls = classify(m)
            psi, info = psi_and_info(m, probs)
            counts.append([n] + [int((cls == k).sum()) for k in range(len(CLASSES))] +
                          [float(psi.mean()), float(info.mean())])
            print(f"n = {n}: {time.time() - t0:.0f} s", flush=True)

    with open(os.path.join(RESULTS, "step3b_counts.csv"), "w") as fh:
        fh.write("n,shared_signal,state_without_signal,all_one_signal,other,psi_mean,info_mean\n")
        for row in counts:
            fh.write(",".join(str(v) for v in row[:5]) + f",{row[5]:.6f},{row[6]:.6f}\n")
    np.savetxt(os.path.join(RESULTS, "step3b_final.csv"), np.column_stack([np.arange(m.n_runs), cls, psi, info]),
               delimiter=",", comments="", header="run,class_0shared_1without_2allone_3other,psi,info_bits",
               fmt=["%d", "%d", "%.6f", "%.6f"])

    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "0.75"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
    cps_l = config.U4_STEP3B_CHECKPOINTS
    bottom = np.zeros(len(cps_l))
    for k, name in enumerate(CLASSES):
        frac = np.array([r[1 + k] for r in counts]) / config.U4_STEP3B_RUNS
        a1.bar(range(len(cps_l)), frac, bottom=bottom, color=colors[k], edgecolor="white", lw=2, label=name)
        bottom += frac
    a1.set_xticks(range(len(cps_l)))
    a1.set_xticklabels([f"10^{int(np.log10(c))}" for c in cps_l])
    a1.set_ylabel(f"fraction of {config.U4_STEP3B_RUNS} runs")
    a1.legend(frameon=False, fontsize=8)
    a2.hist(psi, bins=np.linspace(0, 1, 51), color="#2a78d6")
    a2.axvline(2 / 3, color="#eb6834", ls="--", lw=1.5, label="2/3")
    a2.set_xlabel("ψ: success probability at the last checkpoint")
    a2.legend(frameon=False, fontsize=8)
    a3.hist(info, bins=np.linspace(0, np.log2(3), 51), color="#1baf7a")
    a3.axvline(1.0, color="#eb6834", ls="--", lw=1.5, label="1 bit (2 signals)")
    a3.axvline(np.log2(3), color="0.4", ls=":", lw=1.5, label="log2 3 (state entropy)")
    a3.set_xlabel("I(state ; act) [bit] at the last checkpoint")
    a3.legend(frameon=False, fontsize=8)
    for ax in (a1, a2, a3):
        ax.grid(color="0.92", lw=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "step3b.png"), dpi=150)

    print("\n内訳（本数）  " + " / ".join(CLASSES) + "   ψ の平均   I の平均")
    for row in counts:
        print(f"  n = {row[0]:>7d}: " + " / ".join(str(v) for v in row[1:5]) + f"   {row[5]:.4f}   {row[6]:.4f}")


if __name__ == "__main__":
    main()

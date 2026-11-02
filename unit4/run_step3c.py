"""単元四 手順三 (c)：三状態・三合図・三行為、状態は等確率（論文 §4 でシミュレーション扱い。論文は「部分的な情報の受け渡し」の均衡が実際にありうるかを未解決問題としている）。

規則は手順二と同じ。各判定時刻で分類する（ASSUMPTIONS.md、閾値 U4_CLASS_PROP）：
    完全な言葉：三つの状態がそれぞれ別の合図に閾値以上かたより、受け手もそれぞれの合図に正しい行為を閾値以上で選ぶ。
    同義語＋共有：二つの状態が一つの合図 x を共有（どちらも x に閾値以上）。残りの状態 k は残り二つの合図 y, z を混ぜて使い
                  （y, z のどちらにも閾値未満、合わせて閾値以上）、受け手は y にも z にも行為 k を閾値以上で選ぶ。
    共有＋使われない合図：二つの状態が x を共有し、残りの状態 k は y に閾値以上（受け手は y に行為 k を閾値以上）。合図 z はほとんど使われない。
    その他：どれでもない。
あわせて ψ と I（状態 ; 行為）を計算する（run_step3b の関数を使う）。

出力（results/）：step3c_counts.csv、step3c_final.csv、step3c.png
使い方（sync-lab/unit4 で）:  python3 run_step3c.py
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
from run_step3b import channel, psi_and_info  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CLASSES = ["perfect language", "synonyms + shared", "shared + unused signal", "other"]


def classify(m):
    th = config.U4_CLASS_PROP
    ps, pr = channel(m)
    cls = np.full(m.n_runs, 3)
    for r in range(m.n_runs):
        S, Rr = ps[r], pr[r]
        sig = S.argmax(axis=1)
        dom = S.max(axis=1) >= th
        if dom.all() and len(set(sig.tolist())) == 3 and all(Rr[sig[i], i] >= th for i in range(3)):
            cls[r] = 0
            continue
        for x in range(3):
            sharers = [i for i in range(3) if S[i, x] >= th]
            if len(sharers) != 2:
                continue
            k = ({0, 1, 2} - set(sharers)).pop()
            y, z = [s for s in range(3) if s != x]
            if S[k, y] + S[k, z] < th:
                continue
            if S[k, y] < th and S[k, z] < th and Rr[y, k] >= th and Rr[z, k] >= th:
                cls[r] = 1
            elif (S[k, y] >= th and Rr[y, k] >= th) or (S[k, z] >= th and Rr[z, k] >= th):
                cls[r] = 2
            break
    return cls


def main():
    os.makedirs(RESULTS, exist_ok=True)
    K = config.U4_STEP3C_N
    probs = [1.0 / K] * K
    rng = np.random.default_rng(config.U4_STEP3C_SEED)
    m = SignalingUrns(config.U4_STEP3C_RUNS, n_states=K, n_signals=K, n_acts=K, state_probs=probs)
    cps = set(config.U4_STEP3C_CHECKPOINTS)
    counts, t0 = [], time.time()
    for n in range(1, config.U4_STEP3C_CHECKPOINTS[-1] + 1):
        m.step(rng)
        if n in cps:
            cls = classify(m)
            psi, info = psi_and_info(m, probs)
            counts.append([n] + [int((cls == k).sum()) for k in range(len(CLASSES))] +
                          [float(psi.mean()), float(info.mean())])
            print(f"n = {n}: {time.time() - t0:.0f} s", flush=True)

    with open(os.path.join(RESULTS, "step3c_counts.csv"), "w") as fh:
        fh.write("n,perfect,synonyms_shared,shared_unused,other,psi_mean,info_mean\n")
        for row in counts:
            fh.write(",".join(str(v) for v in row[:5]) + f",{row[5]:.6f},{row[6]:.6f}\n")
    np.savetxt(os.path.join(RESULTS, "step3c_final.csv"), np.column_stack([np.arange(m.n_runs), cls, psi, info]),
               delimiter=",", comments="", header="run,class_0perfect_1synshared_2sharedunused_3other,psi,info_bits",
               fmt=["%d", "%d", "%.6f", "%.6f"])

    matplotlib.rcParams.update({"axes.edgecolor": "0.55", "axes.labelcolor": "0.1",
                                "xtick.color": "0.35", "ytick.color": "0.35"})
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "0.75"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
    cps_l = config.U4_STEP3C_CHECKPOINTS
    bottom = np.zeros(len(cps_l))
    for k, name in enumerate(CLASSES):
        frac = np.array([r[1 + k] for r in counts]) / config.U4_STEP3C_RUNS
        a1.bar(range(len(cps_l)), frac, bottom=bottom, color=colors[k], edgecolor="white", lw=2, label=name)
        bottom += frac
    a1.set_xticks(range(len(cps_l)))
    a1.set_xticklabels([f"10^{int(np.log10(c))}" for c in cps_l])
    a1.set_ylabel(f"fraction of {config.U4_STEP3C_RUNS} runs")
    a1.legend(frameon=False, fontsize=8)
    a2.hist(psi, bins=np.linspace(0, 1, 51), color="#2a78d6")
    a2.axvline(2 / 3, color="#eb6834", ls="--", lw=1.5, label="2/3")
    a2.set_xlabel("ψ: success probability at the last checkpoint")
    a2.legend(frameon=False, fontsize=8)
    a3.hist(info, bins=np.linspace(0, np.log2(3), 51), color="#1baf7a")
    a3.axvline(np.log2(3), color="0.4", ls=":", lw=1.5, label="log2 3")
    a3.set_xlabel("I(state ; act) [bit] at the last checkpoint")
    a3.legend(frameon=False, fontsize=8)
    for ax in (a1, a2, a3):
        ax.grid(color="0.92", lw=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "step3c.png"), dpi=150)

    print("\n内訳（本数）  " + " / ".join(CLASSES) + "   ψ の平均   I の平均")
    for row in counts:
        print(f"  n = {row[0]:>7d}: " + " / ".join(str(v) for v in row[1:5]) + f"   {row[5]:.4f}   {row[6]:.4f}")


if __name__ == "__main__":
    main()

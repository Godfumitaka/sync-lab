# 論文の規則とコードの対応

論文：Argiento, Pemantle, Skyrms, Volkov (2009) "Learning to signal: Analysis of a micro-level reinforcement model",
Stochastic Processes and their Applications 119(2), 373–390（以下、節と式の番号は論文のもの）

コード：`unit4/signaling.py`（行番号はこの版のもの）、数は `config.py` の `U4_*`

この表は **模型の定義（規則）だけ** を対応させる。論文の主定理（§1.4 Theorem 1.1）と考察（§4）は、手順二・三で予想を伺ったあとに照らし合わせるので、ここには書かない。

## 登場するもの

| 論文 | 論文の場所 | コード |
|---|---|---|
| 三者：Nature、Player 1（Sender、送り手）、Player 2（Receiver、受け手） | §1.2 | Nature は `step` の中の乱数（50 行）。送り手と受け手は `SignalingUrns.sender` と `.receiver` |
| Nature の手：状態 1 または 2 | §1.2 | `state`（0 または 1）。50 行 |
| 送り手の手：合図 A または B。送り手は Nature の手を見る | §1.2 | `signal`（0 = A、1 = B）。52 行で `state` の壺から引く |
| 受け手の手：行為 1 または 2。受け手は合図だけを見て、Nature の手は見ない | §1.2 | `act`（0 = 行為 1、1 = 行為 2）。53 行で `signal` の壺から引く（`state` は使わない） |
| 勝ち：受け手の手が Nature の手と等しい。二人とも勝つか、二人とも負ける | §1.2、§1.4 の win_n | `win = act == state`（54 行） |

## 壺

| 論文 | 論文の場所 | コード |
|---|---|---|
| 送り手は Nature の手ごとに壺を持つ（Urn 1、Urn 2）。玉の色は A と B | §1.3 | `sender[r, i, x]`：走行 r、状態 i の壺の、合図 x の玉の数（41 行） |
| 受け手は合図ごとに壺を持つ（urn A、urn B）。玉の色は 1 と 2 | §1.3 | `receiver[r, x, j]`：走行 r、合図 x の壺の、行為 j の玉の数（42 行） |
| V(n, i, x)：時刻 n の壺 i の色 x の玉の数 | §1.4 | 送り手の V(n, i, x) は `sender[r, i, x]`、受け手の V(n, x, j) は `receiver[r, x, j]` |
| 初期の中身：どの壺にも各色一つずつ。V(1, i, x) = 1（i = 1, 2, x = A, B、および i = A, B, x = 1, 2） | §1.3、§1.4 | `np.full(..., init_balls)`（41–42 行）、`U4_INIT_BALLS = 1.0` |

## 一回のやりとり（§1.4 の帰納的な構成）

| 論文 | 論文の場所 | コード |
|---|---|---|
| 独立な一様乱数 U_{n,1}, U_{n,2}, U_{n,3} | §1.4 | `u1, u2, u3 = rng.random(...)`（49 行） |
| N_n = 1 iff U_{n,1} < 1/2（Nature は毎回独立に、1 と 2 を等確率で選ぶ） | §1.3 末、§1.4 | `np.searchsorted(state_cum, u1, side="right")`（50 行）、`U4_STATE_PROBS = [0.5, 0.5]`。u1 < 0.5 なら 0（状態 1） |
| S_n = A iff U_{n,2} < V(n, N_n, A) / (V(n, N_n, A) + V(n, N_n, B))（玉の割合に比例して引く） | §1.4 | `draw(sender[r, state], u2)`（52 行、`draw` は 25–31 行） |
| R_n = 1 iff U_{n,3} < V(n, S_n, 1) / (V(n, S_n, 1) + V(n, S_n, 2)) | §1.4 | `draw(receiver[r, signal], u3)`（53 行） |
| 負けたら、どの壺の中身も変えない | §1.3 | `win` が偽の走行は 56–57 行で更新されない |
| 勝ったら、二人とも、引いた玉の写しを一つ、引いた壺に足す。V(n+1, i, x) = V(n, i, x) + 1（N_n = i, S_n = x, R_n = i のとき送り手の壺、S_n = i, R_n = x, N_n = x のとき受け手の壺）、それ以外は不変 | §1.3、§1.4 | 56–57 行、`U4_REINFORCE = 1.0` |
| win_n = Σ_{k≤n} δ(N_k, R_k)（n 回目までの勝ちの数） | §1.4 | `self.wins`（43 行、58 行で加算） |

## 論文の観察を、コードの点検に使っているもの

| 論文 | 論文の場所 | コード |
|---|---|---|
| 初期が対称なら、V(n+1,1,A) = V(n,1,A) + 1 ⇔ V(n+1,A,1) = V(n,A,1) + 1 など。つまり V(n, i, x) = V(n, x, i) がいつも成り立つ | §1.4（Theorem 1.1 の後） | `selftest` で、毎ステップ `sender == receiver の転置` を確かめる |
| 二択の引き方「U < 割合 なら A」 | §1.4 | `selftest` で、`draw` が二択のとき論文の式と一致することを確かめる |
| 勝つたびに玉は送り手・受け手それぞれちょうど一つ増える | §1.3 | `selftest` で、玉の総数の増え方と勝ちの数の一致を確かめる |

`python3 signaling.py --selftest` は、上の点検だけを行い、成功率や落ち着き先などの結果の統計は出さない。

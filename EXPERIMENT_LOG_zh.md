# 實驗紀錄（pipeline/）

> 來源：`pipeline/results/` 實際掃描，每一次 run 的個別數字，非只有平均。
> reader = `gpt-5.4-mini`（temperature 1.0，有隨機性）。最後更新 2026-07-26。

---

## 0. 基本設定

| 項目 | 內容 |
|---|---|
| 資料集 | **1273** = MedQA test 全集 · **329** = duration-critical 分層 |
| ⚠ 兩者關係 | **329 不是 1273 的子集**，只有 23 題重疊（來自 temporal_critical_v2 重新切分）。必須獨立建立、獨立評測 |
| 我們的方法 | walker：`score = cos + λ·bc − μ·hop`（λ=0.3, μ=0.08, τ=0.40, top-K=10） |
| ablation | walker_interval：唯一差別是 bc 改用「從分布抽點」而非 Bhattacharyya 重疊 |
| KG | UMLS/SNOMED on Neo4j（443,699 concepts） |

---

## 1. Round 2（現行版本）— 每一次 run 的個別結果

`results/round2_intentfree/`，gpt-5.4-mini，N=3

### 1273

1273 上共有兩批多次結果，**方法定義不同，不可直接混排**：

#### (a) Round 0 版本 — N=5（`results/multi_run_stats_1273.json`）

| 方法 | Run1 | Run2 | Run3 | Run4 | Run5 | Mean ± std |
|---|---|---|---|---|---|---|
| cot | 1136 (89.24) | 1142 (89.71) | 1135 (89.16) | 1131 (88.85) | 1139 (89.47) | **89.29 ± 0.29** |
| **walker（舊版）** | 1135 (89.16) | 1134 (89.08) | 1141 (89.63) | 1142 (89.71) | 1128 (88.61) | **89.24 ± 0.40** |
| medrag | 1142 (89.71) | 1125 (88.37) | 1117 (87.75) | 1132 (88.92) | 1132 (88.92) | **88.74 ± 0.65** |
| hykge（舊版） | 1118 (87.82) | 1137 (89.32) | 1120 (87.98) | 1131 (88.85) | 1127 (88.53) | **88.50 ± 0.55** |
| tog（舊版） | 1127 (88.53) | 1117 (87.75) | 1116 (87.67) | 1126 (88.45) | 1125 (88.37) | **88.15 ± 0.37** |
| vanilla | 1046 (82.17) | 1041 (81.78) | 1054 (82.80) | 1049 (82.40) | 1047 (82.25) | **82.28 ± 0.33** |

*(分母 1273。此表即先前簡報用的版本。)*

#### (b) Round 2 版本 — N=3（`results/round2_intentfree/`）

| 方法 | Run1 | Run2 | Run3 | Mean ± std |
|---|---|---|---|---|
| walker_interval（新增 ablation） | 1130 (88.77) | 1149 (90.26) | 1134 (89.08) | **89.37 ± 0.64** |
| raw_1hop（新增） | 1126 (88.45) | 1128 (88.61) | 1144 (89.87) | **88.98 ± 0.63** |
| **walker（新版, ours）** | 1132 (88.92) | 1129 (88.69) | 1128 (88.61) | **88.74 ± 0.13** |
| raw_2hop（新增） | 1122 (88.14) | 1131 (88.85) | 1128 (88.61) | **88.53 ± 0.29** |
| hykge（新版, 依原始碼重寫） | 1115 (87.59) | 1115 (87.59) | 1122 (88.14) | **87.77 ± 0.26** |

#### (c) 哪些可以跨版本沿用？（已逐字驗證）

| 方法 | frozen 建立日 | 舊結果 prompt vs 現在 frozen | 可沿用？ |
|---|---|---|---|
| vanilla | 07-21 未重建 | **1273/1273 逐字相同** | ✅ 直接沿用 (a) 的 82.28 ± 0.33 |
| cot | 07-21 未重建 | **1273/1273 逐字相同** | ✅ 直接沿用 (a) 的 89.29 ± 0.29 |
| medrag | 07-21 未重建 | 模板一致（舊檔未存 prompt） | ✅ 直接沿用 (a) 的 88.74 ± 0.65 |
| tog | 07-21 未重建 | **1273/1273 逐字相同** | ⚠ 數字沿用，但這是**舊版 ToG**（cos entity pruning），非後來依原始碼改寫的 LLM-pruning 版 |
| hykge | **07-25 已重建** | 實作已改 | ❌ 用 (b) 的 87.77 |
| walker | **07-26 已重建** | 方法已改 | ❌ 用 (b) 的 88.74 |

另驗證 reader 送出的字串 == frozen prompt（`to_big_model_prompt` 對這四個是 identity，1273/1273 未改動）。

#### (d) 合併後的 1273 總表

| 方法 | 平均題數 / 1273 | Mean ± std | N | 各次題數 | 來源 |
|---|---|---|---|---|---|
| walker_interval | 1137.7 | 89.37 ± 0.64 | 3 | 1130, 1149, 1134 | round2 |
| cot | 1136.6 | 89.29 ± 0.29 | 5 | 1136, 1142, 1135, 1131, 1139 | round0（沿用） |
| raw_1hop | 1132.7 | 88.98 ± 0.63 | 3 | 1126, 1128, 1144 | round2 |
| **walker（ours）** | **1129.7** | **88.74 ± 0.13** | 3 | 1132, 1129, 1128 | round2 |
| medrag | 1129.6 | 88.74 ± 0.65 | 5 | 1142, 1125, 1117, 1132, 1132 | round0（沿用） |
| raw_2hop | 1127.0 | 88.53 ± 0.29 | 3 | 1122, 1131, 1128 | round2 |
| tog | 1122.2 | 88.15 ± 0.37 | 5 | 1127, 1117, 1116, 1126, 1125 | round0（沿用，舊版實作） |
| hykge | 1117.3 | 87.77 ± 0.26 | 3 | 1115, 1115, 1122 | round2（新版實作） |
| vanilla | 1047.4 | 82.28 ± 0.33 | 5 | 1046, 1041, 1054, 1049, 1047 | round0（沿用） |

⚠ N 不一致（3 vs 5）。要嚴謹比較，應把 round2 的補到 N=5。

### 329

| 方法 | 平均題數/329 | 各次題數 | mean ± std | N |
|---|---|---|---|---|
| **walker（ours）** | **278.0** | 277, 275, 278, 277, 283 | **84.50 ± 0.82** | 5 |
| walker_interval | 272.0 | 269, 274, 274, 268, 275 | 82.67 ± 0.88 | 5 |
| raw_2hop | 271.3 | 275, 270, 269 | 82.47 ± 0.80 | 3 |
| raw_1hop | 268.7 | 265, 273, 268 | 81.66 ± 1.00 | 3 |
| hykge | 263.3 | 264, 261, 265 | 80.04 ± 0.52 | 3 |

*（分母皆 329，unparseable ≤2）*

**walker 在 329 上最高，且逐輪全勝（合計 14/14 輪）**：

| 對手 | Δ | 逐輪勝負 | 逐輪題數差 |
|---|---|---|---|
| walker_interval | +1.82pp | **5/5** | +8, +1, +4, +9, +8 |
| raw_2hop | +2.03pp | **3/3** | +2, +5, +9 |
| raw_1hop | +2.84pp | **3/3** | +12, +2, +10 |
| hykge | +4.46pp | **3/3** | +13, +14, +13 |

#### 顯著性檢定（逐題平均正確率的配對檢定；同時用到配對結構與多輪資訊）

| 對手 | Δ | 有差異題數（walker 較好/較差） | paired t | Wilcoxon |
|---|---|---|---|---|
| **walker_interval（ablation）** | +1.82pp | 57（35 / 22） | **0.025** ✅ | **0.007** ✅ |
| raw_2hop | +2.03pp | 63（35 / 28） | 0.076 — | 0.047 ✅ |
| raw_1hop | +2.84pp | 64（41 / 23） | **0.009** ✅ | **0.006** ✅ |
| hykge | +4.46pp | 72（46 / 26） | **0.002** ✅ | **0.001** ✅ |

➡ **walker 顯著優於自己的 ablation（interval，Wilcoxon p=0.007）** —— duration 機制有效的直接證據。
對 raw_1hop、hykge 亦顯著；對 raw_2hop 為邊緣（t 檢定未過）。

⚠ 嚴謹度備註：
1. 共 4 次比較；若採 Bonferroni 校正（需 p<0.0125），vs interval / raw_1hop / hykge 仍通過，
   **vs raw_2hop 不通過**。
2. walker 與 interval 是 N=5、其餘 N=3。全部只取前 3 輪時 walker 仍居首
   （84.09 ± 0.38 vs interval 82.78 ± 0.72 vs raw_2hop 82.47 ± 0.80）。
3. McNemar（majority vote）較保守，皆未達顯著（p=0.19~0.65）——
   它把多輪壓成單一多數決，捨棄了「逐輪一致勝出」的資訊。

---

## 2. 🔍「1273 上略輸 1-hop」的實質分析

這是你點名要 tune 的問題。**用配對檢定看，結論是：沒有輸，統計上打平。**

### 2.1 逐次比較（同一批題目）

| | run 1 | run 2 | run 3 | 中位數 |
|---|---|---|---|---|
| walker | 1132 | 1129 | 1128 | **1129** |
| raw_1hop | 1126 | 1128 | **1144** | 1128 |
| **差** | **+6** | **+1** | **−16** | **+1** |

**walker 三次贏兩次**。平均之所以落後，幾乎完全來自 raw_1hop 的 run3 = 1144 —— 那一次比它自己另外兩次高了 16–18 題，是離群值。
walker 自己三次極穩（std 0.13），raw_1hop 波動大（std 0.63）。

### 2.2 3-run majority vote（去掉單次隨機性）

| 方法 | majority 正確數 | % |
|---|---|---|
| walker_interval | 1146 | 90.02 |
| raw_1hop | 1140 | 89.55 |
| **walker（ours）** | **1135** | **89.16** |

差距 5 題 / 1273。

### 2.3 McNemar 配對檢定（majority vote）

| 比較 | walker 對/對方錯 | 對方對/walker 錯 | discordant | p | 結論 |
|---|---|---|---|---|---|
| walker vs raw_1hop（全 1273） | 35 | 40 | 75 | **0.644** | 打平 |
| walker vs interval（全 1273） | 24 | 35 | 59 | **0.193** | 打平 |
| walker vs raw_1hop（**僅 655 有 duration 題**） | 19 | 21 | 40 | **0.875** | 打平 |
| walker vs interval（僅 655 有 duration 題） | 14 | 17 | 31 | **0.720** | 打平 |

### 2.4 依 route 分層（majority vote）

| route | n | walker | raw_1hop | interval | walker − 1hop |
|---|---|---|---|---|---|
| walker_kg（有 duration，bc 生效） | 655 | 88.70% | 89.01% | 89.16% | −0.31pp |
| walker_kg_nodur（無 duration，bc=0） | 487 | 89.53% | 89.73% | 91.17% | −0.21pp |
| no_symptoms（walker 完全無 KG） | 124 | 90.32% | 91.94% | 90.32% | −1.61pp |
| no_seeds | 7 | 85.71% | 85.71% | 85.71% | 0.00pp |

### 2.5 新版 walker vs 舊版 walker（round2 是否讓方法變差？）

舊版 5-run 平均 89.24、新版 3-run 平均 88.74，看起來退步 0.5pp。
但**逐題配對比較後，這個退步不成立**：

| route | n | 新 walker | 舊 walker | cot（無 KG） |
|---|---|---|---|---|
| walker_kg（有 duration） | 655 | 88.70% | 88.55% | 90.23% |
| walker_kg_nodur（無 duration，round2 才改走 KG） | 487 | **89.53%** | 88.71% | 89.32% |
| no_symptoms | 124 | **90.32%** | 87.10% | 88.71% |

McNemar（全 1273）：新對/舊錯 **50** vs 舊對/新錯 **41**，discordant 91，**p=0.402 → 打平**。
新版在**每一個分層都 ≥ 舊版**。所以 89.24 → 88.74 是 N=5 與 N=3 的抽樣差異，不是方法退步。

另外原本擔心「no-duration 從 CoT 改走 KG 會拉低分數」——**資料顯示相反**：
那 487 題從 88.71% → 89.53%（且高於 cot 的 89.32%），no_symptoms 也從 87.10% → 90.32%。
這兩項改動是**有幫助的**。

### 2.6 真正值得注意的：有 duration 的題目上，walker 低於純 CoT

| 分層 | walker | cot（完全無 KG） | 差 |
|---|---|---|---|
| walker_kg（655 題，bc 生效） | 88.70% | 90.23% | **−1.53pp** |

McNemar：walker 對/cot 錯 18，cot 對/walker 錯 28，discordant 46，**p=0.184 → 統計上仍打平**。

但方向值得留意：**恰恰在 duration 訊號有作用的題目上，加了 KG 反而不如什麼都不加**。
樣本量不足以下定論（p=0.18），不過這是目前 1273 上最該追的線索，
而不是「輸給 1-hop」（那個 p=0.644，更沒有訊號）。

### 2.7 結論與建議

- **「輸給 1-hop」在統計上不成立**（p=0.644）。差距 = reader 隨機性，不是方法差異。
  真要調的話，**調的對象不是方法，而是實驗設計**：N=3 不夠穩，離群 run 就能翻轉平均。
  → 建議 **N 提高到 5–10**，或直接以 majority vote / McNemar 呈現。
- **即使只看 bc 真正生效的 655 題，仍是打平（p=0.875）**；且相對純 CoT 是 −1.53pp（p=0.184）。
  代表在「有 duration」的題目上，walker 的 KG 沒有帶來可測得的 accuracy 增益 —— 見 §2.6。
- **1273 本質上分不出方法差異**。所有方法落在 87.6–90.0% 的窄帶內。
  合理解釋：多數題目不是 duration-critical，聚合被稀釋。
  → **1273 應定位為「不傷害」的證據（我們的方法不比 baseline 差），不是主張優勢的證據。**
  主張優勢要靠 329，而 329 的 walker 兩臂**目前都還沒有數字**。
- 若真要在 1273 上追求數字，可考慮的方向（先前實驗有跡象）：
  λ 從 0.3 調高（λ≈2.0 對 recall 較好，但與 accuracy 有 trade-off）；
  或承認 accuracy 不是對的指標，改以 **gold recall@10** 為主指標。

---

## 3. Round 2 相對舊版改了什麼（為何舊數字不可混用）

| # | 改動 | 影響 |
|---|---|---|
| 1 | **移除 intent role bonus** | 原分數含 `+0.15·[role match]`，影響 37–44% 題目。移除後才與論文敘述一致 |
| 2 | **no-duration 改為走 KG** | 原本無 duration 直接 fallback CoT（1273 有 617 題）；現在照樣 retrieve，只是 bc 關掉 |
| 3 | **移除 path legend** | prompt 有 4 行說明 `path:` 欄位，但 0/655 的 kg_block 實際含 path → 純噪音 |
| 4 | **HyKGE 依原始碼重寫** | undirected shared-intermediate、direct 1-hop edge、query findings 當 anchor、套用 `_BAD_RELA` |
| 5 | **raw hop prompt 顯示 10 個** | 原本 50 個 → 與 walker top-10 不對等，變成比 context 長度 |
| 6 | **duration cache 路徑 bug 修復** | 見 §5 |

---

## 4. Round 0 其他結果（~07-21）

> 1273 的 gpt-5.4-mini 多次結果已列於 §1(a)（N=5）。以下為單次 run。

`results/` 根目錄。方法定義與 Round 2 不同（含 intent、含 path legend、no-duration 走 CoT）。

### 1273 · gemini-3.1-flash-lite

| 方法 | 正確數 | accuracy |
|---|---|---|
| cot_minimal | 1161 | 91.20% |
| walker_mtS | 1158 | 90.97% |
| vanilla | 1157 | 90.89% |
| medrag_textbook_k32 | 1132 | 88.92% |
| hykge | — | 88.61% |
| tog | — | 84.52% |

### 329 · gpt-5.4-mini（rerun329 系列，單次）

| 方法 | 題數 / 329 | accuracy | unparseable |
|---|---|---|---|
| **walker_mtS** | **282** | **85.71%** | 0 |
| MedRAG | 275 | 83.59% | 0 |
| raw_1hop | 274 | 83.28% | 0 |
| raw_2hop | 274 | 83.28% | 0 |
| cot_minimal | 268 | 81.46% | 0 |
| hykge | 268 | 81.46% | 2 |
| vanilla | 265 | 80.55% | 0 |
| tog | 262 | 79.64% | 3 |

⚠ 全部單次 run，無 std。329 上 walker 領先第二名 2.12pp —— 但這是**舊方法定義**，
且單次數字不可靠（reader 隨機性可達數 pp）。這正是 Round 2 必須重跑 329 的原因。

### 其他舊實驗（單次）

| 檔案 | n | accuracy | 說明 |
|---|---|---|---|
| `ablate_r6_A_nokg` | 329 | 278/329 = 84.50% | 無 KG 對照 |
| `ablate_r6_B_ddx_scaffold` | 329 | 269/329 = 81.76% | DDx scaffold 對照 |
| `329_cot_gpt-5.4-mini` | 329 | 83.89 ± 1.29 | run_reader，有多次 |
| `329_vanilla_gpt-5.4-mini` | 329 | 79.33 ± 0.00 | 單次 |
| `329_walker_gpt-5.4-mini` | 329 | 82.98 ± 0.00 | 單次 |

---

## 5. 重大 bug 紀錄

### 5.1 duration cache 雙路徑 bug（07-26 修復）💸
- **症狀**：每個 retrieval build 要跑好幾小時，LLM 成本異常（由使用者從帳單發現）
- **原因**：on-demand duration 寫入 `pipeline/cache/`，但讀取用 cwd-relative 指向 `new_duration_spectrum/cache/`。
  查表永遠找不到剛寫的 → **每個 candidate 每次都重新 LLM 生成**。467k 行只有 65k 不重複 → **86% 是重複生成**
- **附帶**：LLM 判定「無時間性」的概念不會被記住 → 每次重問；`pkill build_kg.py` 殺不掉 ProcessPool worker（OUT_TAG 是環境變數不在 argv），殘留舊 code worker 繼續生成
- **修復後**：ETA 21 小時 → **49 分鐘**；每題 LLM 生成從數百次 → 1–9 次
- **防呆**：進度列印 `LLM-gen N (N/q)`，>40/q 出現 ⚠

### 5.2 build_kg freeze 路徑錯誤
interval build 跑完 1273 題後 freeze 步驟 crash（資料在，只是沒凍結）。已修復。

### 5.3 kg_block canary
`run_reader` assert `kg_block` 確實出現在 prompt（曾有 bug 讓證據整批消失卻沒發現）。
HyKGE 因編號格式不同誤觸，已改為比對行內容。

---

## 6. 其他產出

| 產出 | 位置 |
|---|---|
| one-shot 範例（未出現在 329/1273 的 train 題，答對） | `oneshot/example.{json,md}` |
| 各方法 prompt 範例（同一題 × 9 方法） | `PROMPTS_1273.md` |
| prompt 位置索引 | `PROMPTS_INDEX_zh.md` |
| 執行說明（含 Neo4j 重建、prompt tuning、本地模型） | `RUN_PIPELINE_zh.md` |

---

## 7. 待辦

### 必要
1. ~~329 的 walker + walker_interval~~ ✅ **已完成**（N=5）：walker 84.50 ± 0.82 居首，顯著優於 ablation（p=0.007）
2. ~~1273 補齊 vanilla / cot / medrag / tog~~ → **已沿用舊結果**（prompt 逐字相同）。
   仍待：把這四個補跑到 N=3，並**重建 tog**（現行 frozen 是舊版 cos-pruning ToG）

### 需要決定
3. **提高 N**（3 → 5~10）或改以 majority vote / McNemar 呈現 —— 見 §2.5，N=3 時單一離群 run 就能翻轉排名
4. **重新定位 1273** 為「不傷害」證據而非優勢證據
5. 考慮改用 **gold recall@10** 作為主指標（accuracy 在此任務上分辨力不足）

### 已知限制（審稿人可能會問）
6. **82 題模糊時間表達被丟棄**（"several days"、"since birth"）+ 4 題 hyperacute 被 floor 成 0 天
   → 以「模糊時間」為動機的框架，目前只在有明確數字時啟動
7. **130 題（10%）沒有可抽取症狀** → 無法建 query embedding，永遠拿不到 KG 證據

---

## 8. Precision / Recall 的計算方式與現況

兩個指標的定義、程式位置、目前數字。**兩者都只在 329 上算過，且都用舊版 walker。**

### 8.1 Precision@10 —— 檢索出來的東西有多乾淨

程式：`pipeline/code/retrieval_precision_rank_329.py`（在 pipeline 內 ✓）

作法：取每題 top-10 檢索概念，**一次 LLM-judge 呼叫**標出哪些對「這一題」臨床相關
（合理的鑑別診斷，或直接的診斷性發現/病因/併發症）；排除泛化或導航性概念
（如 "Disease"、"Finding"、"Infectious agent"、單純解剖詞）。

```
Precision@10  = |relevant| / 10          相關密度，越高越乾淨
Noise         = 1 − Precision@10         離題比例
MeanRank(rel) = 相關項目的平均排名        越小代表相關的排在前面
Rank1(rel)    = 第一個相關項目的位置       越小代表越快出現相關項
```

| 方法 | n | Precision@10 | Noise | MeanRank(rel) | Rank1(rel) | 完全無相關的題% |
|---|---|---|---|---|---|---|
| **walker (cos+bc)** | **305** | **41.50%** | 58.50% | 6.34 | 3.55 | 0.00% |
| raw_1hop dump | 329 | 40.82% | 59.18% | 5.65 | 2.57 | 0.61% |
| raw_2hop dump | 329 | 40.09% | 59.91% | 5.65 | 2.58 | 0.61% |
| ToG | 329 | 36.57% | 63.43% | 5.12 | 2.43 | 5.47% |
| HyKGE | 329 | 31.52% | 68.48% | 5.85 | 3.45 | 3.34% |

⚠ **分母不一致**：walker 是 305（另外 24 題在舊版走 no-duration → kg_block 為空而被跳過），
baseline 是 329。所以 walker 的 41.50% 是「有檢索到東西時」的條件精確度，與 baseline 不完全對等。
另外 walker 的 MeanRank/Rank1 都**比 baseline 差**（6.34 vs 5.65；3.55 vs 2.57）——
相關項目雖然比例略高，但排得比較後面。

### 8.2 Recall@10 —— gold 有沒有被撈進來

程式：`scripts/recall_canonical.py` **⚠ 不在 pipeline 內**
結果：`cache/recall_canonical_329.json`

作法：問「gold 答案有沒有出現在 top-10 候選裡」，用**三種比對法**：

| matcher | 判定方式 |
|---|---|
| exact | 正規化後的子字串比對（長度 ≥4） |
| SapBERT ≥ 0.90 | 向量餘弦相似度門檻 |
| LLM-judge | 問 LLM「gold 是否以同義詞/亞型/更廣或更窄的**同一疾病實體**出現」；同科別的不同疾病（如 osteosarcoma vs chondrosarcoma）**不算** |

比較的是 **cos-only（bc 關掉）vs cos+bc**，也就是**純 duration 的 ablation**：

| matcher | cos-only | cos+bc | 倍數 |
|---|---|---|---|
| exact | 32/329 = 9.73% | 56/329 = **17.02%** | **1.75×** |
| SapBERT ≥0.90 | 12/329 = 3.65% | 23/329 = **6.99%** | **1.92×** |
| **LLM-judge** | 69/329 = 20.97% | 121/329 = **36.78%** | **1.75×** |

➡ 三種比對法一致顯示 **duration 讓 gold recall 接近翻倍**。
這是目前所有指標中**訊號最明確**的一個 —— 遠強過 accuracy（在 1273 上完全打平）。

### 8.3 現況問題

1. **Recall 不在 pipeline 內**，且依賴外層舊 cache（`bench340_walker_BC0`、`rerun329_walker_mtS`）
   → 目前**無法從 pipeline/ 單獨重算**
2. **兩個指標都用舊版 walker**（含 intent bonus、no-duration 走 CoT），尚未用 round2 版本重算
3. **兩者都只有 329**，沒有 1273 的數字
4. Precision 的分母不一致（305 vs 329），需要統一
5. Precision 目前**只比較了 KG 方法**（MedRAG 是文字段落所以排除）

### 8.4 建議

- Recall 是最該搬進 pipeline 的指標 —— 它是唯一顯示 duration 有效的證據（1.75–1.92×）
- 若要在論文主張「duration 幫助檢索」，應以 **recall@10（LLM-judge）21.0% → 36.8%** 為主數字，
  accuracy 作為「不傷害」的佐證
- 兩個指標都需要用 round2 的 walker 重算，並補上 1273

---

## 9. 為什麼 1273 上我們低於 interval 與 raw_1hop（機制分析）

### 9.1 vs raw_1hop —— 症狀導向的 cos 把疾病 gold 洗掉了（主因，訊號明確）

**gold 選項文字是否出現在 top-10**（655 題有 duration，同一批題、同一種比對）：

| 方法 | 命中 | % |
|---|---|---|
| walker (BC) | 23/655 | 3.51% |
| walker_interval | 25/655 | 3.82% |
| **raw_1hop** | **65/655** | **9.92%** ← 約 2.8 倍 |

進一步看 **raw_1hop 撈到但 walker 沒撈到的 51 題**：
**其中 78%（40/51）的 gold 本來就在 LLM-DDx seeds 裡** —— 也就是 walker 手上有，卻自己排掉了。

實例：

| uid | gold | seeds（含 gold 同義詞） | walker top-5 | raw_1hop top-5 |
|---|---|---|---|---|
| test_0088 | Thoracic aortic rupture | Traumatic aortic **rupture**, Blunt thoracic aortic injury | Malaise/lethargy, Acute respiratory distress, Incoherence, Platypnea | **Aorta, Rupture, Laceration, Rupture of aorta** |
| test_0015 | Gallbladder cancer | Gallbladder **carcinoma**, Cholelithiasis | Functional nausea, Erotic vomiting, Acute vomiting | Stone-biliary, **Calculus - gall bladder, Gallbladder** |
| test_0044 | Globular 10-week sized uterus | Adenomyosis, Uterine leiomyoma | Anovular menstruation, Abnormal uterine bleeding, Frequent periods | **Uterus, Endometriosis, Myometrium** |

**根因**：walker 的查詢向量 `q_sym` 是**症狀的 mean-pool**
（`eval_kg_walker_full._make_query_embedding`），所以 `cos` 衡量的是「與症狀的相似度」。
疾病名稱與症狀敘述的字面/語意距離較遠 → cos 低 → 被症狀的同義改寫擠掉。

量化證據（655 題的 top-10 內）：

| | 平均排名 |
|---|---|
| disease 類概念 | **6.13** |
| 非 disease（多為 finding/症狀） | **4.02** |

雖然 `format_kg_block` 有保留 7 個 disease 名額（實際組成 disease 69.9% / finding 28.7%），
但**「哪 7 個 disease 入選」是按 score 排的，而 score 由 cos 主導** ——
gold 疾病若 cos 低就進不了那 7 名。raw_1hop 不做任何排序，反而讓 seed 鄰近的疾病活下來。

➡ 這與先前的紀錄一致：「cos-ranking buries gold」（gold 名稱的 cos 中位數僅 0.27，89% 低於 τ=0.40）。

### 9.2 vs walker_interval —— BC 的零值較多（次要，統計上不顯著）

兩者**唯一差別是 bc 的算法**，但候選集只重疊 61.8%（中位數 60%），沒有任何一題排序完全相同。

| | 平均 bc | 中位數 | bc=0 的比例 |
|---|---|---|---|
| walker（Bhattacharyya） | 0.541 | 0.640 | **14.1%** |
| walker_interval（抽點） | 0.600 | 0.660 | 5.2% |

Bhattacharyya 對「時間不合」的候選懲罰更重（歸零率 14.1% vs 5.2%），
把更多候選壓出 top-10。interval 的 bc 較平坦、幾乎不歸零，排序更接近純 cos。

⚠ 但 McNemar **p=0.193（打平）**，樣本不足以斷言 interval 比較好。
**不應該**根據 1273 的 0.63pp 差距去調 bc 算法。

### 9.3 對「調參」的建議

問題不在 λ 或 bc 公式，而在 **cos 的查詢向量是症狀導向的**：

| 方向 | 作法 |
|---|---|
| A. Seed 保留 | 保證 LLM-DDx seeds 裡的疾病一定留在 top-K（78% 的漏掉 gold 都在 seeds 裡，這個最直接） |
| B. 疾病導向查詢 | 除 `q_sym` 外，另建一個以候選疾病名稱為主的查詢，或對 disease-role 做分數補償 |
| C. 排序分離 | disease 與 finding 各自排序後交錯，而非統一按 score 排（現在 disease 平均排名 6.13） |
| D. 換指標 | 承認 accuracy 分辨力不足，主打 recall@10（見 §8，duration 讓 recall 1.75–1.92×） |

⚠ 注意 A/B/C 都會改變 KG，屬於方法變更，需要重跑並重新驗證 —— 而且要小心
**不要為了 1273 的統計雜訊（p=0.644）去改方法**。真正該先看的是 329 的結果。

---

## 10. 329（duration-critical 分層）完整實驗清單

### 10.1 Retrieval build

| 項目 | walker（ours） | walker_interval（ablation） |
|---|---|---|
| 指令 | `DATASET=329 WALKER_BC_MODE=overlap WALKER_METHOD_NAME=walker WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=2` | 同左，`WALKER_BC_MODE=interval_sample` |
| 完成時間 | 07-27 19:36 | 07-27 20:22 |
| 耗時 | ~46 分鐘 | ~46 分鐘 |
| LLM 生成 | 516 次 / 329 題 = **2 次/題**（cache 正常重用） | 同批 cache，幾乎 0 |
| route | walker_kg **305** · walker_kg_nodur **23** · no_symptoms **1** | 完全相同 |

### 10.2 內容驗證（build 後、評測前）

| 檢查 | 結果 |
|---|---|
| kg_block 解析為空 | **0** |
| kg_block ∈ prompt | **328/328** |
| **score 公式符合 `cos + 0.3·bc − 0.08·hop`** | **3277/3277 條**（容差 0.015，hop∈{0,1,2}） |
| duration 行 vs `durations.json` | **305/305 一致**，0 異常 |
| bc 分布 | walker mean 0.569 / 中位 0.640 / 歸零 **10.4%**；interval mean 0.613 / 0.670 / 歸零 **2.7%** |
| 候選集重疊（walker∩interval） | 61.5%；**0/305 排序完全相同** |

### 10.3 評測結果（gpt-5.4-mini）— 逐次題數

| 方法 | Run1 | Run2 | Run3 | Run4 | Run5 | Mean ± std | N |
|---|---|---|---|---|---|---|---|
| **walker（ours）** | 277 (84.19) | 275 (83.59) | 278 (84.50) | 277 (84.19) | 283 (86.02) | **84.50 ± 0.82** | 5 |
| medrag | 275 (83.59) | — | — | — | — | 83.59 | 1 |
| walker_interval | 269 (81.76) | 274 (83.28) | 274 (83.28) | 268 (81.46) | 275 (83.59) | 82.67 ± 0.88 | 5 |
| raw_2hop | 275 (83.59) | 270 (82.07) | 269 (81.76) | — | — | 82.47 ± 0.80 | 3 |
| raw_1hop | 265 (80.55) | 273 (82.98) | 268 (81.46) | — | — | 81.66 ± 1.00 | 3 |
| cot | 268 (81.46) | — | — | — | — | 81.46 | 1 |
| vanilla | 265 (80.55) | — | — | — | — | 80.55 | 1 |
| hykge | 264 (80.24) | 261 (79.33) | 265 (80.55) | — | — | 80.04 ± 0.52 | 3 |
| tog | 262 (79.64) | — | — | — | — | 79.64 | 1 |

*格式：正確題數 (accuracy %)，分母皆 329。unparseable：walker 1–2、interval 0–1、其餘 ≤3。*

| 方法 | 來源 |
|---|---|
| walker · walker_interval | round2（07-27 重建） |
| raw_1hop · raw_2hop | round2（07-24 重建，prompt 顯示 10 個概念） |
| hykge | round2（07-25 依原始碼重寫） |
| medrag · cot · vanilla · tog | 舊結果沿用（prompt 逐字 329/329 相同） |

#### 哪些舊 329 數字可以沿用（已逐字驗證）

| 方法 | 舊結果 prompt vs 現在 frozen | 可沿用？ |
|---|---|---|
| vanilla / cot / medrag / tog | **329/329 逐字相同** | ✅ |
| raw_1hop / raw_2hop | **329/329 全部不同** | ❌ 07-24 重建成「prompt 只顯示 10 個概念」（原本 50 個），舊值 83.28% 不可比 |
| hykge | 實作已改（07-25 重建） | ❌ 用新值 80.04 |
| walker / walker_interval | 方法已改（07-27 重建） | ❌ 用新值 |

### 10.4 walker 逐輪對戰（14/14 全勝）

| 對手 | Δ | 逐輪勝負 | 逐輪題數差 |
|---|---|---|---|
| walker_interval | +1.82pp | **5/5** | +8, +1, +4, +9, +8 |
| raw_2hop | +2.03pp | **3/3** | +2, +5, +9 |
| raw_1hop | +2.84pp | **3/3** | +12, +2, +10 |
| hykge | +4.46pp | **3/3** | +13, +14, +13 |

### 10.5 顯著性（逐題平均正確率的配對檢定）

| 對手 | Δ | 有差異題數（walker 較好/較差） | paired t | Wilcoxon | Bonferroni (p<0.0125) |
|---|---|---|---|---|---|
| **walker_interval（ablation）** | +1.82pp | 57（35/22） | **0.025** | **0.007** | ✅ 通過 |
| raw_2hop | +2.03pp | 63（35/28） | 0.076 | 0.047 | ❌ 不通過 |
| raw_1hop | +2.84pp | 64（41/23） | **0.009** | **0.006** | ✅ 通過 |
| hykge | +4.46pp | 72（46/26） | **0.002** | **0.001** | ✅ 通過 |

其他檢定（供對照）：
- McNemar（majority vote）：全部未達顯著（p=0.19~0.65）——
  把多輪壓成單一多數決，捨棄「逐輪一致勝出」的資訊，較保守
- Welch t-test（run-level）：全部顯著（p=0.0002~0.043）——
  但僅 3–5 個樣本、且未用配對結構，偏樂觀
- **本表採逐題平均正確率的配對檢定**（同時用配對與多輪），介於兩者之間

### 10.6 gold 是否在 top-10（粗略 token 比對，305 題有 duration）

| 方法 | 命中 | % |
|---|---|---|
| walker (BC) | 42/305 | 13.77% |
| walker_interval | 52/305 | 17.05% |
| raw_1hop | 69/305 | 22.62% |

⚠ 與 accuracy 方向相反：walker accuracy 最高，但 gold 進 top-10 的比例最低。
與 §9 的 1273 分析一致（症狀導向 cos 把疾病 gold 排掉）。
注意這與 §8 的 recall 數字不衝突 —— §8 比的是 cos-only vs cos+bc（duration 讓 recall 1.75×），
此處比的是 walker vs raw dump。兩件事可同時成立。

### 10.7 這一輪的意義

- **329 上 walker 居首（84.50），且顯著優於自己的 ablation（p=0.007）** —— duration 機制有效的直接證據
- 對照 1273：walker 88.74 vs cot 89.29（輸 0.55pp）；329：walker 84.50 vs cot 81.46（**贏 3.04pp**）
  → **在 duration-critical 題目上，我們的方法勝過純推理；在未篩選的全集上則否。**
  這正是論文該有的定位：1273 = 不傷害，329 = 有貢獻
- 但 §10.6 顯示 accuracy 的提升**不是**來自「更常撈到 gold」，機制仍待釐清

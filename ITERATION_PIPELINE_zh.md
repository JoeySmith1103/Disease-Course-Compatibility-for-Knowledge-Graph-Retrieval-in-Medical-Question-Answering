# 疊代 utility：從候選池到結果

要改 walker 的排序方式並量到效果，不需要碰 Neo4j。檢索只做一次，之後所有排序實驗都是對存好的
候選池重排。

```
一次性（貴，需要 Neo4j）
  benchmark + seeds/symptoms/durations ──► walker 走訪 ──► pool/<ds>/<method>.jsonl
                                              每個候選帶 cos / bc / hop / role / path / origin_seed
                                              每題 87–156 個候選，其中只有 10 個會進 prompt

可重複（幾秒，不碰 Neo4j、不呼叫 LLM）
  pool ──► render_from_pool.py ──► frozen/<ds>/<variant>.json

作答（花錢）
  frozen ──► run_reader.py（連續 N 輪） 或 append_run.py（單輪，可累積）
              ──► results/<batch>/<ds>_<variant>_<model>.json
```

本輪測了 40 餘個變體。若每次都重跑檢索需要約 10 小時的走訪；實際檢索只跑一次，其餘都是幾秒的
重排。

---

## 候選池

`pool/<ds>/<method>.jsonl`，一行一題：

```json
{"uid": "...", "gold": "C", "route": "walker_kg", "patient_days": 14,
 "n_candidates": 156,
 "candidates": [{"rank": 1, "cui": "C0458102", "name": "Central crushing chest pain",
                 "role": "finding", "hop": 1, "cos": 0.714, "bc": 0.689,
                 "score": 0.841, "origin_seed": "Crushing chest pain",
                 "bc_onesided": 0.882,
                 "path": [["inverse_isa", "Central crushing chest pain", 0.841]]}]}
```

| dataset | 題數 | 每題候選中位數 | 最多 |
|---|---|---|---|
| 329 | 329 | 156 | 866 |
| medbullets | 308 | 87 | 722 |
| mmlu | 272 | 99 | 837 |

候選同時帶兩種時間相容性：`bc` 是走訪當下算的雙邊 Bhattacharyya 重疊，`bc_onesided` 是預先算好
的單邊 `P(病程 ≥ 已過時間)`。兩者都存在池裡，所以切換 `BC_SRC` 不需要疾病時長快取（那份 48 MB
的資料不在 repo 內）。

`walker` / `walker_interval` 有完整分項；`raw_1hop` / `raw_2hop` 只有概念名（走訪順序，無分數），
`tog` / `hykge` 是知識鏈字串。後四者只能調 top-K。

### 哪些參數能重排、哪些必須重走訪

記錄在每筆記錄的 `params` 欄位，因為這個區分決定了成本：

| 免費（重排池）| 必須重走訪 |
|---|---|
| top_k、λ、μ、utility 形式、role 配額、多樣性、novelty、bc 來源 | **τ (min_score)**、max_hops、neighbor_limit、seeds、query embedding |

τ 是走訪時的**擴展閘門**——分數不過 τ 的節點不會被展開，也不會被記錄。所以調降 τ 會走到從未
被評分的節點，池裡不可能有。`render_from_pool.py` 對此直接拒絕，不會回傳看似合理的近似。

重建池（需 Neo4j）：

```bash
python3 build_pools.py --walk 329          # 重走訪並存池
python3 build_pools.py                     # 僅從既有檔案收集
```

`build_pools.py` 會自動檢查池重排後的 top-10 能否還原 frozen，結果寫進 `frozen_check`。

---

## 逐題可用的額外訊號

候選池裡的 `cos` / `bc` / `hop` 都是**逐候選**的量。若要讓權重隨題目變動——例如某些題目的診斷
根本不依賴病程時間，那些題目的 `bc` 就不該被加權——需要一個**逐題**的訊號。以下這幾份都在
repo 內，可直接以 uid 對上 `benchmark.json` 與候選池：

| 檔案 | 內容 | 粒度 |
|---|---|---|
| `datasets/<ds>/criticality.json` | LLM 判定「這題的診斷有多依賴病程時間」 | 逐題 |
| `datasets/<ds>/durations.json` | 病人已過時間（天），`bc` 的輸入 | 逐題 |
| `datasets/<ds>/seeds.json` / `symptoms.json` / `query_entities.json` | 走訪起點與查詢實體 | 逐題 |
| `datasets/medbullets/temporal_critical.json`、`datasets/mmlu/temporal_critical.json` | 人工判讀的 temporal-critical 題號 | 逐題（子集）|
| `datasets/329/benchmark.json` 的 `verdict` / `audit_source` | 329 自身的分層（`truly_tc` 163、`human_cf` 46）| 逐題 |

`criticality.json` 三個資料集都備妥（329／308／272 題），每題一次 LLM 呼叫，**只讀病歷與題幹、
不讀選項**（讀選項等於讓答案集決定哪些候選被拉高，是間接洩漏）：

```json
"aud_001": {"score": 0.4,
            "decisive_axis": "progression",
            "rationale": "The 72-hour acute progression supports severe infectious sepsis, but
                          diagnosis is driven more by shock, pneumonia signs, and anuric organ
                          failure than duration."}
```

- `score ∈ [0,1]` — 時間對這題的決定性。中位數 329 = 0.50、MedBullets = 0.20、MMLU = 0.10，
  正好對應三者可從時間得到的空間大小。
- `decisive_axis` — 它認為決定性的時間軸。實測分布（標籤由 LLM 自由生成，`duration` 與
  `symptom_duration`、`time_course` 實為同義，使用前要先正規化）：

  | | progression | latency | onset_speed | symptom_duration* | none |
  |---|---|---|---|---|---|
  | 329 | 35% | 22% | 19% | 8% | 15% |
  | MedBullets | 29% | 9% | 15% | 5% | 42% |
  | MMLU | 19% | 11% | 9% | 6% | 54% |

  \* 合併 `symptom_duration` + `duration` + `time_course`。

  這張表就是 `bc` 的已知限制：`bc` 拿「已過時間」比對「疾病總病程」，只模擬 symptom_duration
  這一軸，而該軸只在 5–8% 的題目上是決定性的。真正常見的是 progression（症狀怎麼變化）與
  latency（暴露到發病的間隔），兩者都不是「病程多長」，`bc` 結構上量不到。想改善時間訊號的
  人，這裡是比調 λ 更有空間的方向。
- 典型用法是把 `score` 當成 `a_bc` 的**先驗乘數**而非直接的權重：
  `a_bc ∝ s(q)·base_bc`、`a_cos ∝ (1−s(q))·base_cos`。

分數是 LLM 產生的判斷，不是人工標記；`temporal_critical.json` 與 329 的 `audit_source=human_cf`
（46 題）才是人工判讀的部分。要拿來當評估依據時請用後者，`criticality.json` 適合當方法的輸入。

---

## 改 utility

`render_from_pool.py` 的旋鈕：

| 環境變數 | 預設 | 作用 |
|---|---|---|
| `TOP_K` | 10 | 進 prompt 的候選數 |
| `LAMBDA` | 0.3 | bc 的係數 |
| `MU` | 0.08 | hop 懲罰 |
| `UTILITY` | — | 直接寫式子，如 `UTILITY='cos*bc'`；可用 `cos` `bc` `hop` `score` |
| `BC_SRC` | overlap | `onesided` 改讀候選的 `bc_onesided` 欄位（`P(病程 ≥ 已過時間)`）|
| `DELTA` | 0 | 多樣性懲罰：選第 k 個時扣掉與已選項的名稱重疊 |
| `NOVELTY` | 0 | 題幹重疊懲罰：候選越像題目重述，扣越多 |
| `MAX_HOP` | — | 硬性 hop 上限 |
| `QUOTA_DISEASE` / `QUOTA_OTHER` | 7 / 3 | 角色配額；兩者設 0 則純照分數 |
| `SAMPLE` | — | `random` 從池中均勻抽樣，作為對照 |
| `TEMPLATE_FILE` | — | 換 prompt 模板 |
| `DRY` | — | 只印不寫檔 |

變體名稱自動編碼所有設定，不同設定不會互相覆蓋：

```bash
DATASET=329 METHOD=walker TOP_K=20 LAMBDA=0.6 python3 render_from_pool.py
#   -> frozen/329/walker__k20_l0.6_m0.08.json

DATASET=329 METHOD=walker BC_SRC=onesided DELTA=0.3 python3 render_from_pool.py
#   -> frozen/329/walker__k10_l0.3_m0.08_d0.3_os.json
```

要加新的訊號項，改 `_utility_raw()` 與 `rank()` 即可；記得在變體名稱裡加上對應標籤，否則不同
設定會寫到同一個檔案。

---

## 量測

```bash
# 先跑一次篩選
DATASET=medbullets METHOD=walker__k20_l0.6_m0.08 N_RUNS=1 \
  RESULTS_DIR=results/screen python3 run_reader.py

# 有希望的再補兩次獨立呼叫（不重跑已付費的那次）
DATASET=medbullets METHOD=walker__k20_l0.6_m0.08 \
  RESULTS_DIR=results/screen python3 append_run.py
```

### 兩件會影響結論的量測事實

**一、`run_reader.py` 的 N 輪是連續跑的，std 會嚴重低估。** 同一份 frozen prompt（逐字元驗證
308/308 相同）：

| 量測方式 | 每輪正確數 | std |
|---|---|---|
| 同一次呼叫內連續三輪 | 244 / 243 / 243 | **0.15** |
| 三次獨立呼叫 | 255 / 248 / 239 | **2.13** |

相差 14 倍，而且兩種方式對「該變體是否勝過基準」給出相反答案。`append_run.py` 累積的是獨立
呼叫，其 spread 才是真正限制結論的變異。

**二、13.6% 的題目本身就不穩定。** 三次獨立呼叫下，308 題有 42 題的預測字母會改變（226 題三次
都對、40 題三次都錯）。任何方法能影響的題數都遠小於這個波動範圍。

實務上：

- 要比較的數字必須在**同一批次**內產生，跨批次的列不可相減（同一份 prompt 跨日最多差 2.81pp）
- 真實誤差棒約 ±2pp；小於這個的差距不要當成結論
- 用 `append_run.py` 累積至少 3 次獨立呼叫再判定

---

## 報表

```bash
python3 metrics_report.py 329 medbullets      # accuracy / macro P·R·F1 / parseable / unparseable
python3 critical_subset_report.py medbullets  # temporal-critical 子集，以 vanilla 自身 lift 為基準
python3 old_prompt_report.py                  # 各方法對照，標注來源批次與混批警告
```

329 有 10 個選項字母，其中 J、K 各只有 1 題，macro_* 會被單題主導——那裡請引用 accuracy 或
`weighted_*`，不要引用 macro。

---

## 已知的訊號性質

改 utility 之前值得知道的實測結果（gold 相關 vs 干擾候選的中位數分離度）：

| 訊號 | 329 | MedBullets | 方向 |
|---|---|---|---|
| `cos` | −0.022 | −0.087 | **反向** |
| `bc`（雙邊重疊）| −0.169 | −0.415 | **反向** |
| `bc`（單邊，`BC_SRC=onesided`）| **+0.132** | **+0.169** | 正向 |
| `hop` | hop-0 命中 5.8% vs hop-2 0.22% | — | **正向，26 倍** |

- **cos 反向**是因為它獎勵「像題幹」，而最像題幹的是症狀的同義改寫。實測 top-10 每題有 4.6
  （329）／5.3（MedBullets）個格子與另一格重疊 ≥0.5。`DELTA` 與 `NOVELTY` 兩個旋鈕就是針對
  這件事。
- **雙邊 bc 反向**是因為它拿「已過時間」比對「總病程」。病人在 10 年病程的第 60 天就診完全
  正常，但兩個分布幾乎不重疊，慢性病 gold 因此被壓低（慢性 gold 的 bc 中位 0.477，干擾 0.67）。
  單邊版改問「病程是否可能 ≥ 已過時間」，分離度轉正。
- **角色配額有用**：拿掉後 finding 從 29% 膨脹到 45%，gold@10 反而下降 1.5–2.3pp。它在補償
  cos 對症狀同義字的偏好。

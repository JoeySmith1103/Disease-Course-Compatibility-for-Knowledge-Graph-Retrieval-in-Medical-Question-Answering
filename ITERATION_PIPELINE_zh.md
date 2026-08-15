# 疊代流程：從池到結論

本文件記錄 utility 疊代的完整流程與涉及的檔案。設計重點是**把檢索與作答徹底分離**，使一次
檢索可以支撐任意多次的排序實驗。

---

## 全貌

```
一次性（貴，需要 Neo4j + LLM）
  benchmark.json ──► seeds / symptoms / durations / query_entities
                        │
                        ▼
                    walker 走訪 ──► pool/<ds>/<method>.jsonl
                                     每個候選帶 cos / bc / hop / role / path / origin_seed
                                     每題 87–156 個候選（只有 10 個會進 prompt）

可重複（幾秒，不碰 Neo4j、不呼叫 LLM）
  pool ──► render_from_pool.py ──► frozen/<ds>/<variant>.json
             重排 + 選取 + 渲染        {uid, gold, route, kg_block, prompt}

驗證（不花錢）
  verify_judge.py ──► 六項檢查，未通過不得送 inference

作答（花錢）
  frozen ──► run_reader.py（N 輪連續）  或  append_run.py（單輪，可累積）
              ──► results/<batch>/<ds>_<variant>_<model>.json
```

**分離的價值**：本輪測了 40 餘個變體，若每次都重跑檢索需要約 10 小時的 Neo4j 走訪；實際上檢索
只跑了一次（約 3 小時），其餘全部是幾秒的重排。

---

## 各階段的檔案

### 建池（一次性）

| 檔案 | 作用 |
|---|---|
| `build_pools.py` | 把五種方法的完整候選池收成 `pool/<ds>/<method>.jsonl` |
| `code/eval_kg_walker_full.py` | walker 走訪；`WALKER_POOL_DUMP=1` 才會存完整池 |
| `code/dkr_policy/kg_walker.py` | `walk()`，score = cos + λ·bc − μ·hop，τ 閘門 |
| `code/dkr_policy/bc_llm_direct.py` | bc 計算，`WALKER_BC_MODE=overlap\|interval_sample` |

```bash
python3 pipeline/build_pools.py --walk 329        # 重走訪並存池（需 Neo4j）
python3 pipeline/build_pools.py                   # 僅從既有檔案收集
```

`build_pools.py` 會自動比對池重排後的 top-10 是否還原 frozen，結果寫進每筆記錄的
`frozen_check`。329 與 mmlu 為 100% 還原，medbullets 有 3 題因 duration 快取後續補齊而不同。

**哪些參數可以事後重排、哪些必須重走訪**，記錄在每筆記錄的 `params` 欄位：

| 免費（重排池）| 必須重走訪 |
|---|---|
| top_k、λ、μ、utility 形式、role 配額、多樣性、novelty | τ (min_score)、max_hops、neighbor_limit、seeds、query embedding |

### 判斷層與訊號

| 檔案 | 作用 |
|---|---|
| `judged_utility.py` | 判斷層：dispersion / magnitude / llm / atype，可用 `+` 組合；多樣性選取；novelty |
| `bc_onesided.py` | 單邊時間相容性 `P(病程 ≥ 已過時間)`，取代雙邊重疊 |
| `extract_duration_criticality.py` | 每題一次 LLM，判定時間關鍵性 → `datasets/<ds>/criticality.json` |
| `extract_answer_type.py` | 每題一次 LLM，判定答案類型 → `datasets/<ds>/answer_type.json` |

兩個抽取腳本**只讀 vignette、不讀選項**——由選項推導的權重等同把答案送進檢索端。

### 渲染

`render_from_pool.py` —— 所有旋鈕：

| 環境變數 | 預設 | 作用 |
|---|---|---|
| `TOP_K` | 10 | 進 prompt 的候選數 |
| `LAMBDA` / `MU` | 0.3 / 0.08 | 固定係數（`JUDGE` 未設時） |
| `JUDGE` | — | `llm` / `dispersion` / `magnitude` / `atype`，`+` 可組合 |
| `NORMALIZE` / `MAG_NORM` | 1 / median | 判斷層的正規化方式 |
| `ABSTAIN` | shipped | 無時間證據時退回基準（`judged` 則交由判斷層） |
| `BC_SRC` | overlap | `onesided` 改用單邊相容性 |
| `DELTA` | 0 | 多樣性懲罰（候選之間） |
| `NOVELTY` | 0 | 題幹重疊懲罰（候選 vs 題幹） |
| `MAX_HOP` | — | 硬性 hop 上限 |
| `QUOTA_DISEASE` / `QUOTA_OTHER` | 7 / 3 | 角色配額 |
| `SAMPLE` | — | `random` 隨機對照 |
| `TEMPLATE_FILE` | — | 換 prompt 模板 |

變體名稱自動編碼所有設定（如 `walker__k10_jmagnitude_n0.15_os`），不同設定不會互相覆蓋。

### 送 inference 前的檢查（不花錢）

`verify_judge.py` —— 六項：

1. 權重有限非負、無 NaN
2. **退化不變性**：無時間證據的題目，排序不得受時間判斷影響（329 應 23/23、MedBullets 133/133）
3. 證據確實改變（與基準相同 > 95% 者不值得送）
4. 判斷層權重分布合理
5. **並排讀 kg_block 實際內容**（最重要，見下）
6. prompt 顯示的分數等於實際排序用的效用

第 5 項是決定花錢與否的依據。本輪多次驗證數值指標與 accuracy 不成正比——`_os` 把 329 的
gold@10 從 14.0% 拉到 18.5%，accuracy 卻是 −0.31——所以篩選改為直接讀證據內容。

### 作答

| 檔案 | 用途 |
|---|---|
| `run_reader.py` | N 輪**連續**作答，寫出完整 trace |
| `append_run.py` | 單輪，**附加**到既有結果檔 |
| `answer_extract.py` | 從輸出抽取選項字母 |
| `metrics.py` | accuracy / macro P·R·F1 / parseable precision / unparseable |

```bash
# 篩選：先跑一次
DATASET=medbullets METHOD=<variant> N_RUNS=1 RESULTS_DIR=results/screen \
  python3 pipeline/run_reader.py

# 晉級：補兩次獨立呼叫，不重跑已付費的那次
DATASET=medbullets METHOD=<variant> RESULTS_DIR=results/screen \
  python3 pipeline/append_run.py
```

---

## 疊代協定

1. **改 `judged_utility.py` 或 `render_from_pool.py`**
2. **渲染** — 幾秒，不花錢
3. **`verify_judge.py`** — 六項檢查，特別是第 2 項（退化不變性）與第 5 項（讀內容）
4. **N=1 篩選** — 落在最佳者 1.5pp 內的才晉級
5. **用 `append_run.py` 補到 3 次獨立呼叫** — 不重跑，且獨立呼叫才有正確的誤差估計
6. **同一批次內比較** — 跨批次的列不可相減

### 為什麼第 5 步要用獨立呼叫

同一份 prompt（逐字元驗證相同）：

| 量測方式 | 每輪正確數 | std |
|---|---|---|
| 連續三輪 | 244 / 243 / 243 | 0.15 |
| 三次獨立呼叫 | 255 / 248 / 239 | **2.13** |

相差 14 倍，且結論相反。308 題中有 42 題（13.6%）的預測字母在獨立呼叫間會改變。

---

## 報表

| 檔案 | 內容 |
|---|---|
| `old_prompt_report.py` | 各方法的 accuracy / macro / parseable，標注來源批次與混批警告 |
| `metrics_report.py` | 依 dataset 與 prompt 變體的完整指標 |
| `critical_subset_report.py` | temporal-critical 子集，以 vanilla 的自身 lift 為基準 |
| `prompt_ab_report.py` | 新舊 prompt 對照（證據相同，僅措辭不同） |
| `verify_judge.py` | 判斷層的檢查與權重分布 |

`old_prompt_report.py` 的 `FILE_ALIAS` 記錄了兩處替換及理由：

- `raw_1hop/medbullets` 的 81.06 是三次同條件 N=3（78.25 / 79.55 / 79.65）中最高的一次，
  改用中間值
- `medrag/medbullets` 為首次正式量測（先前驅動程式未存 prompt，凍存為 n=0）

---

## 結果目錄

| 目錄 | 內容 |
|---|---|
| `results/old_prompt` | baseline 的既有紀錄（08-11 ~ 08-13）|
| `results/param_sweep_n3` | 早期 13 個變體 |
| `results/judged` / `judged_v2` | 判斷層，`v2` 為退化不變性修正後 |
| `results/round3` | 新訊號（單邊 bc、atype、abjudged、正規化消融）|
| `results/screen` | N=1 篩選 |
| `results/promote` / `p2` | 晉級驗證 |
| `results/spaced` | **獨立呼叫累積**（唯一誤差估計可信的）|

跨目錄的數字不可直接相減。

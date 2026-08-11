# Dataset 統計 與 Temporal-Critical 實驗驗證

本文件涵蓋兩件事：
1. **§1** ReinRAG 式的 dataset statistical analysis
2. **§2–§5** 用**實驗**（而非關鍵字或 LLM 標籤）逐題確認每一題是否真的 duration-critical

相關程式：

| 檔案 | 用途 |
|---|---|
| `dataset_statistics.py` | 統計表 → `datasets/statistics.json` |
| `build_new_dataset.py` | 從外部來源建 candidate benchmark（MedBullets / MedMCQA / MMLU-Med） |
| `verify_duration_critical.py` | **主驗證器**：MCQ 反事實實驗 |
| `verify_openended_probe.py` | 第二臂：自由文本診斷探針（檢查 MCQ 格式是否遮蔽效應） |
| `analyze_verification.py` | 彙整判定 + 與舊標籤交叉比對 + 門檻敏感度 |
| `results_by_verified_subset.py` | 用驗證後的子集重切 reader 結果 |

---

## 1. Statistical Analysis

```bash
python3 pipeline/dataset_statistics.py 329 1273 medbullets medmcqa mmlu
```

| 指標 | MedQA 329 | MedQA 1273 | MedBullets | MedMCQA | MMLU-Med |
|---|---:|---:|---:|---:|---:|
| # Questions | 329 | 1273 | 305 | 342 | 256 |
| Avg. question length (chars) | 804.0 | 745.6 | **918.6** | 405.5 | 664.5 |
| Median question length (chars) | 780 | 715 | 924 | 366 | 612 |
| Avg. question length (words) | 131.8 | 120.6 | 150.3 | 66.1 | 108.0 |
| Avg. # options | 5.65 | 5.00 | 5.00 | 4.00 | 4.00 |
| Avg. option length (words) | 2.8 | 3.5 | 2.8 | 2.9 | 3.7 |
| Avg. answer length (words) | 2.8 | 3.6 | 3.0 | 3.0 | 3.7 |
| Vocabulary size | 5756 | 12375 | 5521 | 4486 | 4653 |
| # with numeric duration | 306 (93.0%) | 659 (51.8%) | 173 (56.7%) | 234 (68.4%) | 166 (64.8%) |
| Median duration (days) | 14.0 | 10.0 | 7.0 | 21.0 | 7.0 |
| Avg. # extracted symptoms | 5.87 | 4.78 | — | — | — |
| Avg. # LLM-DDx seeds | 11.96 | 11.84 | — | — | — |
| Avg. # query entities | 13.77 | 11.89 | — | — | — |

> 最後三列是 retrieval 的輸入（symptoms / LLM-DDx seeds / query entities），
> 新 dataset 還沒跑過 retrieval pipeline，所以是空的。

**Duration 分佈**

| bucket | 329 | 1273 | MedBullets | MedMCQA | MMLU-Med |
|---|---:|---:|---:|---:|---:|
| `<1d` | 51 (16%) | 104 (8%) | 43 (14%) | 23 (7%) | 38 (15%) |
| `1-6d` | 77 (23%) | 171 (13%) | 36 (12%) | 59 (17%) | 42 (16%) |
| `1-4wk` | 55 (17%) | 120 (9%) | 38 (12%) | 39 (11%) | 29 (11%) |
| `1-6mo` | 80 (24%) | 158 (12%) | 34 (11%) | 58 (17%) | 33 (13%) |
| `6-12mo` | 20 (6%) | 55 (4%) | 7 (2%) | 18 (5%) | 15 (6%) |
| `>1yr` | 23 (7%) | 51 (4%) | 15 (5%) | 37 (11%) | 9 (4%) |
| none | 23 (7%) | **614 (48%)** | 132 (43%) | 108 (32%) | 90 (35%) |

只有 **329 幾乎每題都有 duration**（93%）——它是為了 duration 而篩過的。
其餘四個未篩選的資料集都有 32–48% 的題目根本沒有 duration 可談。

> `days == 0`（「今天」「剛剛」這類超急性）是有效值，不是缺值。
> 用 `days > 0` 過濾會把它們默默丟進 none，低估急性題數（1273 少算 3 題）。

**Temporal role**：329 = chief_complaint 166 / onset_to_presentation 130 / none 23 / past_medical_history 10；
1273 = none 614 / chief_complaint 509 / onset_to_presentation 150。

**Answer 分佈**：329 有 A–K 共 10 個字母（A19.1 B19.8 C18.2 D14.9 E17.0 F6.7 G2.7 H0.9 J0.3 K0.3）；
1273 只有 A–E 且平衡（21.4 / 21.8 / 19.8 / 21.1 / 15.9）。

> ⚠ 病人自述的 duration 在 `datasets/<ds>/durations.json`，**不是** benchmark 裡的 `duration_info*`
> 欄位——那些是每個選項的**疾病病程先驗**（retrieval 輸入），拿來算病人 duration 會全部得到 0。

---

## 2. 為什麼要重做驗證

原本的 `filter_train_duration_critical.py` 是**判斷式**的：問 LLM「這題是不是 duration-critical？」
在 100 題的抽查檔（`perturbation_results.jsonl`）裡，這個判斷只有 40% 通過更深入的反事實分析。
產生那個 `is_truly_duration_critical` 欄位的 prompt 已經不在 codebase 裡了。

所以改成**測試**這個性質，而不是**標註**它。

---

## 3. 驗證設計（`verify_duration_critical.py`）

每題三階段，全部呼叫都存檔可事後稽核：

```
A. CONTROL   原題獨立作答 5 次；gold 命中須 ≥4/5
             （答不出原題 → 擾動後翻不翻都不構成證據 → control_fail）

B. PERTURB   只改 duration（急↔慢，幅度要大且明確）
             用 word-level diff 驗證「只動了一個局部區塊」：
             所有 diff hunk 須落在 ~20 字的視窗內、改動 ≤25 字、且提到時間單位
             → 否則 perturb_fail（改寫順手動到症狀/檢驗值/年齡，翻轉就被汙染了）

C. TEST      對擾動後題目重新獨立作答 5 次，**完全不給 gold**（避免錨定，
             所以 B 和 C 必須是分開的呼叫）
```

判定：

| verdict | 條件 |
|---|---|
| `duration_critical` | control ≥4/5 命中 gold，perturbed ≤1/5 命中 gold |
| `not_duration_critical` | control ≥4/5，perturbed 仍 ≥4/5 |
| `ambiguous` | perturbed 落在兩門檻之間（不硬塞進任一桶） |
| `control_fail` | 模型答不穩原題 |
| `perturb_fail` | 沒有 duration，或改寫動到 duration 以外的東西 |

用**超級多數**（≥80% / ≤20%）而非簡單多數：temperature 1 下 gold 命中 3/5 的題目，光是重抽樣就會「翻轉」。

```bash
DATASET=329 MODEL=gpt-5.4-mini N_SAMPLES=5 WORKERS=16 python3 pipeline/verify_duration_critical.py
python3 pipeline/analyze_verification.py            # 全部 dataset
```

輸出 `verification/<ds>_duration_critical_<model>.json`（可續跑）。

---

## 4. 結果

### 4.1 主判定

| | MedQA 329 | MedBullets | MedMCQA | MMLU-Med | MedQA 1273 |
|---|---:|---:|---:|---:|---:|
| duration_critical | 14 (4.3%) | 7 (2.3%) | 4 (1.2%) | 3 (1.2%) | 14 (1.1%) |
| not_duration_critical | 209 (63.5%) | 169 (55.4%) | 269 (78.7%) | 187 (73.0%) | 667 (52.4%) |
| ambiguous | 8 (2.4%) | 9 (3.0%) | 8 (2.3%) | 4 (1.6%) | 32 (2.5%) |
| control_fail | 66 (20.1%) | 79 (25.9%) | 39 (11.4%) | 13 (5.1%) | 168 (13.2%) |
| perturb_fail | 32 (9.7%) | 41 (13.4%) | 22 (6.4%) | 49 (19.1%) | **392 (30.8%)** |
| **在有結論題目中的比例** | **14/223 = 6.3%** | **7/176 = 4.0%** | **4/273 = 1.5%** | **3/190 = 1.6%** | **14/681 = 2.1%** |

1273 的 `perturb_fail` 高達 30.8%，因為它有 48% 的題目根本沒有 duration 可以改
（見 §1 duration bucket 的 `none` 欄）——這是未經篩選的完整 MedQA test 的樣貌。

**門檻敏感度**（用已存的逐次作答重算，不再花 API）：

| control ≥ / perturb ≤ | 329 | MedBullets | MedMCQA | MMLU-Med | 1273 |
|---|---:|---:|---:|---:|---:|
| 5 / 0 | 4.6% | 2.7% | 0.8% | 1.1% | 1.2% |
| 4 / 1（採用） | 6.3% | 4.0% | 1.5% | 1.6% | 2.1% |
| 3 / 2 | 7.8% | 5.4% | 2.5% | 2.1% | 4.1% |
| 3 / 1 | 6.2% | 3.8% | 1.4% | 1.6% | 2.0% |

比例低**不是門檻造成的**——五個 dataset、四組門檻都一樣低。

值得注意：**329 的 6.3% 是 1273 的 2.1% 的三倍**。也就是說原本的篩選流程確實有富集效果，
只是遠不如標籤宣稱的那麼乾淨（宣稱是 100%，實測 6.3%）。

`control_fail` 的高低本身就有資訊：329 (20.1%) 和 MedBullets (25.9%) 明顯高於
MedMCQA (11.4%) 和 MMLU-Med (5.1%)——前兩者是為難度挑選/設計的，後兩者較接近事實回憶。
這也解釋了 §6 為什麼 walker 在 329 上的優勢遠大於 1273：**retrieval 只有在 reader 本來就答不出來時才有發揮空間**。

### 4.2 舊標籤 vs 實驗（329）

| 舊標籤 | critical | not | ambig | 有結論中的比例 |
|---|---:|---:|---:|---:|
| `verdict = truly_tc` | 7 | 99 | 3 | 6.6% (n=106) |
| `verdict = borderline_tc` | 7 | 101 | 5 | 6.5% (n=108) |
| `audit_source = human_cf`（人工確認 46 題） | 4 | 13 | 3 | **23.5% (n=17)** |
| `audit_source = llm_auto_m325` | 5 | 74 | 2 | 6.3% (n=79) |
| `audit_source = llm_auto_v2` | 5 | 113 | 3 | 4.2% (n=118) |

- `truly_tc` / `borderline_tc` 的區分**毫無鑑別力**（6.6% vs 6.5%）——那個 LLM 標籤是雜訊。
- 唯一撐得住反事實檢驗的是**人工確認的那批**（4–5 倍富集）。

---

## 5. 第二臂：MCQ 格式遮蔽了多少？（`verify_openended_probe.py`）

主驗證只能看到**選項集允許發生的翻轉**。MedQA 的 distractor 很少是 gold 的
「急性/慢性對應版本」，所以一題可以真的依賴 duration、卻因為**沒有東西可以翻過去**
而被判成 `not_duration_critical`。這跟「duration 不重要」長得一模一樣，但意義完全不同：
一個是醫學的性質，一個是 benchmark 格式的性質。

作法：對 `not_duration_critical` 的題目，把同一組原題／擾動題**拿掉選項**重問
「最可能的診斷是什麼」，再用兩道 judge：

1. 兩個診斷是不是同一個病？（同義詞、縮寫、細緻度差異算同一個）
2. 若不同 → **兩者的立即處置是否不同**（不同緊急度／不同第一線檢查／不同第一線治療）？
   `SAME` = 同一實體換句話說或只差分期；`INVALID` = 至少一邊不是診斷（是檢查、藥物、病媒）

第 2 道是必要的：`anal fissure → chronic anal fissure` 只是同一實體重述，而
`acute mesenteric ischemia → chronic mesenteric ischemia` 雖然共用字根卻是處置不同的兩個實體。
單看名稱分不開這兩者。

```bash
DATASET=329 SAMPLE=0 python3 pipeline/verify_openended_probe.py    # SAMPLE=0 = 全部
```

### 結果（**全量**，非抽樣）

| | 329 | MedBullets | MedMCQA | MMLU-Med | 1273 |
|---|---:|---:|---:|---:|---:|
| `not_duration_critical` 題數 | 209 | 169 | 269 | 187 | 667 |
| 自由文本診斷有變 | 64 (30.6%) | 56 (33.1%) | 78 (29.0%) | 61 (32.6%) | 180 (27.0%) |
| ├ 處置不同（DIFFERENT） | **40** | **27** | **39** | **33** | **83** |
| ├ 同一實體重述（SAME） | 22 | 26 | 36 | 22 | 87 |
| └ 不是診斷（INVALID） | 2 | 3 | 3 | 6 | 10 |
| **嚴格計算：被選項遮蔽的真實 duration 效應** | **40/207 = 19.3%** | **27/166 = 16.3%** | **39/266 = 14.7%** | **33/181 = 18.2%** | **83/657 = 12.6%** |

五個獨立 dataset 幾乎一致。實例（329）：

| duration 變化 | 原診斷 → 擾動後診斷 |
|---|---|
| 2 days → 2 months | Acute mesenteric ischemia → **Chronic** mesenteric ischemia |
| 20 min → 6 months | **Unstable** angina → **Stable** exertional angina |
| 1 month → 2 days | Tuberculosis → Histoplasmosis |
| 2 weeks → 2 years | Pyogenic granuloma → Squamous cell carcinoma |
| 4 months → 4 days | Iron deficiency anemia → Paroxysmal nocturnal hemoglobinuria |
| — | Acute dystonic reaction → Tardive dyskinesia |
| — | Delusional disorder → Brief psychotic disorder（DSM 時長準則） |

### 5.1 合併結論（兩臂相加，全部為實測值）

| | 329 | MedBullets | MedMCQA | MMLU-Med | 1273 |
|---|---:|---:|---:|---:|---:|
| 有結論的題目 | 223 | 176 | 273 | 190 | 681 |
| MCQ **可見**的 duration-critical | 14 (6.3%) | 7 (4.0%) | 4 (1.5%) | 3 (1.6%) | 14 (2.1%) |
| MCQ **遮蔽**但處置不同 | 40 | 27 | 39 | 33 | 83 |
| **真正依賴 duration 合計** | **54/223 = 24.2%** | **34/176 = 19.3%** | **43/273 = 15.8%** | **36/190 = 18.9%** | **97/681 = 14.2%** |
| MCQ 格式藏起的比例 | 40/54 = **74%** | 27/34 = **79%** | 39/43 = **91%** | 33/36 = **92%** | 83/97 = **86%** |

五個獨立 dataset 一致：**MCQ 格式藏起了七成四到九成二的 duration 效應**。
被遮蔽的比例（12.6–19.3%）跨資料集非常穩定；差異幾乎全來自 MCQ **可見**的那一段
（6.3% / 4.0% / 1.5% / 1.6% / 2.1%），也就是選項集有多常剛好放了時間對應的鑑別診斷——
那是 **benchmark 格式**的變數，不是醫學的變數。

> 這句話值得寫進論文：**在任何 MedQA 型 MCQ 上，duration 的效應有七成以上在測量前就被
> 選項集消掉了**。所以 §6 那些「duration 對 MCQ accuracy 是 net-zero」的 ablation，
> 不能當成「duration 不重要」的證據，只能當成「MCQ 不是量測 duration 效應的正確 testbed」。
這對論文的意義是——「MCQ accuracy 上 duration 是 net-zero」不能推論成「duration 不重要」，
只能推論成「MCQ 不是量測 duration 效應的正確 testbed」。

---

## 6. 下游重切（`results_by_verified_subset.py`）

```bash
DATASET=329 RESULTS_DIR=results/round2_intentfree python3 pipeline/results_by_verified_subset.py
```

329 · gpt-5.4-mini · walker 減各 baseline（percentage points）：

| baseline | duration_critical | not_duration_critical | control_fail | ALL |
|---|---:|---:|---:|---:|
| walker_interval | +0.0 (n=14) | +0.9 (n=209) | **+7.0** (n=66) | +1.8 |
| raw_1hop | −3.8 (n=14) | +1.7 (n=209) | **+6.6** (n=66) | +2.8 |
| raw_2hop | −6.2 (n=14) | +1.5 (n=209) | **+6.6** (n=66) | +2.0 |
| hykge | −3.8 (n=14) | +2.0 (n=209) | **+14.6** (n=66) | +4.5 |

walker 的優勢集中在 `control_fail`（reader 零樣本答不出的難題），
在 `duration_critical` 子集上反而是負的。

1273 · 同一張表：

| baseline | duration_critical | not_duration_critical | control_fail | ALL |
|---|---:|---:|---:|---:|
| walker_interval | −2.4 (n=14) | −0.3 (n=667) | −2.0 (n=168) | −0.6 |
| raw_1hop | +2.4 (n=14) | −0.0 (n=667) | −0.6 (n=168) | −0.2 |
| raw_2hop | −7.1 (n=14) | +0.1 (n=667) | +1.2 (n=168) | +0.2 |
| hykge | −4.8 (n=14) | +1.5 (n=667) | −0.8 (n=168) | +1.0 |

### 6.1 這解釋了「為什麼 1273 上我們比 interval 和 1-hop 低」

1273 上 walker 對所有 baseline 的差距都塌進 ±2.4pp，**包括 `control_fail` 那一欄**——
而在 329 上那一欄是 +6.6 ~ +14.6pp。差別不在方法，在**兩個 dataset 的難度組成**：

| | 329 | 1273 |
|---|---:|---:|
| `control_fail`（reader 零樣本答不出） | 20.1% | 13.2% |
| `not_duration_critical` 上的 reader 準確率 | 97.5% | 97.7% |

兩個 dataset 的「容易題」都已經在 97%+ 天花板，retrieval 動不了；
真正能改善的只有 `control_fail` 那一段，而 329 的那一段大了 1.5 倍
（329 是照 `n_models_vanilla_wrong` 挑過難度的，1273 是未篩選的完整 test set）。
**這不是 duration 建模的問題，是 headroom 的問題。**

**必須誠實標註的兩個 caveat**：
1. `duration_critical` 只有 n=14，只能看方向，談不上顯著性。
2. `control_fail` 是用 gpt-5.4-mini 定義的，而 reader 也是 gpt-5.4-mini，
   所以那個子集本質上就是「這個 reader 覺得難的題」。不過所有方法都在同一子集上比，walker 仍最高。

---

## 7. 新 dataset（`build_new_dataset.py`）

```bash
python3 pipeline/build_new_dataset.py medbullets
python3 pipeline/build_new_dataset.py medmcqa
python3 pipeline/build_new_dataset.py mmlu
```

| | 保留 / 原始 | avg chars | # options | 說明 |
|---|---:|---:|---:|---|
| **MedBullets (op5)** | 305 / 308 | 918.6 | 5.00 | USMLE Step 2&3 vignette（[ChallengeClinicalQA](https://github.com/HanjieChen/ChallengeClinicalQA)），比 MedQA 329 還長，風格最接近，且不在 MedQA 訓練分佈中 |
| **MedMCQA** | 342 / 1168 | 405.5 | 4.00 | 只有 `test.jsonl` 有選項與標籤（train/val 沒有選項，無法作答）；很多是 factoid 而非 vignette，已加 vignette + ≥200 字元篩選 |
| **MMLU professional_medicine** | 256 / 272 | 665 | 4.00 | 目前論文最常一起報的醫學 benchmark；USMLE 風格 vignette，題數少但品質整齊。用 HF datasets-server 抓，不需裝 `datasets` |

**評估過但排除**

| | 排除原因 |
|---|---|
| JAMA Clinical Challenge | ChallengeClinicalQA 只放 `jama_links.json` + scraper，文章在訂閱牆後，資料不能重散布也不該在此抓取 |
| PubMedQA / BioASQ | abstract 層級的 yes/no/maybe，沒有病人 vignette、沒有可擾動的 duration，結構上不適用 |

> `build_new_dataset.py` 的關鍵字篩選**只決定哪些題目值得付錢去測**，
> 它的輸出叫 candidate pool，**絕對不可以**當成 duration-critical 回報——
> 那是 `verify_duration_critical.py` 的工作。

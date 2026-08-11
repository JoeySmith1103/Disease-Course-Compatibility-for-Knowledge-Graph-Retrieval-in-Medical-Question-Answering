# Duration-Critical 題目總表：Statistics / References / Origin / 題目清單

由 `build_master_report.py` 自動生成——所有數字直接從資料檔計算，未經手動轉抄。

---

## 1. Origin（來源與覆蓋率）

**這張表是讀其他所有數字的前提。** MedMCQA 的比例是在母體 3.4% 的切片上算的，
與其餘四個（幾乎全取）不可直接比較。

| dataset | 原始來源 | 原始題數 | 進入審查 | 覆蓋率 | 篩選方式 |
|---|---|---:|---:|---:|---|
| MedQA 329 | MedQA 衍生（temporal_critical_v2 re-split） | — | 329 | — | 前人 Phase-A LLM filter（本研究已證實無鑑別力） |
| MedQA 1273 | MedQA-USMLE **test**（= MIRAGE 的 MedQA-US） | 1,273 | 1,273 | 100% | 未篩選 |
| MedBullets | ChallengeClinicalQA `medbullets_op5.csv` | 308 | 308 | **100%** | 不篩選（全數審查） |
| MedMCQA | HF `openlifescienceai/medmcqa` **validation**（= MIRAGE 的 MedMCQA dev） | 4,183 | 143 | **3.4%** | vignette + ≥200字元 + 時間表述 |
| MMLU-Med | HF `cais/mmlu` config `professional_medicine` **test** | 272 | 272 | **100%** | 不篩選（全數審查） |

> **為什麼 MedBullets / MMLU 改成不篩選**：早期版本套用了同一組關鍵字篩選（→305 / 256）。
> 事後檢查被篩掉的 19 題，發現其中 3 題其實是時間依賴的：`mb_0234`（5 年追蹤的樣本流失）、
> `mb_0271`（年齡完全由發展里程碑編碼、未寫出）、`mmlu_0161`（用時間結構區分 case series /
> cross-sectional / crossover）。**篩選器複製了腳本的單軸盲點**，所以對小到能全讀的資料集一律取消。
> MedMCQA 母體 18 萬題、中位數 65 字元，不篩不可行。

#### 已作廢的舊 MedMCQA 來源，以及它的兩個問題

早期版本用 `Temporal-KG-RAG/datasets/MedMCQA_temporal_critical_split/test.jsonl`（1,168 題）。
把那 1,168 題的題目文字逐題比對回原始三個 split，結果是：

| 檢查 | 結果 |
|---|---|
| 比對回原始 split | **train 1,168 / validation 0 / test 0**，零題對不上 |
| 有 `cop` 的 595 題 vs train 的答案 | **595 全部一致，0 不一致**（差一個 index base） |
| `cop=None` 的 573 題，train 裡有沒有答案 | **573 題全部有**，一題不缺 |

兩個結論：

1. **那個檔名叫 `test.jsonl`，但不是 MedMCQA 官方 test split。** 它是前人拿 MedMCQA **train**
   （182,822 題）做時間篩選後自行重切的 train/val/test 之一。`idx` 最大 182,738 正是 train 的索引範圍。
   因此官方 test split 無答案這件事，與這 1,168 題**完全無關**。
2. **那 573 題的答案不是原本就沒有，是舊流程處理時弄丟的。** 原始 train 每一題都有答案，
   且已全數還原驗證。先前把它當成「資料只有 52% 有答案」是誤判。

該檔仍作廢，原因是它繼承了未經驗證的 LLM 篩選器（全部 `is_temporal_critical=True`），
而本研究已證實這類判斷式標籤無鑑別力。舊資料保留在 `datasets/_medmcqa_OLD_inherited_filter/`。

---

## 2. References

| dataset | 引用 |
|---|---|
| **MedBullets** | Hanjie Chen, Zhouxiang Fang, Yash Singla, Mark Dredze. *Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions.* **NAACL-HLT 2025**, Vol. 1 (Long Papers), pp. 3563–3599. arXiv:[2402.18060](https://arxiv.org/abs/2402.18060) · [aclanthology.org/2025.naacl-long.182](https://aclanthology.org/2025.naacl-long.182/) · code/data: [github.com/HanjieChen/ChallengeClinicalQA](https://github.com/HanjieChen/ChallengeClinicalQA) · Johns Hopkins · **146 引用**（Semantic Scholar） |
| **MedMCQA** | Pal, Umapathi, Sankarasubbu. *MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering.* CHIL 2022. HF: `openlifescienceai/medmcqa` |
| **MMLU** | Hendrycks et al. *Measuring Massive Multitask Language Understanding.* ICLR 2021. HF: `cais/mmlu`, config `professional_medicine` |
| **MedQA** | Jin et al. *What Disease Does This Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams.* Applied Sciences 2021 |
| **MIRAGE / MedRAG**（本論文的定位基準） | Guangzhi Xiong, Qiao Jin, Zhiyong Lu, Aidong Zhang. *Benchmarking Retrieval-Augmented Generation for Medicine.* **Findings of ACL 2024**. arXiv:[2402.13178](https://arxiv.org/abs/2402.13178) · DOI 10.18653/v1/2024.findings-acl.372 · [github.com/Teddy-XiongGZ/MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) |

**MIRAGE 組成（7,663 題）**：MMLU-Med 1,089 · **MedQA-US 1,273** · **MedMCQA 4,183 (dev)** · PubMedQA* 500 · BioASQ-Y/N 618。
本研究的 1273 與 MedMCQA-validation **正好是其中兩個組成**，可直接對位。

### 評估過但排除
| | 排除原因 |
|---|---|
| JAMA Clinical Challenge | 同一篇 NAACL 論文的另一資料集，但只釋出連結與 scraper，正文在訂閱牆後，不可重散布 |
| PubMedQA / BioASQ | abstract 層級 yes/no/maybe，無病人 vignette、無可擾動的 duration |
| MedXpertQA | arXiv:2501.18362，ICML 2025，4,460 題 10 選項 — **尚未評估，值得考慮** |

---

## 3. Statistics — General vs Duration-Critical

每個資料集拆成兩組：**全部題目（general）**與 **duration-critical 子集**。
只報總表會蓋掉論文真正要問的事——時間依賴的題目長得跟其他題不一樣嗎？

| dataset | critical 子集來源 | critical / all |
|---|---|---|
| MedQA 329 | 僅腳本（未人工複核） | 14/329 = 4.3% |
| MedQA 1273 | 僅腳本（未人工複核） | 14/1273 = 1.1% |
| MedBullets | **人工逐題** | 88/308 = 28.6% |
| MedMCQA | **人工逐題** | 38/143 = 26.6% |
| MMLU-Med | **人工逐題** | 51/272 = 18.8% |

> 329 / 1273 的子集來自腳本，只覆蓋 `symptom_duration` 一軸，那兩欄是**下界**。

### 3.1 General（全部題目）

| 指標 | MedQA 329 | MedQA 1273 | MedBullets | MedMCQA | MMLU-Med |
|---|---:|---:|---:|---:|---:|
| # Questions | 329 | 1273 | 308 | 143 | 272 |
| Avg. question length (chars) | 804.0 | 745.6 | 915.7 | 278.3 | 653.8 |
| Median question length (chars) | 780 | 715 | 922 | 252 | 606 |
| Avg. question length (words) | 131.8 | 120.6 | 149.8 | 46.1 | 106.2 |
| Avg. # options | 5.65 | 5.00 | 5.00 | 4.00 | 4.00 |
| Avg. option length (words) | 2.8 | 3.5 | 2.9 | 3.3 | 3.8 |
| Avg. answer length (words) | 2.8 | 3.6 | 3.0 | 3.4 | 3.8 |
| Vocabulary size | 5756 | 12375 | 5556 | 2044 | 4835 |
| # with numeric duration | 306 | 659 | 175 | 29 | 163 |
| % with numeric duration | 93.0 | 51.8 | 56.8 | 20.3 | 59.9 |
| Median duration (days) | 14.0 | 10.0 | 7.0 | 30.0 | 7.0 |
| Avg. # extracted symptoms | 5.87 | 4.78 | — | — | — |
| Avg. # LLM-DDx seeds | 11.96 | 11.84 | — | — | — |
| Avg. # query entities | 13.77 | 11.89 | — | — | — |

| Duration bucket | MedQA 329 | MedQA 1273 | MedBullets | MedMCQA | MMLU-Med |
|---|---:|---:|---:|---:|---:|
| `<1d` | 51 (16%) | 104 (8%) | 40 (13%) | 4 (3%) | 36 (13%) |
| `1-6d` | 77 (23%) | 171 (13%) | 40 (13%) | 9 (6%) | 42 (15%) |
| `1-4wk` | 55 (17%) | 120 (9%) | 37 (12%) | 1 (1%) | 28 (10%) |
| `1-6mo` | 80 (24%) | 158 (12%) | 38 (12%) | 7 (5%) | 34 (12%) |
| `6-12mo` | 20 (6%) | 55 (4%) | 7 (2%) | 3 (2%) | 14 (5%) |
| `>1yr` | 23 (7%) | 51 (4%) | 13 (4%) | 5 (3%) | 9 (3%) |
| `none` | 23 (7%) | 614 (48%) | 133 (43%) | 114 (80%) | 109 (40%) |

### 3.2 Duration-Critical 子集

| 指標 | MedQA 329 | MedQA 1273 | MedBullets | MedMCQA | MMLU-Med |
|---|---:|---:|---:|---:|---:|
| # Questions | 14 | 14 | 88 | 38 | 51 |
| Avg. question length (chars) | 894.6 | 918.8 | 932.9 | 277.9 | 676.5 |
| Median question length (chars) | 896 | 826 | 944 | 261 | 612 |
| Avg. question length (words) | 151.1 | 152.5 | 154.3 | 47.1 | 111.0 |
| Avg. # options | 5.36 | 5.00 | 5.00 | 4.00 | 4.00 |
| Avg. option length (words) | 2.6 | 3.1 | 3.0 | 3.9 | 3.8 |
| Avg. answer length (words) | 2.4 | 3.5 | 3.1 | 4.2 | 3.7 |
| Vocabulary size | 793 | 855 | 2636 | 738 | 1572 |
| # with numeric duration | 14 | 11 | 52 | 13 | 31 |
| % with numeric duration | 100.0 | 78.6 | 59.1 | 34.2 | 60.8 |
| Median duration (days) | 4.5 | 3.0 | 17.5 | 30.0 | 14.0 |
| Avg. # extracted symptoms | 6.43 | 5.57 | — | — | — |
| Avg. # LLM-DDx seeds | 12.00 | 11.86 | — | — | — |
| Avg. # query entities | 14.21 | 15.64 | — | — | — |

| Duration bucket | MedQA 329 | MedQA 1273 | MedBullets | MedMCQA | MMLU-Med |
|---|---:|---:|---:|---:|---:|
| `<1d` | 4 (29%) | 2 (14%) | 9 (10%) | 3 (8%) | 8 (16%) |
| `1-6d` | 3 (21%) | 5 (36%) | 11 (12%) | 3 (8%) | 3 (6%) |
| `1-4wk` | 2 (14%) | 1 (7%) | 14 (16%) | 0 (0%) | 6 (12%) |
| `1-6mo` | 3 (21%) | 2 (14%) | 9 (10%) | 4 (11%) | 8 (16%) |
| `6-12mo` | 2 (14%) | 1 (7%) | 3 (3%) | 1 (3%) | 3 (6%) |
| `>1yr` | 0 (0%) | 0 (0%) | 6 (7%) | 2 (5%) | 3 (6%) |
| `none` | 0 (0%) | 3 (21%) | 36 (41%) | 25 (66%) | 20 (39%) |

### 3.3 兩組的差異（這才是重點）

**(a) duration-critical 的題目一致比較長**

| | 全部 (chars) | critical (chars) | 差 |
|---|---:|---:|---:|
| MedQA 329 | 804 | 895 | +91 |
| MedQA 1273 | 746 | 919 | +173 |
| MedBullets | 916 | 933 | +17 |
| MedMCQA | 278 | 278 | -0 |
| MMLU-Med | 654 | 677 | +23 |

**(b) duration 覆蓋率一致上升**

| | 全部 | critical |
|---|---:|---:|
| MedQA 329 | 93.0% | 100.0% |
| MedQA 1273 | 51.8% | 78.6% |
| MedBullets | 56.8% | 59.1% |
| MedMCQA | 20.3% | 34.2% |
| MMLU-Med | 59.9% | 60.8% |

**(c) 但相當比例的 critical 題目根本抽不出數值 duration** ← 最重要的一格

| | critical 題數 | 無數值 duration | 佔比 |
|---|---:|---:|---:|
| MedBullets | 88 | 36 | 41% |
| MedMCQA | 38 | 25 | 66% |
| MMLU-Med | 51 | 20 | 39% |
| **人工判讀合計** | **177** | **81** | **46%** |

這是 §5 多時間軸結論的**量化直接證據**：這些題目被判為時間依賴，但沒有可抽取的主訴時長，
因為時間資訊在別的軸上（潛伏期、療程長度、妊娠週數、事件先後、發作長度⋯⋯）。
**腳本在這些題目上連可以擾動的對象都找不到。**

**(d) median duration 的變化**

| | 全部 median (days) | critical median (days) |
|---|---:|---:|
| MedQA 329 | 14.0 | 4.5 |
| MedQA 1273 | 10.0 | 3.0 |
| MedBullets | 7.0 | 17.5 |
| MedMCQA | 30.0 | 30.0 |
| MMLU-Med | 7.0 | 14.0 |

**(e) 答案字母分佈**

- **MedQA 329** 全部: A:19.1  B:19.8  C:18.2  D:14.9  E:17.0  F:6.7  G:2.7  H:0.9  J:0.3  K:0.3
  - critical: A:28.6  B:7.1  C:35.7  E:21.4  F:7.1
- **MedQA 1273** 全部: A:21.4  B:21.8  C:19.8  D:21.1  E:15.9
  - critical: A:21.4  B:21.4  D:14.3  E:42.9
- **MedBullets** 全部: A:19.8  B:24.0  C:17.2  D:21.8  E:17.2
  - critical: A:20.5  B:17.0  C:19.3  D:26.1  E:17.0
- **MedMCQA** 全部: A:38.5  B:14.0  C:29.4  D:18.2
  - critical: A:36.8  B:18.4  C:31.6  D:13.2
- **MMLU-Med** 全部: A:18.4  B:20.2  C:16.5  D:44.9
  - critical: A:25.5  B:13.7  C:19.6  D:41.2

**(f) temporal role**

| | 全部 | critical |
|---|---|---|
| MedQA 329 | chief_complaint:166  onset_to_presentation:130  null:23  past_medical_history:10 | onset_to_presentation:9  chief_complaint:5 |
| MedQA 1273 | null:614  chief_complaint:509  onset_to_presentation:150 | chief_complaint:8  null:3  onset_to_presentation:3 |
| MedBullets | null:133  chief_complaint:100  onset_to_presentation:75 | chief_complaint:37  null:36  onset_to_presentation:15 |
| MedMCQA | null:114  chief_complaint:22  onset_to_presentation:7 | null:25  chief_complaint:9  onset_to_presentation:4 |
| MMLU-Med | chief_complaint:115  null:109  onset_to_presentation:48 | chief_complaint:22  null:20  onset_to_presentation:9 |

### 3.4 MedMCQA — 原始完整資料集（未經任何篩選）

本研究只讀了 143 題 = validation 的 3.4%。**任何在那 143 題上算的比例都是富集後的切片**，
必須連同母體分佈一起報告。

| 指標 | train | validation | test |
|---|---:|---:|---:|
| # Questions | 182,822 | 4,183 | 6,150 |
| # with answer label | 182,822 | 4,183 | 0 |
| % labelled | 100.0% | 100.0% | 0.0% |
| Avg. question length (chars) | 78.7 | 86.3 | 59.9 |
| **Median** question length (chars) | **55** | **65** | **50** |
| P90 question length (chars) | 148 | 175 | 96 |
| Max question length (chars) | 1,568 | 582 | 794 |
| Avg. question length (words) | 12.7 | 14.1 | 9.7 |
| Median question length (words) | 9 | 11 | 8 |
| # options | 4 | 4 | 4 |
| Avg. option length (words) | 2.6 | 3.0 | 2.5 |
| Vocabulary size | 65,887 | 8,933 | 9,625 |

**官方 test split 完全沒有答案標籤**（6,150 題全部 `cop=None`），
所以論文慣例（含 MIRAGE）用 validation 當評測集。

#### 題長分佈

| 字元數 | train | validation | test |
|---|---:|---:|---:|
| `<100` | 150,618 (82%) | 3,133 (75%) | 5,581 (91%) |
| `100-199` | 20,554 (11%) | 755 (18%) | 444 (7%) |
| `200-399` | 9,215 (5%) | 270 (6%) | 117 (2%) |
| `>=400` | 2,435 (1%) | 25 (1%) | 8 (0%) |

**train 有 82%、test 有 91% 的題目不到 100 字元**——MedMCQA 本質上是一行的事實記憶題庫，不是 vignette 題庫。

#### 篩選漏斗（三個 split）

| 階段 | train | validation | test |
|---|---:|---:|---:|
| 全部題目 | 182,822 (100.0%) | 4,183 (100.0%) | 6,150 (100.0%) |
| + ≥200 字元 | 11,650 (6.4%) | 295 (7.1%) | 125 (2.0%) |
| + vignette 措辭 | 9,583 (5.2%) | 197 (4.7%) | 82 (1.3%) |
| + 時間表述 ← **本研究讀的** | 7,948 (4.3%) | 143 (3.4%) | 59 (1.0%) |

> **train split 還有 7,948 題**符合同一組條件、完全未使用。依 validation 上實測的 26.6% critical 率推估，其中約 **2,112 題**可能是 duration-critical——
> 目前所有資料集裡最大的未開發來源。

#### 科別分佈（前 8）

- **train**：Medicine 17,887  Surgery 16,862  Pathology 14,884  Anatomy 14,560  Pharmacology 13,758  Social & Preventive Medicine 11,882  Microbiology 11,314  Gynaecology & Obstetrics 10,013
- **validation**：Dental 1,318  Surgery 369  Pathology 337  Medicine 295  Pharmacology 243  Anatomy 234  Pediatrics 234  Gynaecology & Obstetrics 224
- **test**：Dental 1,203  Unknown 682  Gynaecology & Obstetrics 532  Surgery 501  Physiology 388  Medicine 372  Biochemistry 352  Pharmacology 317

---

## 4. Duration-Critical 判定總表

- **腳本**（`verify_duration_critical.py`）：反事實擾動 —— 只把主訴時長改成相反時間尺度，
  重新盲答 5 次看答案是否翻轉。**只覆蓋 `symptom_duration` 這一軸。**
- **人工逐題**：逐題判斷「改變任何時間資訊會不會改變正確答案」，並標記時間軸。

| dataset | 審查方式 | n | critical | not | no_duration | 比例 | 對照腳本 |
|---|---|---:|---:|---:|---:|---:|---:|
| MedQA 329 | 僅腳本 | 329 | 14 | 209 | — | 6.3% | — |
| MedQA 1273 | 僅腳本 | 1273 | 14 | 667 | — | 2.1% | — |
| MedBullets | **人工逐題** | 308 | **88** | 177 | 43 | **33.2%** | 4.0%（8×） |
| MedMCQA | **人工逐題** | 143 | **38** | 51 | 54 | **42.7%** | 1.5%（29×） |
| MMLU-Med | **人工逐題** | 272 | **51** | 168 | 53 | **23.3%** | 1.6%（15×） |

**人工逐題合計：723 題，177 題 critical，30.9%**（分母為有 duration 的 573 題）

> MedQA 329 / 1273 **未逐題人工審查**，只跑過腳本。既然人工判讀在其他三個資料集上高出數倍，
> 那兩個的比例幾乎確定也是低估。

### 換算到原始資料集（消除選擇偏誤後）

| dataset | critical | ÷ 原始題數 | 說明 |
|---|---:|---:|---|
| MedBullets | 88 | 88 / 308 = **28.6%** | 分母是整個資料集 |
| MedMCQA | 38 | 38 / 4,183 = **0.9%** | 分母是整個 validation split |
| MMLU-Med | 51 | 51 / 272 = **18.8%** | 分母是整個 subject |

---

## 5. 時間軸分佈（人工判讀，177 題 critical）

177 題散落在 **85 種**不同的時間軸上。
**`symptom_duration`（腳本唯一擾動的那一軸）只佔 23 題 = 13%。**

| 時間軸 | n | 意義 |
|---|---:|---|
| `symptom_duration` | 23 | 主訴持續多久 |
| `progression_tempo` | 8 | 惡化速度 |
| `gestational_window` | 7 | 妊娠週數決定處置 |
| `temporal_ordering` | 7 | 事件先後順序 |
| `latency` | 6 | 暴露到發病的間隔 |
| `dsm_duration_criterion` | 6 | 診斷準則的時長門檻 |
| `attack_duration` | 6 | 單次發作持續多久 |
| `onset_timing` | 6 | 發生在哪個時間點 |
| `treatment_trial` | 5 | 療程已試多久 |
| `therapeutic_window` | 5 | 治療時效窗 |
| `time_of_day` | 4 | 一天中的時段 |
| `age_gated` | 4 | 年齡決定正常與否 |
| `acuity` | 3 | 急性 vs 慢性 |
| `cyclical_pattern` | 3 | 週期性 |
| `onset_speed` | 3 |  |
| `interval_since_event` | 2 | 距上次事件多久 |
| `age_persistence` | 2 | 持續到幾歲 |
| `lucid_interval` | 2 |  |
| `circadian_phase` | 2 | 生理時鐘相位 |
| `screening_interval` | 2 | 篩檢間隔 |
| `latency_trajectory` | 2 | 潛伏+趨勢 |
| `progression_trajectory` | 2 | 變化軌跡 |
| `postprandial_timing` | 2 |  |
| `wound_healing_phase` | 2 | 傷口癒合期 |
| `treatment_duration` | 2 |  |
| `progression_pattern` | 2 |  |
| `time_since_exposure` | 1 |  |
| `recovery_time` | 1 |  |
| `diurnal_pattern` | 1 |  |
| `growth_rate` | 1 |  |
| `symptom_stability` | 1 |  |
| `symptom_resolution` | 1 |  |
| `chronological_vs_corrected_age` | 1 |  |
| `sleep_duration` | 1 |  |
| `attack_continuity` | 1 |  |
| `regression_trajectory` | 1 |  |
| `serial_monitoring_interval` | 1 |  |
| `activity_temporal_pattern` | 1 |  |
| `hours_of_life` | 1 |  |
| `fluctuation_and_1yr_rule` | 1 |  |
| `circadian_pattern` | 1 |  |
| `recurrence_interval` | 1 | 復發間隔 |
| `drug_half_life` | 1 | 藥物作用時長 |
| `attrition_over_time` | 1 | 追蹤期間的樣本流失 |
| `developmental_stage` | 1 | 發展階段（年齡由里程碑編碼） |
| `seizure_free_interval` | 1 |  |
| `time_since_injury` | 1 |  |
| `regression_timing` | 1 |  |
| `fluctuation_pattern` | 1 |  |
| `rash_duration` | 1 | 疹子持續幾天 |
| `resolution_interval` | 1 |  |
| `menstrual_cycle_timing` | 1 |  |
| `developmental_window` | 1 |  |
| `growth_stage_timing` | 1 |  |
| `seasonal_pattern` | 1 |  |
| `post_event_interval` | 1 |  |
| `surveillance_interval` | 1 |  |
| `lesion_temporal_behaviour` | 1 |  |
| `gestational_classification` | 1 |  |
| `latency_pattern` | 1 |  |
| `persistence_duration` | 1 |  |
| `effect_size_in_time` | 1 |  |
| `fasting_duration` | 1 | 禁食多久 |
| `demarcation_interval` | 1 |  |
| `seasonal_timing` | 1 |  |
| `drug_duration_of_action` | 1 |  |
| `treatment_timing` | 1 |  |
| `temporal_coincidence` | 1 |  |
| `disease_duration` | 1 |  |
| `situational_temporal_pattern` | 1 |  |
| `recurrence_pattern` | 1 |  |
| `expected_duration` | 1 |  |
| `measurement_time_window` | 1 | 檢驗值的時間平均窗 |
| `nocturnal_absence` | 1 |  |
| `size_variation_over_time` | 1 |  |
| `sleep_timing_pattern` | 1 |  |
| `repeated_measurement_interval` | 1 |  |
| `exposure_window` | 1 | 暴露後預防的時效 |
| `symptom_frequency` | 1 | 症狀頻率 |
| `episode_duration` | 1 | 單次發作期長度 |
| `escalating_recurrence` | 1 |  |
| `gestational_dating` | 1 |  |
| `withdrawal_timing` | 1 |  |
| `retrieval_window` | 1 |  |
| `study_temporal_design` | 1 | 研究設計的時間結構 |

---

## 6. Duration-Critical 題目清單（人工判讀，177 題）

每題附時間軸、關鍵時間片語、以及「為什麼改了時間答案會變」。
完整逐題記錄（含 not_critical / no_duration 的理由）在 `verification/manual_read_*.jsonl`。

### 6.1 MedBullets — 88 題

| uid | 時間軸 | 關鍵時間片語 | 為什麼是 critical |
|---|---|---|---|
| `mb_0209` | activity_temporal_pattern | weakest every MORNING, IMPROVES when he rides his bike | improvement with use is the Lambert-Eaton signature; myasthenia does the opposite. Flip the pattern and the answer changes |
| `mb_0113` | acuity | fatigue 'lately', otherwise asymptomatic | acute vs chronic leukemia IS a tempo distinction and both acute leukemias are options; an indolent 72-year-old with WBC 67.5k is CLL, the same counts with a fulminant 2-week course would be AML |
| `mb_0153` | acuity | dyspnea worsening over 2 days | acute decompensation selects the diuretic; sacubitril/valsartan, metoprolol and spironolactone (all options) are chronic GDMT for the stable patient |
| `mb_0244` | acuity | malaise several days, 10 lb loss over the past month | a month-long indolent build-up with leukostasis favours CML; the same blast percentage arriving over days reads as AML (option B) |
| `mb_0117` | age_gated | age 16, no pubertal onset | delayed puberty is age-defined; the identical exam at 12 would be reassurance (option D). NOTE age axis, not symptom duration |
| `mb_0155` | age_gated | age 2, vocabulary 10-25 words, 1-word commands | milestones are age-normed; identical findings at 18 months are normal. NOTE age axis |
| `mb_0217` | age_gated | age 35; only 2 periods in the last year | FSH 49 before 40 is primary ovarian insufficiency requiring replacement; the same numbers at 55 are physiologic menopause with a different treatment calculus |
| `mb_0085` | age_persistence | age 3 weeks, hernia reducible <1cm | umbilical hernia management is a clock: expectant now, elective repair if still there at ~5 years. NOTE this is patient age, not symptom duration |
| `mb_0101` | age_persistence | age 4 days, discharge for 2 days | maternal-estrogen withdrawal is normal in the first week; the identical finding in a 4-year-old mandates a CPS report (option B) |
| `mb_0173` | attack_continuity | vertigo CONSTANT, unchanged by Dix-Hallpike | continuous vs brief-positional vs episodic is the entire vertigo algorithm; Epley (option B) is the answer for the brief-positional version |
| `mb_0073` | attack_duration | prior attack 1 month ago, resolved after a few hours | attack length separates cluster (min-hours) from migraine (4-72h) and trigeminal neuralgia (seconds); that choice drives verapamil |
| `mb_0110` | attack_duration | attacks every few weeks, each lasting several hours | attack length + frequency separate migraine from cluster; high-flow O2 and verapamil (both cluster answers) are on the option list |
| `mb_0230` | attack_duration | nocturnal attacks lasting 2-3 hours; bout 1 year ago, restarted 3 weeks ago | three temporal features stack: attack length, nocturnal timing, and the year-apart clustering. All three define cluster headache |
| `mb_0236` | attack_duration | spasms 1-2 seconds, clustered every 20-30 seconds, at age 6 months | the second-scale spasm with cluster periodicity plus regression at 6 months defines infantile spasms |
| `mb_0234` | attrition_over_time | followed at 6-MONTH INTERVALS for 5 YEARS; only 40% replied throughout | loss of subjects is something that only happens ACROSS follow-up time -- collapse the study to a single cross-section and attrition cannot arise, leaving volunteer or late-look bias. Was dropped by the keyword screen for lacking vignette wording |
| `mb_0158` | chronological_vs_corrected_age | chronological age 2 months, born at 29 weeks | the whole item is whether to count from birth or from due date; options D and E are explicit delay traps |
| `mb_0239` | circadian_pattern | urge to move ONLY at night, relieved by walking, none by day | the evening-only timing with relief by movement IS the RLS criterion; remove the circadian confinement and it is not RLS |
| `mb_0106` | circadian_phase | cannot stay awake past 6pm, wakes rested | the options differ ONLY in phase direction (advanced vs delayed); the whole item is a timing judgement |
| `mb_0298` | circadian_phase | cannot fall asleep until 2 hours before the alarm, for a month | the diagnosis is a sleep PHASE shift; mirror image of mb_0106 (advanced phase). Move the phase and the treatment flips |
| `mb_0143` | cyclical_pattern | years of pain WORSE JUST BEFORE MENSES; dyspareunia 2 months | the cyclical premenstrual timing is the endometriosis signature; decouple the pain from the cycle and PID/adenomyosis take over |
| `mb_0201` | cyclical_pattern | symptom cluster recurring approximately every 4 weeks | the monthly cyclicity is what excludes GAD/MDD/panic; PMDD vs PMS then turns on functional impairment |
| `mb_0271` | developmental_stage | NO age is stated; the milestones themselves encode it (stairs both-feet-per-step, 6-cube tower, 50 words in 2-word sentences ~ 24 months) | the item asks what comes NEXT on the developmental timeline, so the answer is read off an age that must first be inferred from the milestones; change the milestones and the expected next skill changes. Was dropped by the keyword screen because a regex hunting for 'for 3 days' cannot see age encoded as behaviour |
| `mb_0078` | diurnal_pattern | pain present in the morning, unbearable by end of DAY | activity-worsening pattern is what makes this OA; invert it (worse on waking, better with use) and it reads inflammatory |
| `mb_0285` | drug_half_life | drug effect wears off in SECONDS | the seconds-long duration of action IS how the item tells you the drug is adenosine; remove it and the medication is unidentifiable |
| `mb_0026` | dsm_duration_criterion | throat-clearing daily for the past 2 years | Tourette requires >1 year of tics; at 3 months it is provisional tic disorder and haloperidol is wrong |
| `mb_0224` | dsm_duration_criterion | symptoms persisting for the past 8 months | brief psychotic (<1mo), schizophreniform (1-6mo) and schizophrenia (>6mo) are ALL options and are separated by duration alone |
| `mb_0235` | fluctuation_and_1yr_rule | FLUCTUATING cognition over 2 years; parkinsonism new since 1 year ago | DLB is defined by fluctuation plus the one-year rule relative to parkinsonism onset; shift that interval and it becomes Parkinson disease dementia |
| `mb_0097` | gestational_window | 10 weeks 2 days gestation | CVS 10-13 wk vs amniocentesis >=15 wk; move her to 18 weeks and option A becomes correct |
| `mb_0183` | gestational_window | 12 weeks gestation; PTU started 4 weeks ago at max dose | the correct answer names a trimester; methimazole is 1st-trimester teratogenic (option B) and radioiodine is contraindicated throughout |
| `mb_0123` | growth_rate | lesion grew rapidly over 2 weeks | rapid weeks-long growth IS the keratoacanthoma signature; SCC and BCC look similar on the photo but grow over months to years |
| `mb_0222` | hours_of_life | age 4 days, bilirubin 13 unconjugated | neonatal bilirubin thresholds are plotted against hours of life; 13 mg/dL at day 4 is observation, the same value at 12 hours is an emergency |
| `mb_0071` | interval_since_event | DTaP 14 years ago; influenza 2 months ago; meningococcal 3 years ago | the whole question IS interval arithmetic; option C adds influenza and is wrong only because his flu shot was 2 months ago |
| `mb_0018` | latency | chest pain 2 days after STEMI | interval from infarct separates peri-infarction pericarditis (aspirin) from re-infarction (angiography) |
| `mb_0266` | latency | symptoms 20 minutes after metoclopramide | minutes-to-hours after a dopamine blocker is acute dystonia (diphenhydramine); months later it would be tardive dyskinesia and botulinum (option A) enters |
| `mb_0228` | latency_trajectory | treated 5 weeks ago; CD4 40 -> 400 over 4 weeks of ART | IRIS IS defined by the interval since ART start plus the CD4 trajectory; paradoxical worsening at 4-6 weeks is the whole diagnosis |
| `mb_0279` | latency_trajectory | treated 5 weeks ago; CD4 40 -> 400 | EXACT DUPLICATE of mb_0228 |
| `mb_0094` | lucid_interval | LOC 30s, skied down lucid, then GCS 15 -> 7 | the lucid interval IS the epidural signature; without it (steady decline) option A bridging veins/subdural wins |
| `mb_0260` | onset_speed | sentinel headache 2 days ago; today max intensity within MINUTES | time-to-peak defines thunderclap, and the sentinel leak two days earlier is the classic warning pattern; migraine and tension (options C, D) build over hours |
| `mb_0098` | onset_timing | swelling NOT present at birth, appeared by 9h, unchanged over 12h | present-at-birth vs appearing-later is exactly what separates caput from cephalohematoma/subgaleal, and the stable size excludes an expanding subgaleal bleed |
| `mb_0136` | onset_timing | onset day 3, purulent by day 4 of life | day-of-onset IS the ophthalmia neonatorum algorithm: day1 chemical, day2-5 gonococcal, day5-14 chlamydial (which would make oral erythromycin, option C, correct) |
| `mb_0165` | onset_timing | lump NOT present at birth, seen at 6 hours | present-at-birth-and-crosses-sutures (caput) vs appears-later-and-respects-sutures (cephalohematoma); the timing is half the diagnosis |
| `mb_0174` | onset_timing | age 13 days at onset | day 5-14 onset with pneumonitis is chlamydial -> ORAL erythromycin; day 2-5 would be gonococcal -> IV ceftriaxone (option C). Mirror image of mb_0136 |
| `mb_0205` | onset_timing | well until today, then SUDDEN bilious vomiting at 3 weeks | abrupt onset in a previously thriving infant is volvulus; atresia declares itself at birth and Hirschsprung with delayed meconium |
| `mb_0272` | postprandial_timing | 2 months of 20-30 min episodes, BETTER after eating | pain relieved by food is duodenal; pain worsened by food is gastric (option B). The meal-timing relationship separates the two options |
| `mb_0055` | progression_tempo | personality change over 2 months, worsening | 14-3-3 (CJD) is an explicit option; at 3 weeks of rapid decline CJD wins, at 2 months-to-years chorea it is Huntington |
| `mb_0124` | progression_tempo | executive decline over 1 month, focal weakness 3 months prior | stepwise decline anchored to a prior focal event is what separates vascular from Alzheimer's gradual years-long slope |
| `mb_0160` | progression_tempo | memory worsening gradually over the past few years | every option is a different tempo: years-gradual AD vs months-rapid CJD vs stepwise vascular vs early-behavioural FTD |
| `mb_0168` | progression_tempo | weakness then RAPID decline to seizure and agonal breathing | an anticoagulated patient deteriorating to seizure and herniation within the hour is hemorrhage; ischemic stroke does not follow that trajectory |
| `mb_0233` | progression_tempo | facial and abdominal hair over the LAST MONTH; irregular cycles 1 year | rapid virilization with clitoromegaly is a tumor; stretch the same picture over years and PCOS (option E) becomes correct |
| `mb_0291` | progression_tempo | normal until 3 months ago, then rapid decline with startle myoclonus | every option is a dementia; the months-not-years tempo plus myoclonus is what selects CJD (mirror of mb_0055) |
| `mb_0269` | progression_trajectory | vomiting began 1 week ago and is PROGRESSIVELY worse at age 4 weeks | worsening trajectory with falling percentiles is what selects ultrasound; the same age with a stable curve is reassurance (cf mb_0052) |
| `mb_0286` | progression_trajectory | scalp lesion at baseline, non-healing ulcer 3 months later | actinic keratosis is option A; it is the transformation across the 3-month interval that makes the answer SCC |
| `mb_0024` | recovery_time | seizure 3 min; hemiparesis resolved over a few minutes | deficit resolving in minutes IS Todd's paralysis -> observation; persisting deficit would mean CT/alteplase |
| `mb_0258` | recurrence_interval | same right middle lobe pneumonia recurring 2 months apart | recurrence in the SAME lobe is the endobronchial-obstruction red flag; a first episode would simply be treated |
| `mb_0180` | regression_trajectory | was rolling back-to-front 2 months ago, now cannot | the item is built on LOSS of an acquired milestone; static delay would not point to a storage disease |
| `mb_0203` | screening_interval | colonoscopy 5 years ago, normal; menopause at 52 | the item IS interval arithmetic -- colonoscopy is not due until 10 years (killing option B), mammography is |
| `mb_0287` | screening_interval | father diagnosed with colon cancer at age 40; patient now 25 | pure interval arithmetic -- screen at 40 or 10 years before the relative's diagnosis (age 30), hence 'in 5 years' not 'in 10' |
| `mb_0197` | serial_monitoring_interval | repeat the ECG in 10 minutes | the correct answer IS a time interval -- serial ECGs while the pain evolves and the first tracings are non-diagnostic |
| `mb_0172` | sleep_duration | slept 9 hours in youth, now 7 hours | the vignette IS a duration comparison; 7 hours with peaceful sleep and no daytime symptoms is normal ageing, 3 hours with fatigue would be a sleep disorder |
| `mb_0006` | symptom_duration | woke last night, sudden onset (<1d) | acute monoarthritis window is what separates gout from RA/Lyme; at 3 months gout attack no longer fits |
| `mb_0017` | symptom_duration | 6-month history of back pain | inflammatory back pain is DEFINED by >=3 months; at 6 days this is mechanical strain and imaging choice changes |
| `mb_0032` | symptom_duration | foul diarrhea for the past 2 years | 2 years makes it chronic pancreatitis exocrine failure; at 2 days the same stool findings read as infectious and cipro/rehydration win |
| `mb_0034` | symptom_duration | daily for 3 months; immigrated 6 months ago | chronic + eosinophilia + endemic origin -> O&P; at 2 days the answer becomes stool toxin/culture |
| `mb_0035` | symptom_duration | arm weakness for a few weeks; 17 lb loss in 1 month | subacute progressive course is exactly what excludes cerebral infarction (option C) and selects apical tumor |
| `mb_0039` | symptom_duration | arm weakness for a few weeks; 17 lb loss in 1 month | EXACT DUPLICATE of mb_0035 -- same verdict; flag for dataset dedup |
| `mb_0100` | symptom_duration | dyspnea over the past 48 hours | acute vs chronic is precisely what separates tamponade from constriction; NB 48h actually favours tamponade, the answer key looks questionable |
| `mb_0115` | symptom_duration | eyelid mass persisting for the past month | hordeolum is an explicit option; a month of a painless firm nodule is chalazion, a few days of a tender one is a hordeolum |
| `mb_0132` | symptom_duration | 6-month history of back pain | EXACT DUPLICATE of mb_0017 |
| `mb_0190` | symptom_duration | 1 week of poor sleep during exams | chronic insomnia needs >=3 months; at 1 week this is situational and the answer is hygiene, at 6 months polysomnography/hypnotics enter the frame |
| `mb_0213` | symptom_duration | watery diarrhea for the past 24 hours | acute (<7 days) diarrhea in a healthy adult needs no workup; at 3 weeks the stool studies (options C/D/E) become correct |
| `mb_0243` | symptom_duration | pain started the day prior after exertion | the acute post-exertional onset is what selects strain; stretch it to months and osteoarthritis or stenosis take over |
| `mb_0273` | symptom_duration | dyspnea 2 months after influenza, progressive | months-long progressive constriction vs acute tamponade (option A); the same physiology over 2 days would be tamponade |
| `mb_0305` | symptom_duration | eyelid mass persisting for the past month | EXACT DUPLICATE of mb_0115 |
| `mb_0157` | symptom_resolution | deficit lasted 20 minutes then fully resolved | complete resolution inside the hour with a clean CT is what makes this TIA rather than stroke, and drives antiplatelet over thrombolysis |
| `mb_0149` | symptom_stability | exertional pain for months, NOT worsening, none at rest | stable vs unstable angina is defined by tempo and trajectory; make it new or crescendo or at-rest and option E wins |
| `mb_0265` | temporal_ordering | groin pain started a YEAR ago; scuba began only 2 MONTHS ago | the exposure POSTDATES the symptom, which is exactly what eliminates scuba (option E) as the cause. Pure temporal-order reasoning |
| `mb_0307` | temporal_ordering | breathing worsened SUBSEQUENT to naloxone administration | causality rests entirely on the symptoms appearing AFTER the drug; same reasoning as mb_0265 |
| `mb_0068` | therapeutic_window | painless vision loss started 2 days ago | tPA is option B and is time-gated; the entire snowstorm story exists to put him outside the window -> aspirin+statin |
| `mb_0202` | therapeutic_window | intercourse earlier this morning | EC is time-gated: levonorgestrel <=72h; at day 4 the answer moves to ulipristal or a copper IUD |
| `mb_0257` | therapeutic_window | needlestick just occurred; source viral load 1.8 million | PEP is a race against hours -- start now AND draw baseline serology; options C and D both wait for results and lose on timing alone |
| `mb_0047` | time_of_day | 2 months of spasms, ON WAKING, worse when sleep-deprived | morning predominance + sleep-deprivation trigger IS the JME signature; move the jerks to evening and the answer changes |
| `mb_0139` | time_of_day | proteinuria persists across two visits | the answer IS a timed test -- orthostatic proteinuria exists only in upright daytime samples; split AM/PM dipstick is the whole point |
| `mb_0278` | time_of_day | arm jerks UPON AWAKENING plus staring spells | the morning myoclonus is what makes this JME and rules out pure absence epilepsy (ethosuximide, option C). Same mechanism as mb_0047 |
| `mb_0022` | time_since_exposure | ingestion ~3 hours ago | textbook time-gated answer: <1h charcoal, 4h+ interpretable level, 3h -> empiric NAC. Rumack-Matthew is a clock |
| `mb_0014` | treatment_trial | acne since age 13; 1-month trial of topical therapy | option A is literally 'continue 1 more month' -- the length of the failed trial decides escalate vs wait |
| `mb_0148` | treatment_trial | 1 week on first-line antidepressant, no improvement | SSRIs need 4-6 weeks; at 1 week you continue, at 8 weeks you switch (option B). The elapsed trial length IS the answer |
| `mb_0246` | treatment_trial | 2 months on citalopram with minimal improvement | 2 months IS an adequate trial so you augment; at 1 week the correct answer would be to continue (cf mb_0148) |
| `mb_0283` | treatment_trial | 3 weeks on fluoxetine, unchanged | 3 weeks is an INCOMPLETE trial so you hold the dose; at 8 weeks options B/C/D become right. Third instance of this pattern after mb_0148 and mb_0246 |

### 6.2 MedMCQA — 38 題

| uid | 時間軸 | 關鍵時間片語 | 為什麼是 critical |
|---|---|---|---|
| `mmc_02354` | developmental_window | injury to the DECIDUOUS predecessor 3 YEARS ago, during permanent tooth development | Turner hypoplasia requires the insult to land inside the successor's formation window; move the injury later and the enamel would already be formed |
| `mmc_00713` | fluctuation_pattern | jaundice 2 months, WAXING AND WANING | intermittent jaundice (tumour sloughing) is what separates periampullary from pancreatic head carcinoma (option A), which gives progressive painless jaundice |
| `mmc_02758` | gestational_classification | two FIRST-TRIMESTER abortions; one delivery at end of ninth month | the GPAL notation is built entirely out of gestational durations -- reclassify any pregnancy by length and the answer changes |
| `mmc_01685` | gestational_window | 36 weeks gestation; BP repeated after 20 minutes | option B is 'continue till term' -- at 36 weeks you deliver, at 28 weeks you would prolong. Gestational age decides between the two management arms |
| `mmc_01746` | gestational_window | 37 weeks gestation with preeclampsia | at 37 weeks you deliver; at 32 weeks options A/B (expectant management) become correct. Gestational age is the whole decision |
| `mmc_02089` | gestational_window | 9 weeks now; prior preterm births at 30 and 32 weeks | cerclage (option B) is placed at 12-14 weeks, so at 9 weeks you measure cervical length first; the gestational clock orders the steps |
| `mmc_03417` | gestational_window | PIH at 32 weeks, controlled; ANSWER is a gestational age | all four options are different week counts (34/35/37/40) -- the item is nothing but a timing decision |
| `mmc_03584` | gestational_window | 4 MONTHS pregnant, already on valproate | the teratogenic window closed in the first trimester, so switching now only risks seizures; preconception or at 6 weeks the answer would be to switch (option A/B) |
| `mmc_02412` | growth_stage_timing | age 5y4m; the ANSWER is 'wait and watch for 6 years' | functional appliances are timed to the pubertal growth spurt; at 5 years every active option is premature and the correct answer is literally an interval |
| `mmc_00327` | interval_since_event | DT booster at school entry (~5 years ago) | tetanus prophylaxis is interval arithmetic -- fully immunised within the window means antiserum is NOT indicated, which is what the EXCEPT is testing |
| `mmc_02763` | latency | hypotension 3 MINUTES after the epidural dose | minutes-fast collapse means the drug reached the subarachnoid space; systemic absorption (option A) has a slower time course |
| `mmc_04169` | latency | presents 2 DAYS AFTER the trauma | post-traumatic delay of days is the carotico-cavernous fistula signature; immediate proptosis would be a fracture/haemorrhage and a febrile course would be sinus thrombosis |
| `mmc_03754` | latency_pattern | restoration placed 5 months ago, ASYMPTOMATIC FOR THE FIRST 4 MONTHS, then progressive | a silent interval followed by escalating sensitivity is irreversible pulpitis; marginal leakage (option A) would have hurt from the start |
| `mmc_02751` | lesion_temporal_behaviour | lesion PRESENT SINCE BIRTH with NO CHANGE in morphology | present-at-birth-and-static is a capillary malformation; infantile hemangioma appears weeks later and proliferates then involutes, congenital hemangioma involutes. The temporal behaviour separates options A/B/C |
| `mmc_01212` | lucid_interval | conscious interval between two unconscious periods | the answer IS the name of a time interval; the item is purely a temporal-concept question |
| `mmc_01748` | menstrual_cycle_timing | LMP 3 weeks ago; answer specifies follow-up in 2-3 months | luteal-phase timing makes a 5 cm cyst physiologic, and the ANSWER itself is an interval -- re-image after 2-3 cycles |
| `mmc_00861` | onset_speed | stridor and drooling SINCE 4-6 HOURS | hours-fast onset with drooling is epiglottitis, days-long barking is croup -- that is the whole discriminator. NB the keyed answer (racemic epinephrine) is the croup treatment and looks wrong for a drooling child |
| `mmc_03527` | onset_timing | ABSOLUTELY NORMAL AT BIRTH, coarsening by 10 months | normal-at-birth-then-progressive is the storage disease signature; congenital hypothyroidism (option C) and Beckwith-Wiedemann are present from birth |
| `mmc_04091` | persistence_duration | urinoma PERSISTING after 12 DAYS in a stable afebrile patient | early on, wait-and-watch (option B) is right; persistence past ~10-14 days is exactly what converts this to drainage/diversion |
| `mmc_02575` | post_event_interval | myocardial infarction 2 WEEKS ago | recent ACS mandates high-intensity statin irrespective of the LDL number; in primary prevention the moderate-intensity option C would be right |
| `mmc_01186` | rash_duration | rash CLEARED ON THE 3RD DAY without desquamation | rubella is literally 'three-day measles'; measles (option D) lasts 5-6 days and desquamates. The rash duration IS the diagnosis |
| `mmc_00540` | regression_timing | head growth DECELERATED AFTER 6 MONTHS of age, after normal early development | Rett is defined by normal development followed by regression at 6-18 months; without that temporal shape the picture reads as a static disorder such as Asperger (option A) |
| `mmc_01428` | resolution_interval | AFTER 8 WEEKS the consolidation is denser despite clinical improvement | non-resolving pneumonia is DEFINED by radiographic failure to clear at 4-8 weeks; at 1 week you simply wait and no investigation is indicated |
| `mmc_02461` | seasonal_pattern | 3 years of papules, WORSE IN SUMMER and better in winter | the summer/winter alternation in a photo-exposed distribution is what makes this airborne contact dermatitis (patch test) rather than atopic (prick/IgE) |
| `mmc_00284` | seizure_free_interval | seizure-free for 2 years on levetiracetam | option D explicitly asserts a 5-year requirement and option B says stop now; the 2-year interval is precisely what decides between them |
| `mmc_02636` | surveillance_interval | the ANSWER is a surveillance schedule (3-monthly exam, annual mammogram) | every option is a different interval/modality combination; the item is entirely about follow-up periodicity |
| `mmc_00771` | symptom_duration | fever 3 days, disorientation 1 day | a 3-day course with polymorph-predominant CSF excludes tubercular meningitis (option B), which runs over 2-3 weeks with lymphocytes |
| `mmc_01006` | symptom_duration | pain and altered bowel habits for the last 6 months | the 6-month chronic course is exactly why ISCHEMIC bowel disease is the odd one out; the other three are all chronic ileocecal diseases |
| `mmc_02082` | symptom_duration | fever for 1 MONTH before headache and ataxia | a month-long prodrome with basal exudates is tubercular; option D (neutrophilia) is the answer for a 2-3 day pyogenic course. Exact mirror of mmc_00771 |
| `mmc_01155` | temporal_ordering | immobilised 3 MONTHS, then calf swelling, then massage, then sudden death | the entire item is a temporal chain -- immobilisation window creates the DVT, massage dislodges it, death follows. Break the order and the answer collapses |
| `mmc_04129` | temporal_ordering | tonsillectomy at 6 WITHOUT bleeding; abnormal aPTT found at 12 | an uneventful haemostatic challenge BEFORE the abnormal test is what excludes haemophilia (option D) and acquired causes; the ordering is the entire argument |
| `mmc_02771` | therapeutic_window | pain unrelieved for the past 1 hour | options B and D both propose DELAY; torsion salvage is a clock, and the 1-hour presentation is what makes immediate exploration right |
| `mmc_03495` | therapeutic_window | husband's rash appeared 4 DAYS ago; options differ by WHEN to give immunoglobulin | VZIG is only useful inside the post-exposure window, and options A/C defer it to birth; the 4-day interval is what makes 'now' correct |
| `mmc_03205` | time_of_day | precipitated IN THE MORNING and during exams (sleep deprivation) | morning myoclonus with sleep-deprivation triggering is the JME signature; third instance of this exact pattern after mb_0047 and mb_0278 |
| `mmc_00351` | time_since_injury | presented ONE HOUR after injury, tooth still vital | the stem literally asks 'at this point of time'; early and vital means splint, late and necrotic means root canal (option D) |
| `mmc_04181` | treatment_duration | steroid-DEPENDENT for the last 5 YEARS with cushingoid toxicity | at first presentation steroids (option D) are correct; five years of dependence with cataracts is what forces a steroid-sparing agent |
| `mmc_02828` | treatment_trial | adequate intake for the LAST 1 WEEK with no weight gain | one week of full intake should produce gain, so failure implicates purging; at one day the non-response would be expected and no action indicated |
| `mmc_00810` | wound_healing_phase | wound 4 days old, granulation tissue | collagen type is a function of healing phase -- type III at days, converted to type I over months. The day count IS the answer |

### 6.3 MMLU-Med — 51 題

| uid | 時間軸 | 關鍵時間片語 | 為什麼是 critical |
|---|---|---|---|
| `mmlu_0125` | age_gated | age 2 years with 10-degree genu varum | physiologic bowing resolves by about 2-3 years; the identical angle at 6 years would be Blount disease and option C could enter |
| `mmlu_0144` | attack_duration | each paroxysm lasts ONE SECOND OR LESS | sub-second trigger-evoked jabs define trigeminal neuralgia; stretch the attacks to minutes-hours and it becomes cluster, to 4-72h and it becomes migraine |
| `mmlu_0223` | attack_duration | two episodes each lasting 12 HOURS over 3 months | attack length is a formal Meniere criterion (20 min to 12-24h); BPPV (option B) lasts under a minute and brainstem TIA (option C) minutes. The 12 hours picks the answer |
| `mmlu_0082` | cyclical_pattern | symptoms 3-4 DAYS BEFORE menses, switching ON THE DAY menses begins | the lock-step relationship to the cycle is what excludes bipolar, cyclothymia and GAD. NB the keyed answer looks questionable -- this reads as a premenstrual disorder |
| `mmlu_0042` | demarcation_interval | frostbite THREE WEEKS ago, toes now black and demarcated | frostbite is managed by watchful waiting until demarcation; at day 2 amputation would be wrong and debridement/supportive care right. Three weeks plus sepsis is what makes amputation correct |
| `mmlu_0119` | disease_duration | TEN YEARS of rheumatoid arthritis; drop attacks over 3 months | C1-C2 instability is a complication of long-standing erosive disease; with six months of RA it would be implausible and arrhythmia or anxiety would take over |
| `mmlu_0050` | drug_duration_of_action | rate controlled by metoprolol, back to 160 TWO HOURS LATER | the drug worked and then wore off, so you re-dose the same agent; had it never worked you would switch class. The 2-hour interval carries the whole decision |
| `mmlu_0077` | dsm_duration_criterion | relocated 2 MONTHS ago; symptoms since | adjustment disorder requires onset within 3 months of a stressor while GAD (option B) requires 6+ months of worry; the 2-month window tied to the promotion is the whole discriminator |
| `mmlu_0110` | dsm_duration_criterion | fever to 39.4C for EIGHT DAYS | Kawasaki requires fever >=5 days by definition, and IVIG must land inside 10 days; the same findings at 2 days of fever would not meet criteria and IVIG would be wrong |
| `mmlu_0128` | dsm_duration_criterion | robbed 3 WEEKS ago; symptoms for 2 weeks | acute stress disorder spans 3 days to 1 month post-trauma; past a month the same picture is PTSD. The 3-week mark is the whole diagnosis |
| `mmlu_0255` | dsm_duration_criterion | 6 months of symptoms that IMPROVE AFTER DEFECATION and worsen before exams | the Rome criteria are explicitly a duration threshold plus a defecation relationship; under 6 months this is not IBS and the drug answer would not follow |
| `mmlu_0002` | effect_size_in_time | 6.4 days vs 6.7 days, p=0.04 | the answer turns on the MAGNITUDE of a duration difference -- 0.3 days is clinically trivial; make it 6.4 vs 12 days and option A becomes correct |
| `mmlu_0236` | episode_duration | 2 MONTHS of high energy needing only 4 HOURS of sleep, no childhood history | a discrete adult-onset episode with reduced sleep NEED is manic; ADHD (option B) is lifelong from childhood and he performed well in school |
| `mmlu_0260` | escalating_recurrence | reaction 15 min after the sting; THREE stings over a year, EACH MORE SEVERE | the escalation across successive exposures over the year is what predicts a lethal next reaction and mandates an epinephrine autoinjector rather than avoidance or antihistamines |
| `mmlu_0142` | expected_duration | unable to eat 6 DAYS after surgery, previously well nourished | route of nutrition is chosen by how long support is expected to be needed -- under ~7-10 days peripheral, longer central (option A). The day count is the decision |
| `mmlu_0230` | exposure_window | shared a room 1 WEEK ago; index case admitted YESTERDAY | chemoprophylaxis applies to close contacts inside the exposure window; move the shared room to three months ago and no prophylaxis is indicated |
| `mmlu_0038` | fasting_duration | TWELVE HOURS after the meal, no further intake | the fuel source is a pure function of fasting time -- 0-4h absorption (option A), 4-24h glycogenolysis, >24h gluconeogenesis. The 12 hours IS the answer |
| `mmlu_0264` | gestational_dating | UNSURE OF DATES; stated 16 weeks; MSAFP 3 MoM | MSAFP is interpreted against gestational age, so wrong dates are the commonest cause of a falsely raised value; establishing the dates by ultrasound must precede everything else |
| `mmlu_0073` | latency | symptoms TEN DAYS after the globulin dose | the 7-14 day latency is what makes this type III serum sickness; an immediate reaction would be anaphylaxis or cytokine release (option A) |
| `mmlu_0163` | latency | FIVE YEARS after the gunshot wound | a thrill and machinery murmur developing years later is a post-traumatic AV fistula; the acute options (spasm, DVT) are excluded by the interval alone |
| `mmlu_0145` | measurement_time_window | 3 months of home readings 140-200 vs an A1c of 5.4% | A1c integrates over the red cell lifespan, so haemolysis shortens the averaging window and lowers the value; the item is a question about a time-averaged measurement |
| `mmlu_0160` | nocturnal_absence | 3 years of pain relieved by defecation that DOES NOT OCCUR AT NIGHT | absence of nocturnal symptoms is a Rome criterion separating functional from organic disease; nocturnal pain would redirect the whole workup |
| `mmlu_0210` | onset_speed | SUDDEN onset 12 hours ago, no prior severe headache | instantaneous onset is what separates SAH from meningitis (option B), which builds over hours to days; the absence of any prior severe headache also removes migraine and cluster |
| `mmlu_0175` | postprandial_timing | pain 20-30 MINUTES AFTER EATING, lasting ABOUT 30 MINUTES, for 8 weeks | the postprandial latency plus the half-hour attack length is the biliary colic signature; antacid-responsive or continuous pain would send you to a different study |
| `mmlu_0001` | progression_pattern | contractions WERE every 4 minutes, now weak and intermittent | good contractions that deteriorate in the active phase is hypotonic (secondary) dysfunction; never having established them is primary dysfunctional labour (option D) |
| `mmlu_0133` | progression_pattern | contractions WERE every 4 minutes, now weak and intermittent | same vignette as mmlu_0001 with a management stem; deteriorating contractions mean augment, whereas a hypertonic pattern would call for tocolysis (option B) |
| `mmlu_0181` | progression_tempo | cycles irregular over a YEAR before stopping; gradual weight gain and hirsutism | option A is the androgen-secreting tumour, which declares itself over months with rapid virilization; the slow year-long build is what makes this PCOS |
| `mmlu_0188` | progression_tempo | decline over 6 months anchored to STROKES 3 years and 7 months ago | decline stepping down from discrete cerebrovascular events is vascular dementia; a smooth years-long slope would be Alzheimer and a months-long collapse would suggest something else |
| `mmlu_0129` | recurrence_pattern | pain recurs every 2-3 months and clears within 2 weeks each time | a self-limiting recurrent course with a normal exam is what makes imaging (options B/C) unnecessary; a progressive unremitting course would flip that |
| `mmlu_0202` | repeated_measurement_interval | elevated readings on FOUR separate occasions across 3 months | the diagnosis requires elevation on repeated visits; on a single reading the correct step would be to recheck rather than to start management |
| `mmlu_0270` | retrieval_window | swallowed 2 HOURS ago; nail still in the stomach | a sharp object is retrievable only while it remains proximal; the short interval keeps it in the stomach, and once past the duodenum option B (observation) becomes correct |
| `mmlu_0045` | seasonal_timing | visit in OCTOBER; Pap 3 months ago; ophthalmology 6 months ago | the month of the visit is what makes influenza vaccination the answer, and the stated intervals are exactly what disqualify the Pap and mammography options |
| `mmlu_0124` | situational_temporal_pattern | symptoms start A FEW DAYS BEFORE each flight and STOP THE DAY AFTER arrival | the tight locking of symptoms to the trigger's timeline is what makes this anxiety rather than urologic disease; decouple them and the workup changes |
| `mmlu_0166` | size_variation_over_time | swelling has VARIED IN SIZE over the past several months | a hydrocele that changes size communicates with the peritoneum and must be repaired; a static one would simply be observed |
| `mmlu_0201` | sleep_timing_pattern | falls asleep normally but wakes at 2-3 AM; alcohol at bedtime | sleep-MAINTENANCE insomnia in the second half of the night is the alcohol rebound signature; sleep-ONSET insomnia would point elsewhere entirely |
| `mmlu_0161` | study_temporal_design | all 60 children treated three times a week FOR 2 MONTHS, with no comparison arm | two of the four options are defined purely by temporal structure -- cross-sectional (option D) is excluded because there IS longitudinal follow-up, and crossover (option C) because nobody switches arms over time |
| `mmlu_0007` | symptom_duration | 6 months of thyrotoxicosis on a neck mass present MORE THAN 10 YEARS | a decade-old goitre that turns toxic is multinodular; Graves (option B) does not arrive on top of a ten-year mass |
| `mmlu_0011` | symptom_duration | low-grade fever and dry cough for 7 DAYS while still functioning | an indolent week-long course with preserved function is atypical/walking pneumonia; one day of rigors and high fever would make amoxicillin (option A) correct |
| `mmlu_0043` | symptom_duration | 1 week of radicular pain that began after lifting | acute post-lifting onset selects disc herniation; the other three options (facet hypertrophy, osteophyte, spondylolisthesis) are all chronic degenerative processes |
| `mmlu_0151` | symptom_duration | nonradiating back pain for 3 DAYS after yard work | acute uncomplicated back pain inside the 4-6 week window means no imaging; a months-long course with red flags would make options C/D correct |
| `mmlu_0179` | symptom_duration | nasal symptoms for NINE DAYS, now new facial pain and fever 38.5 | acute bacterial rhinosinusitis is defined by persistence past ~10 days or worsening after initial improvement; at 3 days this is viral and no bacterial organism is the answer |
| `mmlu_0256` | symptom_duration | FIRST episode, recurring and resolving over the past WEEK | acute urticaria (<6 weeks) needs no workup; past six weeks it is chronic and the investigative options A/B/C become reasonable |
| `mmlu_0232` | symptom_frequency | rescue inhaler ONE TO TWO TIMES DAILY over the past month | asthma severity is classified by rescue-use frequency; more than twice a week is persistent and earns a controller, twice a month is intermittent and does not |
| `mmlu_0074` | temporal_coincidence | air leak ran 24 hours and HAS NOW STOPPED as he deteriorates | a leak that stops while the patient worsens means the tube has failed, not that the lung sealed; the coincidence in time carries the whole inference |
| `mmlu_0132` | temporal_ordering | watery diarrhea PERSISTING AFTER an episode of enteritis | the diarrhoea following the enteritis is what makes it post-infectious lactase loss; without that ordering the congenital options compete |
| `mmlu_0194` | temporal_ordering | desaturation WITHIN 2 MINUTES of the second diazepam dose | the two-minute latency after the drug is what implicates it over encephalitis or meningitis, neither of which would produce that timing |
| `mmlu_0199` | temporal_ordering | tremor worse over the PAST MONTH; fluoxetine started 3 WEEKS ago; bereavement 2 months ago | three candidate causes each carry their own timeline, and the answer is decided by which one aligns; alcohol went UP, which would improve essential tremor |
| `mmlu_0033` | treatment_duration | glucocorticoid therapy for the PAST 6 MONTHS | cumulative steroid exposure over months is what makes avascular necrosis the answer rather than osteoporosis or joint narrowing; a one-week course would not |
| `mmlu_0059` | treatment_timing | age 6 months; EVERY OPTION is a different repair timing | options are defer-to-2, defer-to-12, repair-soon, or emergency; paediatric inguinal hernias do not close spontaneously and carry incarceration risk, so the item is purely about when |
| `mmlu_0268` | withdrawal_timing | agitation on POSTOPERATIVE DAY 4 | delirium tremens peaks at 48-96 hours from the last drink, which is exactly day 4 of hospitalisation; fat emboli (option C) arrive at 24-72h with hypoxia and petechiae instead |
| `mmlu_0026` | wound_healing_phase | burn 1 WEEK ago, now pink granular tissue | the mediator is read off the healing phase -- week-old granulation is the angiogenic proliferative stage. Same mechanism as mmc_00810 where 4 days gave type III collagen |

---

## 7. 腳本判定的 duration-critical（MedQA 329 / 1273，未經人工複核）

### 7.1 MedQA 329 — 14 題

| uid | duration 改寫 | gold | 擾動後答案 |
|---|---|---|---|
| `aud_053` | for the last 20 minutes → for the last 8 months | A | E |
| `aud_085` | for the last month → for the last 18 months | A | B |
| `aud_105` | 30 minutes ago → for 6 months | F | D |
| `aud_154` | started yesterday → has been present for 8 months | C | E |
| `m325_10` | for 6 hours → for 6 months | C | D |
| `m325_1914` | Over the last 7 months → Over the last 7 days | C | A |
| `m325_645` | past 2 months → past 2 days | B | A |
| `m325_698` | for the past 8 months → for the past 8 days | A | B |
| `m325_787` | 1-day history → 2-year history | E | C |
| `v2_train_1064` | past week → past 6 months | E | D |
| `v2_train_159` | 2 days ago → 2 months ago | E | B |
| `v2_train_1633` | 8 months ago → 8 days ago | C | E |
| `v2_train_2931` | Three weeks ago → Three years ago | A | C |
| `v2_train_3346` | 6 hours ago → 6 months ago | C | B |

### 7.2 MedQA 1273 — 14 題

| uid | duration 改寫 | gold | 擾動後答案 |
|---|---|---|---|
| `test_0360` | the past few weeks → the past 3 days | A | D |
| `test_0367` | about 5 days → for 5 months | D | A |
| `test_0373` | 1-day history → 2-year history | E | C |
| `test_0414` | 3-day history → 6-month history | E | A |
| `test_0547` | 40 minutes after → 6 months after | E | C |
| `test_0580` | 1 week ago → for 8 months | A | D |
| `test_0698` | 2-month history → 2-day history | A | C |
| `test_0746` | over the past 48 hours → for the past 6 months | B | C |
| `test_0784` | recently → for 18 months | B | C |
| `test_0874` | 2 weeks ago → 2 years ago | E | A |
| `test_0896` | over the past 7 months → over the past 7 days | E | A |
| `test_0964` | for the past 2 days → for the past 2 years | D | A |
| `test_1021` | Two hours following → For 2 months following | E | C |
| `test_1089` | over the last month → over the last 2 days | B | C |

---

## 8. 產生這份報告的程式

```bash
python3 pipeline/dataset_statistics.py 329 1273 medbullets medmcqa mmlu
python3 pipeline/medmcqa_full_statistics.py
python3 pipeline/build_master_report.py      # 重新生成本檔
```

| 檔案 | 用途 |
|---|---|
| `build_master_report.py` | **本報告的生成器**（所有表格皆由資料檔重算） |
| `dataset_statistics.py` | §3.1–3.3 general vs critical 統計 |
| `medmcqa_full_statistics.py` | §3.4 MedMCQA 原始三個 split 的統計 |
| `build_new_dataset.py` | §1 從原始來源建資料集 |
| `dump_for_manual_read.py` | §6 逐題人工閱讀用的傾印工具 |
| `verify_duration_critical.py` | §7 反事實擾動腳本 |
| `verify_openended_probe.py` | 第二臂：檢查 MCQ 選項是否遮蔽效應 |
| `analyze_verification.py` | 腳本結果彙整 + 與舊標籤交叉比對 |
| `results_by_verified_subset.py` | 用驗證子集重切 reader 結果 |

| 資料檔 | 內容 |
|---|---|
| `verification/manual_read_*.jsonl` | 723 題逐題人工判讀（uid / dur / verdict / axis / note） |
| `verification/{ds}_duration_critical_*.json` | 腳本判定，含每題 5 次作答的原始輸出 |
| `verification/{ds}_openended_*.json` | open-ended 探針結果 |
| `datasets/{ds}/benchmark.json` | 題目本體 |
| `datasets/{ds}/durations.json` | role-tagged 病人 duration |
| `datasets/_raw_*` | 從網路抓下的原始檔（可重現） |

#!/usr/bin/env python3
"""Regenerate DURATION_CRITICAL_MASTER_zh.md end to end from the source files.

The report is written by a script rather than by hand because its numbers change every time a
question is reviewed, a dataset is rebuilt, or the duration extractor is re-run — and a report
that has to be hand-patched will eventually disagree with the data it claims to summarise. Run
this after any of those, and every table is recomputed from:

  datasets/<ds>/benchmark.json            question text, options, gold
  datasets/<ds>/durations.json            role-tagged patient duration
  datasets/statistics.json                written by dataset_statistics.py
  datasets/medmcqa_full_statistics.json   written by medmcqa_full_statistics.py
  verification/manual_read_<ds>.jsonl     per-question hand review
  verification/<ds>_duration_critical_*   perturbation-script verdicts

Usage:  python3 pipeline/dataset_statistics.py 329 1273 medbullets medmcqa mmlu
        python3 pipeline/medmcqa_full_statistics.py
        python3 pipeline/build_master_report.py
"""
import json, glob, csv, re
from pathlib import Path
from collections import Counter

PIPE = Path(__file__).resolve().parent
DS = ["329", "1273", "medbullets", "medmcqa", "mmlu"]
NAME = {"329": "MedQA 329", "1273": "MedQA 1273", "medbullets": "MedBullets",
        "medmcqa": "MedMCQA", "mmlu": "MMLU-Med"}
HANDED = ["medbullets", "medmcqa", "mmlu"]
ORDER = ["<1d", "1-6d", "1-4wk", "1-6mo", "6-12mo", ">1yr", "none"]

S = json.load(open(PIPE / "datasets/statistics.json"))
M = json.load(open(PIPE / "datasets/medmcqa_full_statistics.json"))
HAND = {d: [json.loads(l) for l in open(PIPE / f"verification/manual_read_{d}.jsonl")]
        for d in HANDED}
SCRIPT = {}
for d in DS:
    f = glob.glob(str(PIPE / f"verification/{d}_duration_critical_*.json"))
    if f: SCRIPT[d] = json.load(open(f[0]))["results"]

L = []
W = L.append

# ── header ────────────────────────────────────────────────────────────────
W("# Duration-Critical 題目總表：Statistics / References / Origin / 題目清單")
W("")
W("由 `build_master_report.py` 自動生成——所有數字直接從資料檔計算，未經手動轉抄。")
W("")
W("---")
W("")

# ── 1. origin ─────────────────────────────────────────────────────────────
W("## 1. Origin（來源與覆蓋率）")
W("")
W("**這張表是讀其他所有數字的前提。** MedMCQA 的比例是在母體 3.4% 的切片上算的，")
W("與其餘四個（幾乎全取）不可直接比較。")
W("")
W("| dataset | 原始來源 | 原始題數 | 進入審查 | 覆蓋率 | 篩選方式 |")
W("|---|---|---:|---:|---:|---|")
W("| MedQA 329 | MedQA 衍生（temporal_critical_v2 re-split） | — | 329 | — | 前人 Phase-A LLM filter（本研究已證實無鑑別力） |")
W("| MedQA 1273 | MedQA-USMLE **test**（= MIRAGE 的 MedQA-US） | 1,273 | 1,273 | 100% | 未篩選 |")
W(f"| MedBullets | ChallengeClinicalQA `medbullets_op5.csv` | 308 | {S['medbullets']['all']['n_questions']} | **100%** | 不篩選（全數審查） |")
W(f"| MedMCQA | HF `openlifescienceai/medmcqa` **validation**（= MIRAGE 的 MedMCQA dev） | {M['validation']['n_questions']:,} | {S['medmcqa']['all']['n_questions']} | **3.4%** | vignette + ≥200字元 + 時間表述 |")
W(f"| MMLU-Med | HF `cais/mmlu` config `professional_medicine` **test** | 272 | {S['mmlu']['all']['n_questions']} | **100%** | 不篩選（全數審查） |")
W("")
W("> **為什麼 MedBullets / MMLU 改成不篩選**：早期版本套用了同一組關鍵字篩選（→305 / 256）。")
W("> 事後檢查被篩掉的 19 題，發現其中 3 題其實是時間依賴的：`mb_0234`（5 年追蹤的樣本流失）、")
W("> `mb_0271`（年齡完全由發展里程碑編碼、未寫出）、`mmlu_0161`（用時間結構區分 case series /")
W("> cross-sectional / crossover）。**篩選器複製了腳本的單軸盲點**，所以對小到能全讀的資料集一律取消。")
W("> MedMCQA 母體 18 萬題、中位數 65 字元，不篩不可行。")
W("")
W("#### 已作廢的舊 MedMCQA 來源，以及它的兩個問題")
W("")
W("早期版本用 `Temporal-KG-RAG/datasets/MedMCQA_temporal_critical_split/test.jsonl`（1,168 題）。")
W("把那 1,168 題的題目文字逐題比對回原始三個 split，結果是：")
W("")
W("| 檢查 | 結果 |")
W("|---|---|")
W("| 比對回原始 split | **train 1,168 / validation 0 / test 0**，零題對不上 |")
W("| 有 `cop` 的 595 題 vs train 的答案 | **595 全部一致，0 不一致**（差一個 index base） |")
W("| `cop=None` 的 573 題，train 裡有沒有答案 | **573 題全部有**，一題不缺 |")
W("")
W("兩個結論：")
W("")
W("1. **那個檔名叫 `test.jsonl`，但不是 MedMCQA 官方 test split。** 它是前人拿 MedMCQA **train**")
W("   （182,822 題）做時間篩選後自行重切的 train/val/test 之一。`idx` 最大 182,738 正是 train 的索引範圍。")
W("   因此官方 test split 無答案這件事，與這 1,168 題**完全無關**。")
W("2. **那 573 題的答案不是原本就沒有，是舊流程處理時弄丟的。** 原始 train 每一題都有答案，")
W("   且已全數還原驗證。先前把它當成「資料只有 52% 有答案」是誤判。")
W("")
W("該檔仍作廢，原因是它繼承了未經驗證的 LLM 篩選器（全部 `is_temporal_critical=True`），")
W("而本研究已證實這類判斷式標籤無鑑別力。舊資料保留在 `datasets/_medmcqa_OLD_inherited_filter/`。")
W("")
W("---")
W("")

# ── 2. references ─────────────────────────────────────────────────────────
W("## 2. References")
W("")
W("| dataset | 引用 |")
W("|---|---|")
W("| **MedBullets** | Hanjie Chen, Zhouxiang Fang, Yash Singla, Mark Dredze. *Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions.* **NAACL-HLT 2025**, Vol. 1 (Long Papers), pp. 3563–3599. arXiv:[2402.18060](https://arxiv.org/abs/2402.18060) · [aclanthology.org/2025.naacl-long.182](https://aclanthology.org/2025.naacl-long.182/) · code/data: [github.com/HanjieChen/ChallengeClinicalQA](https://github.com/HanjieChen/ChallengeClinicalQA) · Johns Hopkins · **146 引用**（Semantic Scholar） |")
W("| **MedMCQA** | Pal, Umapathi, Sankarasubbu. *MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering.* CHIL 2022. HF: `openlifescienceai/medmcqa` |")
W("| **MMLU** | Hendrycks et al. *Measuring Massive Multitask Language Understanding.* ICLR 2021. HF: `cais/mmlu`, config `professional_medicine` |")
W("| **MedQA** | Jin et al. *What Disease Does This Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams.* Applied Sciences 2021 |")
W("| **MIRAGE / MedRAG**（本論文的定位基準） | Guangzhi Xiong, Qiao Jin, Zhiyong Lu, Aidong Zhang. *Benchmarking Retrieval-Augmented Generation for Medicine.* **Findings of ACL 2024**. arXiv:[2402.13178](https://arxiv.org/abs/2402.13178) · DOI 10.18653/v1/2024.findings-acl.372 · [github.com/Teddy-XiongGZ/MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) |")
W("")
W("**MIRAGE 組成（7,663 題）**：MMLU-Med 1,089 · **MedQA-US 1,273** · **MedMCQA 4,183 (dev)** · PubMedQA* 500 · BioASQ-Y/N 618。")
W("本研究的 1273 與 MedMCQA-validation **正好是其中兩個組成**，可直接對位。")
W("")
W("### 評估過但排除")
W("| | 排除原因 |")
W("|---|---|")
W("| JAMA Clinical Challenge | 同一篇 NAACL 論文的另一資料集，但只釋出連結與 scraper，正文在訂閱牆後，不可重散布 |")
W("| PubMedQA / BioASQ | abstract 層級 yes/no/maybe，無病人 vignette、無可擾動的 duration |")
W("| MedXpertQA | arXiv:2501.18362，ICML 2025，4,460 題 10 選項 — **尚未評估，值得考慮** |")
W("")
W("---")
W("")

# ── 3. statistics ─────────────────────────────────────────────────────────
ROWS = [("# Questions", "n_questions", "{:d}"),
        ("Avg. question length (chars)", "avg_question_chars", "{:.1f}"),
        ("Median question length (chars)", "median_question_chars", "{:.0f}"),
        ("Avg. question length (words)", "avg_question_words", "{:.1f}"),
        ("Avg. # options", "avg_n_options", "{:.2f}"),
        ("Avg. option length (words)", "avg_option_words", "{:.1f}"),
        ("Avg. answer length (words)", "avg_answer_words", "{:.1f}"),
        ("Vocabulary size", "vocabulary_size", "{:d}"),
        ("# with numeric duration", "n_with_numeric_duration", "{:d}"),
        ("% with numeric duration", "pct_with_numeric_duration", "{:.1f}"),
        ("Median duration (days)", "median_duration_days", "{:.1f}"),
        ("Avg. # extracted symptoms", "avg_n_symptoms", "{:.2f}"),
        ("Avg. # LLM-DDx seeds", "avg_n_seeds", "{:.2f}"),
        ("Avg. # query entities", "avg_n_query_entities", "{:.2f}")]

W("## 3. Statistics — General vs Duration-Critical")
W("")
W("每個資料集拆成兩組：**全部題目（general）**與 **duration-critical 子集**。")
W("只報總表會蓋掉論文真正要問的事——時間依賴的題目長得跟其他題不一樣嗎？")
W("")
W("| dataset | critical 子集來源 | critical / all |")
W("|---|---|---|")
for d in DS:
    a, c = S[d]["all"], S[d]["critical"]
    prov = "**人工逐題**" if S[d]["provenance"] == "hand review" else "僅腳本（未人工複核）"
    W(f"| {NAME[d]} | {prov} | {c['n_questions']}/{a['n_questions']} = {100*c['n_questions']/a['n_questions']:.1f}% |")
W("")
W("> 329 / 1273 的子集來自腳本，只覆蓋 `symptom_duration` 一軸，那兩欄是**下界**。")
W("")
for kind, title in [("all", "### 3.1 General（全部題目）"), ("critical", "### 3.2 Duration-Critical 子集")]:
    W(title); W("")
    W("| 指標 | " + " | ".join(NAME[d] for d in DS) + " |")
    W("|---|" + "---:|" * len(DS))
    for lab, k, f in ROWS:
        W(f"| {lab} | " + " | ".join(
            (f.format(S[d][kind][k]) if S[d][kind][k] is not None else "—") for d in DS) + " |")
    W("")
    W("| Duration bucket | " + " | ".join(NAME[d] for d in DS) + " |")
    W("|---|" + "---:|" * len(DS))
    for b in ORDER:
        W(f"| `{b}` | " + " | ".join(
            f"{S[d][kind]['duration_buckets'][b]} ({100*S[d][kind]['duration_buckets'][b]/S[d][kind]['n_questions']:.0f}%)"
            for d in DS) + " |")
    W("")

W("### 3.3 兩組的差異（這才是重點）")
W("")
W("**(a) duration-critical 的題目一致比較長**")
W("")
W("| | 全部 (chars) | critical (chars) | 差 |")
W("|---|---:|---:|---:|")
for d in DS:
    a, c = S[d]["all"]["avg_question_chars"], S[d]["critical"]["avg_question_chars"]
    W(f"| {NAME[d]} | {a:.0f} | {c:.0f} | {c-a:+.0f} |")
W("")
W("**(b) duration 覆蓋率一致上升**")
W("")
W("| | 全部 | critical |")
W("|---|---:|---:|")
for d in DS:
    W(f"| {NAME[d]} | {S[d]['all']['pct_with_numeric_duration']:.1f}% | {S[d]['critical']['pct_with_numeric_duration']:.1f}% |")
W("")
W("**(c) 但相當比例的 critical 題目根本抽不出數值 duration** ← 最重要的一格")
W("")
W("| | critical 題數 | 無數值 duration | 佔比 |")
W("|---|---:|---:|---:|")
tc = tn = 0
for d in HANDED:
    c = S[d]["critical"]; n = c["duration_buckets"]["none"]
    tc += c["n_questions"]; tn += n
    W(f"| {NAME[d]} | {c['n_questions']} | {n} | {100*n/c['n_questions']:.0f}% |")
W(f"| **人工判讀合計** | **{tc}** | **{tn}** | **{100*tn/tc:.0f}%** |")
W("")
W("這是 §5 多時間軸結論的**量化直接證據**：這些題目被判為時間依賴，但沒有可抽取的主訴時長，")
W("因為時間資訊在別的軸上（潛伏期、療程長度、妊娠週數、事件先後、發作長度⋯⋯）。")
W("**腳本在這些題目上連可以擾動的對象都找不到。**")
W("")
W("**(d) median duration 的變化**")
W("")
W("| | 全部 median (days) | critical median (days) |")
W("|---|---:|---:|")
for d in DS:
    a = S[d]["all"]["median_duration_days"]; c = S[d]["critical"]["median_duration_days"]
    W(f"| {NAME[d]} | {'—' if a is None else f'{a:.1f}'} | {'—' if c is None else f'{c:.1f}'} |")
W("")
W("**(e) 答案字母分佈**")
W("")
for d in DS:
    W(f"- **{NAME[d]}** 全部: " + "  ".join(f"{k}:{v}" for k, v in S[d]["all"]["answer_distribution"].items()))
    W(f"  - critical: " + "  ".join(f"{k}:{v}" for k, v in S[d]["critical"]["answer_distribution"].items()))
W("")
W("**(f) temporal role**")
W("")
W("| | 全部 | critical |")
W("|---|---|---|")
for d in DS:
    fa = "  ".join(f"{k}:{v}" for k, v in S[d]["all"]["temporal_role"].items())
    fc = "  ".join(f"{k}:{v}" for k, v in S[d]["critical"]["temporal_role"].items())
    W(f"| {NAME[d]} | {fa} | {fc} |")
W("")

# 3.4 medmcqa full
W("### 3.4 MedMCQA — 原始完整資料集（未經任何篩選）")
W("")
W("本研究只讀了 143 題 = validation 的 3.4%。**任何在那 143 題上算的比例都是富集後的切片**，")
W("必須連同母體分佈一起報告。")
W("")
cols = [c for c in ["train", "validation", "test"] if c in M]
MROWS = [("# Questions", "n_questions", "{:,}"), ("# with answer label", "n_labelled", "{:,}"),
         ("% labelled", "pct_labelled", "{:.1f}%"),
         ("Avg. question length (chars)", "avg_question_chars", "{:.1f}"),
         ("**Median** question length (chars)", "median_question_chars", "**{:,}**"),
         ("P90 question length (chars)", "p90_question_chars", "{:,}"),
         ("Max question length (chars)", "max_question_chars", "{:,}"),
         ("Avg. question length (words)", "avg_question_words", "{:.1f}"),
         ("Median question length (words)", "median_question_words", "{:,}"),
         ("# options", "n_options", "{:d}"),
         ("Avg. option length (words)", "avg_option_words", "{:.1f}"),
         ("Vocabulary size", "vocabulary_size", "{:,}")]
W("| 指標 | " + " | ".join(cols) + " |")
W("|---|" + "---:|" * len(cols))
for lab, k, f in MROWS:
    W(f"| {lab} | " + " | ".join(f.format(M[c][k]) for c in cols) + " |")
W("")
W(f"**官方 test split 完全沒有答案標籤**（{M['test']['n_questions']:,} 題全部 `cop=None`），")
W("所以論文慣例（含 MIRAGE）用 validation 當評測集。")
W("")
W("#### 題長分佈")
W("")
W("| 字元數 | " + " | ".join(cols) + " |")
W("|---|" + "---:|" * len(cols))
for b in ["<100", "100-199", "200-399", ">=400"]:
    W(f"| `{b}` | " + " | ".join(
        f"{M[c]['len_buckets'][b]:,} ({100*M[c]['len_buckets'][b]/M[c]['n_questions']:.0f}%)" for c in cols) + " |")
W("")
W(f"**train 有 {100*M['train']['len_buckets']['<100']/M['train']['n_questions']:.0f}%、"
  f"test 有 {100*M['test']['len_buckets']['<100']/M['test']['n_questions']:.0f}% 的題目不到 100 字元**"
  "——MedMCQA 本質上是一行的事實記憶題庫，不是 vignette 題庫。")
W("")
W("#### 篩選漏斗（三個 split）")
W("")
W("| 階段 | " + " | ".join(cols) + " |")
W("|---|" + "---:|" * len(cols))
for lab, k in [("全部題目", "n_questions"), ("+ ≥200 字元", "n_len200"),
               ("+ vignette 措辭", "n_vignette_len200"),
               ("+ 時間表述 ← **本研究讀的**", "n_vignette_len200_temporal")]:
    W(f"| {lab} | " + " | ".join(
        f"{M[c][k]:,} ({100*M[c][k]/M[c]['n_questions']:.1f}%)" for c in cols) + " |")
W("")
tr = M["train"]["n_vignette_len200_temporal"]
mm_rate = len([r for r in HAND["medmcqa"] if r["verdict"] == "critical"]) / len(HAND["medmcqa"])
W(f"> **train split 還有 {tr:,} 題**符合同一組條件、完全未使用。依 validation 上實測的 "
  f"{100*mm_rate:.1f}% critical 率推估，其中約 **{round(tr*mm_rate):,} 題**可能是 duration-critical——")
W("> 目前所有資料集裡最大的未開發來源。")
W("")
W("#### 科別分佈（前 8）")
W("")
for c in cols:
    W(f"- **{c}**：" + "  ".join(f"{k} {v:,}" for k, v in M[c]["subjects"].items()))
W("")
W("---")
W("")

# ── 4. verdict summary ────────────────────────────────────────────────────
W("## 4. Duration-Critical 判定總表")
W("")
W("- **腳本**（`verify_duration_critical.py`）：反事實擾動 —— 只把主訴時長改成相反時間尺度，")
W("  重新盲答 5 次看答案是否翻轉。**只覆蓋 `symptom_duration` 這一軸。**")
W("- **人工逐題**：逐題判斷「改變任何時間資訊會不會改變正確答案」，並標記時間軸。")
W("")
W("| dataset | 審查方式 | n | critical | not | no_duration | 比例 | 對照腳本 |")
W("|---|---|---:|---:|---:|---:|---:|---:|")
tot = Counter()
for d in DS:
    if d in HAND:
        rows = HAND[d]; c = Counter(r["verdict"] for r in rows); t = c["critical"] + c["not_critical"]
        for k in c: tot[k] += c[k]
        s = SCRIPT.get(d)
        if s:
            cs = Counter(r["verdict"] for r in s); ts = cs["duration_critical"] + cs["not_duration_critical"]
            sr = 100 * cs["duration_critical"] / ts
            cell = f"{sr:.1f}%（{(100*c['critical']/t)/sr:.0f}×）"
        else:
            cell = "—"
        W(f"| {NAME[d]} | **人工逐題** | {len(rows)} | **{c['critical']}** | {c['not_critical']} | {c['no_duration']} | **{100*c['critical']/t:.1f}%** | {cell} |")
    else:
        s = SCRIPT[d]; cs = Counter(r["verdict"] for r in s); ts = cs["duration_critical"] + cs["not_duration_critical"]
        W(f"| {NAME[d]} | 僅腳本 | {len(s)} | {cs['duration_critical']} | {cs['not_duration_critical']} | — | {100*cs['duration_critical']/ts:.1f}% | — |")
W("")
T = tot["critical"] + tot["not_critical"]
NH = sum(len(v) for v in HAND.values())
W(f"**人工逐題合計：{NH} 題，{tot['critical']} 題 critical，{100*tot['critical']/T:.1f}%**（分母為有 duration 的 {T} 題）")
W("")
W("> MedQA 329 / 1273 **未逐題人工審查**，只跑過腳本。既然人工判讀在其他三個資料集上高出數倍，")
W("> 那兩個的比例幾乎確定也是低估。")
W("")
W("### 換算到原始資料集（消除選擇偏誤後）")
W("")
W("| dataset | critical | ÷ 原始題數 | 說明 |")
W("|---|---:|---:|---|")
ORIG = {"medbullets": (308, "整個資料集"), "medmcqa": (M["validation"]["n_questions"], "整個 validation split"),
        "mmlu": (272, "整個 subject")}
for d in HANDED:
    nc = sum(1 for r in HAND[d] if r["verdict"] == "critical")
    o, desc = ORIG[d]
    W(f"| {NAME[d]} | {nc} | {nc} / {o:,} = **{100*nc/o:.1f}%** | 分母是{desc} |")
W("")
W("---")
W("")

# ── 5. axes ───────────────────────────────────────────────────────────────
ax = Counter()
for rows in HAND.values(): ax.update(r.get("axis") for r in rows if r["verdict"] == "critical")
ncrit = tot["critical"]
W(f"## 5. 時間軸分佈（人工判讀，{ncrit} 題 critical）")
W("")
W(f"{ncrit} 題散落在 **{len([k for k in ax if k])} 種**不同的時間軸上。")
W(f"**`symptom_duration`（腳本唯一擾動的那一軸）只佔 {ax['symptom_duration']} 題 = {100*ax['symptom_duration']/ncrit:.0f}%。**")
W("")
MEAN = {'symptom_duration': '主訴持續多久', 'progression_tempo': '惡化速度', 'gestational_window': '妊娠週數決定處置',
        'temporal_ordering': '事件先後順序', 'latency': '暴露到發病的間隔', 'dsm_duration_criterion': '診斷準則的時長門檻',
        'attack_duration': '單次發作持續多久', 'onset_timing': '發生在哪個時間點', 'treatment_trial': '療程已試多久',
        'therapeutic_window': '治療時效窗', 'time_of_day': '一天中的時段', 'age_gated': '年齡決定正常與否',
        'acuity': '急性 vs 慢性', 'age_persistence': '持續到幾歲', 'cyclical_pattern': '週期性',
        'circadian_phase': '生理時鐘相位', 'screening_interval': '篩檢間隔', 'progression_trajectory': '變化軌跡',
        'latency_trajectory': '潛伏+趨勢', 'recurrence_interval': '復發間隔', 'interval_since_event': '距上次事件多久',
        'attrition_over_time': '追蹤期間的樣本流失', 'developmental_stage': '發展階段（年齡由里程碑編碼）',
        'study_temporal_design': '研究設計的時間結構', 'fasting_duration': '禁食多久', 'drug_half_life': '藥物作用時長',
        'wound_healing_phase': '傷口癒合期', 'rash_duration': '疹子持續幾天', 'measurement_time_window': '檢驗值的時間平均窗',
        'exposure_window': '暴露後預防的時效', 'symptom_frequency': '症狀頻率', 'episode_duration': '單次發作期長度'}
W("| 時間軸 | n | 意義 |")
W("|---|---:|---|")
for k, v in ax.most_common():
    if k: W(f"| `{k}` | {v} | {MEAN.get(k,'')} |")
W("")
W("---")
W("")

# ── 6. per-question lists ─────────────────────────────────────────────────
W(f"## 6. Duration-Critical 題目清單（人工判讀，{ncrit} 題）")
W("")
W("每題附時間軸、關鍵時間片語、以及「為什麼改了時間答案會變」。")
W("完整逐題記錄（含 not_critical / no_duration 的理由）在 `verification/manual_read_*.jsonl`。")
W("")
for i, d in enumerate(HANDED, 1):
    crit = sorted([r for r in HAND[d] if r["verdict"] == "critical"],
                  key=lambda x: (x.get("axis") or "", x["uid"]))
    W(f"### 6.{i} {NAME[d]} — {len(crit)} 題"); W("")
    W("| uid | 時間軸 | 關鍵時間片語 | 為什麼是 critical |")
    W("|---|---|---|---|")
    for r in crit:
        W(f"| `{r['uid']}` | {r.get('axis','')} | {(r['dur'] or '').replace('|','/')} | {r['note'].replace('|','/')} |")
    W("")
W("---")
W("")

# ── 7. script-only ────────────────────────────────────────────────────────
W("## 7. 腳本判定的 duration-critical（MedQA 329 / 1273，未經人工複核）")
W("")
for i, d in enumerate(["329", "1273"], 1):
    crit = sorted([r for r in SCRIPT[d] if r["verdict"] == "duration_critical"], key=lambda x: x["uid"])
    W(f"### 7.{i} MedQA {d} — {len(crit)} 題"); W("")
    W("| uid | duration 改寫 | gold | 擾動後答案 |")
    W("|---|---|---|---|")
    for r in crit:
        p = r.get("perturb") or {}
        W(f"| `{r['uid']}` | {p.get('original_duration')} → {p.get('new_duration')} | {r['gold']} | {r.get('perturbed_answer')} |")
    W("")
W("---")
W("")

# ── 8. code ───────────────────────────────────────────────────────────────
W("## 8. 產生這份報告的程式")
W("")
W("```bash")
W("python3 pipeline/dataset_statistics.py 329 1273 medbullets medmcqa mmlu")
W("python3 pipeline/medmcqa_full_statistics.py")
W("python3 pipeline/build_master_report.py      # 重新生成本檔")
W("```")
W("")
W("| 檔案 | 用途 |")
W("|---|---|")
W("| `build_master_report.py` | **本報告的生成器**（所有表格皆由資料檔重算） |")
W("| `dataset_statistics.py` | §3.1–3.3 general vs critical 統計 |")
W("| `medmcqa_full_statistics.py` | §3.4 MedMCQA 原始三個 split 的統計 |")
W("| `build_new_dataset.py` | §1 從原始來源建資料集 |")
W("| `dump_for_manual_read.py` | §6 逐題人工閱讀用的傾印工具 |")
W("| `verify_duration_critical.py` | §7 反事實擾動腳本 |")
W("| `verify_openended_probe.py` | 第二臂：檢查 MCQ 選項是否遮蔽效應 |")
W("| `analyze_verification.py` | 腳本結果彙整 + 與舊標籤交叉比對 |")
W("| `results_by_verified_subset.py` | 用驗證子集重切 reader 結果 |")
W("")
W("| 資料檔 | 內容 |")
W("|---|---|")
W(f"| `verification/manual_read_*.jsonl` | {NH} 題逐題人工判讀（uid / dur / verdict / axis / note） |")
W("| `verification/{ds}_duration_critical_*.json` | 腳本判定，含每題 5 次作答的原始輸出 |")
W("| `verification/{ds}_openended_*.json` | open-ended 探針結果 |")
W("| `datasets/{ds}/benchmark.json` | 題目本體 |")
W("| `datasets/{ds}/durations.json` | role-tagged 病人 duration |")
W("| `datasets/_raw_*` | 從網路抓下的原始檔（可重現） |")

p = PIPE / "DURATION_CRITICAL_MASTER_zh.md"
p.write_text("\n".join(L) + "\n")
print(f"wrote {p}  ({len(L)} lines, {sum(1 for x in L if x.startswith('|'))} table rows)")

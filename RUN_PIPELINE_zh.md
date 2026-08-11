# 完整 pipeline 執行說明（含 Neo4j 重新 retrieve）

整個流程分兩個階段：**建立並凍結（build & freeze）** 每個方法的「每題 prompt + KG」，
再 **讀取回答（read）**。retrieval 只跑一次並凍結成檔案，reader 之後直接重播即可。
`DATASET` 填 `1273` 或 `329`。所有指令都在 repo 根目錄（`new_duration_spectrum/`）執行，
不要在 `pipeline/` 裡面跑。

---

## 0. 前置作業

**Neo4j**（只有 retrieval 的步驟 2–4 需要）。資料庫在 msi-gpu 主機，先開 tunnel：


> **憑證**：SSH 與 Neo4j 的帳密不寫在這份文件裡。設定 `SSH_HOST` / `SSH_PORT` / `SSH_USER` /
> `SSH_PASS` / `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` 環境變數後再執行以下指令；
> 實際值放在未進版控的 `CLAUDE.local.md`。

```bash
sshpass -p "$SSH_PASS" ssh -N -f -L 7687:localhost:7687 -L 4645:localhost:4645 \
  -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 -p "$SSH_PORT" "$SSH_USER@$SSH_HOST"

# 驗證：應該回傳 443699
python3 -c "import sys;sys.path.insert(0,'pipeline/code');sys.path.insert(0,'pipeline/code/dkr_policy');\
from umls_neo4j import get_driver;\
print(get_driver().session().run('MATCH (c:Concept) RETURN count(c) AS n').single()['n'])"
```

**API keys** 放在 `pipeline/api_key/`。**環境**：SSH-200 的 venv，或任何裝了
`torch transformers neo4j numpy openai` 的 Python。

⚠ **記憶體**：每個 retrieval worker 會佔約 4 GB（SapBERT 矩陣）。在 38 GB 機器上
`WALKER_N_WORKERS=3` 是安全值；開到 6 曾造成 OOM 重開機。
另外，砍掉任何 build 之後，一定要再跑 `pkill -9 -f eval_kg_walker_full.py`
（因為 OUT_TAG 是環境變數、不在指令列裡，用 pgrep 抓 tag 會漏掉那些 worker）。

---

## 階段 A — 建立並凍結所有方法 → `frozen/<ds>/<method>.json`

### 1. 純 prompt 方法（不需要 Neo4j）
```bash
DATASET=1273 python3 pipeline/build_prompt_only.py       # → vanilla.json, cot.json
```

### 2. 我們的方法（walker）與其 ablation（需要 Neo4j）
```bash
# overlap = 我們的方法（bc = Bhattacharyya 重疊），輸出 frozen/1273/walker.json
DATASET=1273 WALKER_BC_MODE=overlap         WALKER_METHOD_NAME=walker \
  WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=3 python3 pipeline/build_kg.py

# interval ablation（bc = 從分布抽點），輸出 frozen/1273/walker_interval.json
DATASET=1273 WALKER_BC_MODE=interval_sample WALKER_METHOD_NAME=walker_interval \
  WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=3 python3 pipeline/build_kg.py
```

每 25 題存一次檔，**可續跑**（重跑同指令即可接續）。
每行進度會印 `LLM-gen N (N/q)`，也就是 duration 生成的成本；
**要盯這個數字 —— 超過 40/q 代表 cache 沒有被重用**（路徑錯了或有殘留的 orphan worker）。
若要改分數公式，先改 `code/dkr_policy/kg_walker.py` 的 `walk()`，再重跑這一步。

| 環境變數 | 意義 |
|---|---|
| `WALKER_RETRIEVAL_ONLY=1` | **一定要加** —— 跳過 build 內多餘的 reader 呼叫（reader 在階段 B 才跑） |
| `WALKER_N_WORKERS` | process 數；每個約 4 GB。38 GB 機器建議 3 |
| `WALKER_BC_MODE` | `overlap`（我們的方法）或 `interval_sample`（ablation） |
| `WALKER_METHOD_NAME` | 輸出檔名 —— `walker_interval` 會寫到另一個檔，不會覆蓋 `walker.json` |

### 3. Raw KG dump（需要 Neo4j）
```bash
DATASET=1273 python3 pipeline/build_raw_hops.py          # → raw_1hop.json, raw_2hop.json
```
prompt 裡放 10 個概念（與 walker 的 top-10 對齊），但記錄中保留 50 個供後續分析。

### 4. KG-RAG baselines（需要 Neo4j + LLM；先寫到 cache，再 freeze）
```bash
DD=pipeline/datasets/1273

# ToG
BENCH_PATH=$DD/benchmark.json SEEDS_PATH=$DD/seeds.json SYM_PATH=$DD/symptoms.json \
  TOG_LLM=gpt-5.4-mini KG_OUT_TAG=1273 TOG_WORKERS=4 python3 pipeline/code/tog_baseline.py
DATASET=1273 METHOD=tog   SRC=cache/tog_baseline_1273__gpt-5.4-mini.json   python3 pipeline/freeze_baseline.py

# HyKGE
BENCH_PATH=$DD/benchmark.json SEEDS_PATH=$DD/seeds.json SYM_PATH=$DD/symptoms.json \
  HYKGE_LLM=gpt-5.4-mini KG_OUT_TAG=1273 HYKGE_WORKERS=2 python3 pipeline/code/hykge_baseline.py
DATASET=1273 METHOD=hykge SRC=cache/hykge_baseline_1273__gpt-5.4-mini.json python3 pipeline/freeze_baseline.py

# MedRAG（教科書；不需 Neo4j，但需要 LLM）
BENCH_PATH=$DD/benchmark.json MEDRAG_OUT_TAG=1273 python3 pipeline/code/run_medrag_textbook.py
DATASET=1273 METHOD=medrag SRC=cache/1273_medrag_textbook_k32__gpt-5.4-mini.json python3 pipeline/freeze_baseline.py
```
ToG / HyKGE 都用論文預設值（`TOG_WIDTH/DEPTH=3`、`TOG_REL_THRESH=0.2`、
`TOG_ENTITY_PRUNE=llm`；`HYKGE_HOP=3`、`HYKGE_TOPK=10`）。
與原論文/原始碼的差異，寫在各檔案開頭的 fidelity 註解。

---

## 階段 B — 讀取回答 → `results/<ds>_<method>_<model>.json`

這階段**不需要 Neo4j**。凍結完成後，任何模型都能直接套用任何方法：

```bash
for M in vanilla cot raw_1hop raw_2hop medrag tog hykge walker walker_interval; do
  DATASET=1273 METHOD=$M MODEL=gpt-5.4-mini N_RUNS=3 WORKERS=10 \
    RESULTS_DIR=results/round2_intentfree python3 pipeline/run_reader.py
done
```

- `MODEL` 依名稱自動路由（`code/llm_client.py`）：`gpt-*`→OpenAI、`gemini-3*`/`gemma*`→litellm、
  `google:<m>`→Gemini 直連、`together:<m>`→Together、`vllm:<m>`/`qwen*`→本地 vLLM。
- reader 有隨機性（temperature 1.0）→ **N_RUNS≥3，回報 mean±std**。
  輸出會存下每題每次的 prompt 與原始輸出，方便事後稽核。
- **小模型 / 開源模型**（Qwen 等）改用 `run_reader_block.py`：送同一份凍結 prompt，
  只把答案格式換成 ReinRAG 的 `[Answer]` 區塊（可續跑、會記錄 `n_unparseable`）。
  在 Together 上 `WORKERS≈3` 即可 —— **unparseable 偏高通常是被限流，不是模型變差**。

---

## 階段 B'（新增）— 調 prompt：不用重跑 retrieval

`frozen/<ds>/<method>.json` 裡 **`kg_block` 是跟 `prompt` 分開存的**，
所以「換 prompt 措辭」完全不需要碰 Neo4j，也不需要任何 LLM 呼叫 —— 幾秒就好。
證據完全不變，變的只有措辭，因此 prompt 變體之間是**乾淨的受控對照**。

### 1. 寫一份新 prompt
可用的 placeholder：`{question}` `{options_block}` `{kg_block}` `{patient_dur_str}`
（省略 `{kg_block}` 就是「同樣題目、不給證據」的對照組）

```bash
cat > my_prompt.txt <<'EOF'
[TASK] You are a diagnostician. Pick the single best option.

[CASE]
{question}

[OPTIONS]
{options_block}

[SYMPTOM DURATION] {patient_dur_str}

[KG EVIDENCE — ranked by symptom match (cos) + duration fit (bc)]
{kg_block}

Reason briefly, then answer with one letter in <a></a> tags.
EOF
```

### 2. 重繪（秒級，不碰 Neo4j / 不呼叫 LLM）
```bash
DATASET=1273 METHOD=walker VARIANT=myv1 TEMPLATE_FILE=my_prompt.txt \
  python3 pipeline/rerender_prompts.py
# → frozen/1273/walker__myv1.json
```
也可以直接用 `prompts.py` 裡現成的模板：`TEMPLATE=RAW_KG`（取代 `TEMPLATE_FILE`）。
輸出會印 `kg_block injected : N/總數`，確認證據確實有進到 prompt。

### 3. 丟給任何模型（含本地模型）
```bash
# 本地 vLLM
VLLM_BASE_URL=http://localhost:8000/v1 \
DATASET=1273 METHOD=walker__myv1 MODEL=vllm:Qwen3.5-9B N_RUNS=3 WORKERS=8 \
  RESULTS_DIR=results/prompt_tuning python3 pipeline/run_reader.py

# 小模型建議改用 block reader（ReinRAG [Answer] 格式，parse 失敗率低很多）
VLLM_BASE_URL=http://localhost:8000/v1 \
DATASET=1273 MODEL=vllm:Qwen3.5-9B METHODS=walker__myv1 RUN=1 WORKERS=8 \
  python3 pipeline/run_reader_block.py
```

`MODEL` 前綴決定路由（`code/llm_client.py`）：
`vllm:<model>` 或 `qwen*` → 本地 vLLM（讀 `VLLM_BASE_URL`，預設 `http://localhost:8000/v1`）。
本地模型沒有 rate limit，`WORKERS` 可以開高。

### 掃多個 prompt 變體
```bash
for V in v1 v2 v3; do
  DATASET=1273 METHOD=walker VARIANT=$V TEMPLATE_FILE=prompt_$V.txt python3 pipeline/rerender_prompts.py
  DATASET=1273 METHOD=walker__$V MODEL=vllm:Qwen3.5-9B N_RUNS=3 \
    RESULTS_DIR=results/prompt_tuning python3 pipeline/run_reader.py
done
```
所有變體共用同一份 KG 證據，所以差異純粹來自 prompt。

---

## 哪個程式產生哪個 frozen 檔

| frozen 檔 | 由誰產生 | 需要 Neo4j？ |
|---|---|---|
| `vanilla.json`, `cot.json` | `build_prompt_only.py` | 否 |
| `walker.json` | `build_kg.py`（BC_MODE=overlap） | **是** |
| `walker_interval.json` | `build_kg.py`（BC_MODE=interval_sample） | **是** |
| `raw_1hop.json`, `raw_2hop.json` | `build_raw_hops.py` | **是** |
| `tog.json` | `tog_baseline.py` → `freeze_baseline.py` | **是** |
| `hykge.json` | `hykge_baseline.py` → `freeze_baseline.py` | **是** |
| `medrag.json` | `run_medrag_textbook.py` → `freeze_baseline.py` | 否 |

若要重建 329 的整套，把階段 A 每一步都改成 `DATASET=329`
（並設 `KG_OUT_TAG=329` / `MEDRAG_OUT_TAG=329`），階段 B 也用 `DATASET=329`。
注意：**329 是獨立的資料集，不是 1273 的子集**（只有 23 題重疊），必須獨立建立與獨立評測。

---

## 目前 1273 結果（`results/round2_intentfree/`，gpt-5.4-mini，N=3）

| 方法 | 1273 accuracy |
|---|---|
| walker_interval | 89.37 |
| raw_1hop | 88.98 ± 0.63 |
| **walker（我們的方法）** | **88.74 ± 0.13** |
| raw_2hop | 88.53 ± 0.29 |
| hykge | 87.77 ± 0.26 |

在未經篩選的完整 1273 上，各方法在統計上是打平的 —— 因為大多數題目本來就不是
duration-critical，整體聚合數字分不出差異。時間關鍵的分層在 329（獨立資料集）。

---

## 自足性（self-contained）驗證

`pipeline/` **已可獨立執行全部流程**，不依賴外層任何檔案。2026-07-26 實測：
以 `cwd=pipeline/` 完整跑過 build_kg（Neo4j retrieval + freeze）與 run_reader，
兩階段都通過。

修正過的外部依賴：
1. `build_oneshot_example.py` 原本 `sys.path` 指向 `../scripts/` → 已把
   `extract_patient_duration_llm.py` 複製進 `pipeline/code/`
2. `build_kg.py` 的 subprocess 原本 `cwd=` 外層 → 改成 `cwd=pipeline/`，
   讓 `DKR_MATRIX_PKL="cache/..."` 這類 cwd-relative 路徑落在 `pipeline/cache`
3. `spectrum_mdn.py` 的 api_key fallback 路徑 → 改指 `pipeline/api_key`
4. 補上 `build_prompt_only.py`（vanilla/cot 原本沒有 builder，只有舊腳本產生過）

唯一的外部條件是**服務**而非檔案：Neo4j（bolt://localhost:7687）與 LLM API。

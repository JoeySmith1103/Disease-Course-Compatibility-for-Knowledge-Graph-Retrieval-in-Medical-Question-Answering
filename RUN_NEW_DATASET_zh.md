# 在 MedBullets / MMLU-Med 上跑我們的方法 — 完整流程

`RUN_PIPELINE_zh.md` 寫的是 329/1273 **已經備妥輸入檔**之後的流程。
新資料集多了一個「階段 0：準備輸入」，因為 `benchmark.json` 以外的四個檔案都要現生。

所有指令都在 repo 根目錄 `new_duration_spectrum/` 執行，**不要進 `pipeline/`**。
`DATASET` 填 `medbullets` 或 `mmlu`。

---

## 0. 環境與連線

### 0.1 SSH：Neo4j tunnel（只有需要 Neo4j 的步驟才要）

UMLS 知識圖在 msi-gpu 主機，本機要先開通道把 7687 轉過來：


> **憑證**：SSH 與 Neo4j 的帳密不寫在這份文件裡。設定 `SSH_HOST` / `SSH_PORT` / `SSH_USER` /
> `SSH_PASS` / `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` 環境變數後再執行以下指令；
> 實際值放在未進版控的 `CLAUDE.local.md`。

```bash
sshpass -p "$SSH_PASS" ssh -N -f -L 7687:localhost:7687 -L 4645:localhost:4645 \
  -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 -p "$SSH_PORT" "$SSH_USER@$SSH_HOST"
```

| 項目 | 值 |
|---|---|
| host | `$SSH_HOST` port `$SSH_PORT`（見 CLAUDE.local.md，未進版控） |
| user / pw | `$SSH_USER` / `$SSH_PASS`（見 CLAUDE.local.md，未進版控） |
| 轉發埠 | `7687`（bolt）、`4645` |
| Neo4j 連線 | `bolt://localhost:7687`，帳密 `$NEO4J_USER` / `$NEO4J_PASSWORD`（見 CLAUDE.local.md，未進版控） |

**驗證通道**（應回傳 `443699`）：

```bash
python3 -c "import sys;sys.path.insert(0,'pipeline/code');sys.path.insert(0,'pipeline/code/dkr_policy');\
from umls_neo4j import get_driver;\
print(get_driver().session().run('MATCH (c:Concept) RETURN count(c) AS n').single()['n'])"
```

通道斷掉的徵兆是 build 卡住或 `ServiceUnavailable`。重開前先確認舊的沒殘留：

```bash
pgrep -fa "ssh -N -f -L 7687" || echo "沒有既有 tunnel"
```

### 0.2 SSH：跑大模型 / vLLM 的機器（可選）

只有階段 B 想用本地開源模型當 reader 才需要。retrieval 不需要 GPU。

```bash
ssh 200                                                    # ProxyJump 經 190
source /mnt/NAS/home/tommylee/TimeRAG/.env/bin/activate
# 2× RTX 4090，tensor_parallel_size=2
# 模型在 /mnt/NAS/home/tommylee/hf_models/
```

啟 vLLM 之後，本機 reader 用 `MODEL=vllm:<name>` 並設 `VLLM_BASE_URL`。

### 0.3 API keys 與記憶體

- API keys 放 `pipeline/api_key/`。
- ⚠ **每個 retrieval worker 約佔 4 GB**（SapBERT 矩陣）。38 GB 機器 `WALKER_N_WORKERS=3` 是安全值，
  開到 6 曾造成 OOM 重開機。
- ⚠ 砍掉任何 build 之後，**一定要**再跑一次 `pkill -9 -f eval_kg_walker_full.py`。
  `WALKER_OUT_TAG` 是環境變數、不在指令列裡，用 pgrep 抓 tag 會漏掉 ProcessPool 的 worker，
  殘留的 orphan 會繼續燒 LLM 額度。

---

## 階段 0 — 準備輸入檔（新資料集才需要）

`build_kg.py` 會把五個檔案接進 walker：

| 檔案 | 內容 | 產生方式 | 現況 |
|---|---|---|---|
| `benchmark.json` | 題目 / 選項 / 答案 | `build_new_dataset.py` | ✅ 已有 |
| `durations.json` | role-tagged 病人 duration | `code/extract_patient_duration_llm.py` | ✅ 已有 |
| `seeds.json` | LLM 鑑別診斷假設 → walker 種子 CUI | `prepare_dataset_inputs.py` | ❌ 要生 |
| `symptoms.json` | 主訴症狀 → SapBERT query embedding | `prepare_dataset_inputs.py` | ❌ 要生 |
| `query_entities.json` | 依角色分類的實體 → 多角色種子 | `prepare_dataset_inputs.py` | ❌ 要生 |

`intents.json` **不需要**——移除 intent-aware retrieval 之後 `build_kg.py` 就不再讀它。

```bash
# 一次生三個檔（可續跑；LLM 呼叫，不需要 Neo4j）
DATASET=medbullets MODEL=gpt-5.4-mini WORKERS=8 python3 pipeline/prepare_dataset_inputs.py
DATASET=mmlu       MODEL=gpt-5.4-mini WORKERS=8 python3 pipeline/prepare_dataset_inputs.py
```

約 3 × n 次 LLM 呼叫（MedBullets 308 題 ≈ 924 次，MMLU 272 題 ≈ 816 次）。
只想重生其中一個檔就加 `ONLY=seeds` / `ONLY=symptoms` / `ONLY=query_entities`。

**檢查**（`symptoms` 空集合是正常的——純生理學題本來就沒有症狀）：

```bash
python3 -c "
import json
for d in ['medbullets','mmlu']:
    b=len(json.load(open(f'pipeline/datasets/{d}/benchmark.json')))
    for f in ['durations','seeds','symptoms','query_entities']:
        j=json.load(open(f'pipeline/datasets/{d}/{f}.json'))
        ne=sum(1 for v in j.values() if any(v.get(k) for k in v))
        print(f'{d:11}{f:16} {len(j):4d}/{b}  非空 {ne}')"
```

---

## 階段 A — 建立並凍結各方法 → `frozen/<ds>/<method>.json`

### A1. 純 prompt（不需 Neo4j）
```bash
DATASET=medbullets python3 pipeline/build_prompt_only.py    # → vanilla.json, cot.json
```

### A2. 我們的方法 walker + interval ablation（**需要 Neo4j**）
```bash
DATASET=medbullets WALKER_BC_MODE=overlap         WALKER_METHOD_NAME=walker \
  WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=3 python3 pipeline/build_kg.py

DATASET=medbullets WALKER_BC_MODE=interval_sample WALKER_METHOD_NAME=walker_interval \
  WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=3 python3 pipeline/build_kg.py
```

| 環境變數 | 意義 |
|---|---|
| `WALKER_RETRIEVAL_ONLY=1` | **必加**——跳過 build 內多餘的 reader 呼叫 |
| `WALKER_N_WORKERS` | process 數，每個約 4 GB |
| `WALKER_BC_MODE` | `overlap`（我們的方法）/ `interval_sample`（ablation） |
| `WALKER_METHOD_NAME` | 決定輸出檔名，避免互相覆蓋 |

每 25 題存檔一次、可續跑。進度行會印 `LLM-gen N (N/q)`，
**要盯這個數字——超過 40/q 代表 duration cache 沒被重用**（路徑錯或有 orphan worker）。

分數公式 `score = cos + λ·bc − μ·hop` 在 `code/dkr_policy/kg_walker.py` 的 `walk()`；
改公式後要重跑這一步。

### A3. Raw KG dump（**需要 Neo4j**）
```bash
DATASET=medbullets python3 pipeline/build_raw_hops.py       # → raw_1hop.json, raw_2hop.json
```

### A4. KG-RAG baselines（**ToG / HyKGE 需要 Neo4j**；MedRAG 不用）
```bash
DD=pipeline/datasets/medbullets

BENCH_PATH=$DD/benchmark.json SEEDS_PATH=$DD/seeds.json SYM_PATH=$DD/symptoms.json \
  TOG_LLM=gpt-5.4-mini KG_OUT_TAG=medbullets TOG_WORKERS=4 python3 pipeline/code/tog_baseline.py
DATASET=medbullets METHOD=tog SRC=cache/tog_baseline_medbullets__gpt-5.4-mini.json \
  python3 pipeline/freeze_baseline.py

BENCH_PATH=$DD/benchmark.json SEEDS_PATH=$DD/seeds.json SYM_PATH=$DD/symptoms.json \
  HYKGE_LLM=gpt-5.4-mini KG_OUT_TAG=medbullets HYKGE_WORKERS=2 python3 pipeline/code/hykge_baseline.py
DATASET=medbullets METHOD=hykge SRC=cache/hykge_baseline_medbullets__gpt-5.4-mini.json \
  python3 pipeline/freeze_baseline.py

BENCH_PATH=$DD/benchmark.json MEDRAG_OUT_TAG=medbullets python3 pipeline/code/run_medrag_textbook.py
DATASET=medbullets METHOD=medrag SRC=cache/medbullets_medrag_textbook_k32__gpt-5.4-mini.json \
  python3 pipeline/freeze_baseline.py
```

`KG_OUT_TAG` / `MEDRAG_OUT_TAG` **一定要跟 DATASET 一致**，否則會寫到 1273 的 cache 檔上。

---

## 階段 B — 讀取回答 → `results/<ds>_<method>_<model>.json`

**不需要 Neo4j。** 凍結完成後任何模型都能重播同一份 prompt。

```bash
for M in vanilla cot raw_1hop raw_2hop medrag tog hykge walker walker_interval; do
  DATASET=medbullets METHOD=$M MODEL=gpt-5.4-mini N_RUNS=3 WORKERS=10 \
    RESULTS_DIR=results/newdatasets python3 pipeline/run_reader.py
done
```

- reader 有隨機性（temperature 1.0）→ **N_RUNS≥3，回報 mean±std**。
- `RESULTS_DIR` 一定要設，否則會跟舊結果混在一起。
- 小模型 / 開源模型改用 `run_reader_block.py`（換成 ReinRAG `[Answer]` 格式、可續跑）。

---

## 完整最短路徑（MedBullets 為例）

```bash
# 0) tunnel
sshpass -p "$SSH_PASS" ssh -N -f -L 7687:localhost:7687 -L 4645:localhost:4645 \
  -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes -p "$SSH_PORT" "$SSH_USER@$SSH_HOST"

# 1) 準備輸入（LLM，約 924 次呼叫）
DATASET=medbullets WORKERS=8 python3 pipeline/prepare_dataset_inputs.py

# 2) 凍結各方法
DATASET=medbullets python3 pipeline/build_prompt_only.py
DATASET=medbullets WALKER_BC_MODE=overlap WALKER_METHOD_NAME=walker \
  WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=3 python3 pipeline/build_kg.py
DATASET=medbullets python3 pipeline/build_raw_hops.py

# 3) 評測
for M in vanilla cot raw_1hop raw_2hop walker; do
  DATASET=medbullets METHOD=$M MODEL=gpt-5.4-mini N_RUNS=3 WORKERS=10 \
    RESULTS_DIR=results/newdatasets python3 pipeline/run_reader.py
done

# 4) 收工
pkill -9 -f eval_kg_walker_full.py
```

---

## 這兩個資料集的特別注意事項

**1. MedBullets 有 38% 的題目需要看圖**（`Figure A` 的 ECG / 抹片 / 影像），純文字不可能答對。
   這會壓低所有方法的絕對分數，但**對所有方法一視同仁**，方法間比較仍然有效。
   報告時要標明，或另外切出無圖子集當敏感度分析：

```bash
python3 -c "
import json,re
b=json.load(open('pipeline/datasets/medbullets/benchmark.json'))
img=[x['uid'] for x in b if re.search(r'figure\s+[a-e]\b',x['question'],re.I)]
print(f'含圖 {len(img)}/{len(b)} = {100*len(img)/len(b):.1f}%')
json.dump(img, open('pipeline/datasets/medbullets/_image_uids.json','w'))"
```

**2. MMLU-Med 有一批研究設計 / 流行病學 / 生理實驗題**（不是病人 vignette），
   對它們做 KG 檢索沒有意義。不必事先剔除——`no_duration` 的判讀記錄已經標出來了，
   分析時可用 `verification/manual_read_mmlu.jsonl` 切子集。

**3. 兩者都沒有 train split**，所以沒有 MedQA 329 那種 73% 來自 train 的汙染疑慮：
   MedBullets 308 題就是整個資料集，MMLU 用的是官方 test（272）。這點在論文裡是加分項。

**4. 選項數不同**：MedBullets 5 選項、MMLU 4 選項、MedQA 5–12 選項。
   隨機猜測基準分別是 20% / 25%，跨資料集比較絕對分數時要換算。

---

## 需要 Neo4j 的步驟一覽

| 步驟 | Neo4j |
|---|---|
| 階段 0 準備輸入 | 否 |
| A1 `build_prompt_only.py` | 否 |
| A2 `build_kg.py`（walker） | **是** |
| A3 `build_raw_hops.py` | **是** |
| A4 ToG / HyKGE | **是** |
| A4 MedRAG | 否 |
| 階段 B `run_reader.py` | 否 |

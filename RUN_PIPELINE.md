# Running the full pipeline (incl. Neo4j re-retrieval) — runbook

Two phases: **build/freeze** each method's per-question prompt+KG (retrieval; needs Neo4j),
then **read** (answer; needs only an LLM API). Retrieval is done once and frozen; the reader
replays it. `DATASET` is `1273` or `329` throughout. Run everything from the repo root
(`new_duration_spectrum/`), not from `pipeline/`.

---

## 0. Prerequisites

**Neo4j** (only needed for the retrieval builds — steps 2–4). It lives on the msi-gpu host; open
the tunnel first:

> **憑證**：SSH 與 Neo4j 的帳密不寫在這份文件裡。設定 `SSH_HOST` / `SSH_PORT` / `SSH_USER` /
> `SSH_PASS` / `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` 環境變數後再執行以下指令；
> 實際值放在未進版控的 `CLAUDE.local.md`。

```bash
sshpass -p "$SSH_PASS" ssh -N -f -L 7687:localhost:7687 -L 4645:localhost:4645 \
  -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 -p "$SSH_PORT" "$SSH_USER@$SSH_HOST"
# verify: should return 443699
python3 -c "import sys;sys.path.insert(0,'pipeline/code');sys.path.insert(0,'pipeline/code/dkr_policy');\
from umls_neo4j import get_driver;\
print(get_driver().session().run('MATCH (c:Concept) RETURN count(c) AS n').single()['n'])"
```
**API keys** are in `pipeline/api_key/`. **Env** — the SSH-200 venv, or any Python with
`torch transformers neo4j numpy openai`.

⚠ **Memory**: each retrieval worker holds ~4 GB (SapBERT matrix). `WALKER_N_WORKERS=3` is safe on
38 GB; 6 caused an OOM reboot. After killing any build, also `pkill -9 -f eval_kg_walker_full.py`
(the OUT_TAG is an env var, so pgrep on it misses the workers).

---

## PHASE A — build & freeze every method → `frozen/<ds>/<method>.json`

### 1. Prompt-only methods — NO Neo4j
```bash
DATASET=1273 python3 pipeline/build_prompt_only.py       # → vanilla.json, cot.json
```

### 2. OUR METHOD (walker) + its ablation — needs Neo4j
```bash
# overlap = our method (bc = Bhattacharyya). writes frozen/1273/walker.json
DATASET=1273 WALKER_BC_MODE=overlap         WALKER_METHOD_NAME=walker \
  WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=3 python3 pipeline/build_kg.py

# interval ablation (bc = point sample). writes frozen/1273/walker_interval.json
DATASET=1273 WALKER_BC_MODE=interval_sample WALKER_METHOD_NAME=walker_interval \
  WALKER_RETRIEVAL_ONLY=1 WALKER_N_WORKERS=3 python3 pipeline/build_kg.py
```
Checkpointed every 25 Qs, **resumable** (re-run to continue). Each progress line prints
`LLM-gen N (N/q)` = duration-generation cost; **watch it — >40/q means the cache isn't being
reused** (path bug or orphaned worker). To change the score, edit `code/dkr_policy/kg_walker.py`
`walk()` first, then re-run this.

### 3. Raw KG dumps — needs Neo4j
```bash
DATASET=1273 python3 pipeline/build_raw_hops.py          # → raw_1hop.json, raw_2hop.json
```

### 4. KG-RAG baselines — needs Neo4j + LLM (each writes to cache/, then freeze)
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

# MedRAG (textbook; no Neo4j, but LLM)
BENCH_PATH=$DD/benchmark.json MEDRAG_OUT_TAG=1273 python3 pipeline/code/run_medrag_textbook.py
DATASET=1273 METHOD=medrag SRC=cache/1273_medrag_textbook_k32__gpt-5.4-mini.json python3 pipeline/freeze_baseline.py
```
ToG/HyKGE settings are the paper defaults (`TOG_WIDTH/DEPTH=3`, `TOG_REL_THRESH=0.2`,
`TOG_ENTITY_PRUNE=llm`; `HYKGE_HOP=3 HYKGE_TOPK=10`). See the fidelity notes atop each file.

---

## PHASE B — read (answer) → `results/<ds>_<method>_<model>.json`

No Neo4j. Once frozen, point any model at any method:
```bash
for M in vanilla cot raw_1hop raw_2hop medrag tog hykge walker walker_interval; do
  DATASET=1273 METHOD=$M MODEL=gpt-5.4-mini N_RUNS=3 WORKERS=10 \
    RESULTS_DIR=results/round2_intentfree python3 pipeline/run_reader.py
done
```
- `MODEL` routes by name (`code/llm_client.py`): `gpt-*`→OpenAI, `gemini-3*`/`gemma*`→litellm,
  `google:<m>`→Gemini, `together:<m>`→Together, `vllm:<m>`/`qwen*`→vLLM.
- Reader is stochastic (temp 1.0) → **N_RUNS≥3, report mean±std**. Output stores prompt + raw
  output per question per run.
- **Small / open models** (Qwen…): use `run_reader_block.py` instead (ReinRAG `[Answer]` format,
  resumable, records `n_unparseable`; on Together keep `WORKERS≈3` — high unparseable = throttling).

---

## Which builder makes which frozen file

| frozen file | built by | Neo4j? |
|---|---|---|
| `vanilla.json`, `cot.json` | `build_prompt_only.py` | no |
| `walker.json` | `build_kg.py` (BC_MODE=overlap) | **yes** |
| `walker_interval.json` | `build_kg.py` (BC_MODE=interval_sample) | **yes** |
| `raw_1hop.json`, `raw_2hop.json` | `build_raw_hops.py` | **yes** |
| `tog.json` | `tog_baseline.py` → `freeze_baseline.py` | **yes** |
| `hykge.json` | `hykge_baseline.py` → `freeze_baseline.py` | **yes** |
| `medrag.json` | `run_medrag_textbook.py` → `freeze_baseline.py` | no |

To reproduce the whole set for 329, run every phase-A step with `DATASET=329` (and
`KG_OUT_TAG=329` / `MEDRAG_OUT_TAG=329`), then phase B with `DATASET=329`.
Note: **329 is a separate set, not a subset of 1273** — it must be built and read independently.

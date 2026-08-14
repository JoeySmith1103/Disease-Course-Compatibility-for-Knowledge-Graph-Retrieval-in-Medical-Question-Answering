# Duration-Guided KG Retrieval Pipeline

這份 repo 是「用病程時間輔助 KG retrieval，再讓 reader LLM 回答醫學選擇題」的實驗 pipeline。所有指令都假設你站在 **repo 根目錄** 執行，也就是這個 `README.md` 所在的資料夾；這份 GitHub clone 已經不是外層專案裡的 `pipeline/` 子資料夾，所以指令不要再加 `pipeline/` 前綴。

核心切成兩階段：

```text
Stage A retrieval/freeze：datasets/<ds>/benchmark.json + durations/seeds/symptoms/query_entities
  -> Neo4j / baseline retrieval
  -> frozen/<ds>/<method>.json

Stage B reader：frozen/<ds>/<method>.json
  -> run_reader.py 或 run_reader_vllm_batch.py
  -> results/<...>.json
```

重點是 frozen file 內同時存 `kg_block` 和 `prompt`。所以 prompt tuning 可以只重新 render wording，不必重跑 Neo4j retrieval；這也是 `__revised` prompt A/B 有意義的原因。

## Environment

建議一律用 venv：

```bash
cd /home/jjouyang/Disease-Course-Compatibility-for-Knowledge-Graph-Retrieval-in-Medical-Question-Answering
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

基本 reader / report：

```bash
python3 -m pip install openai requests tqdm scipy pandas numpy
```

需要 Stage A retrieval 時再補 Neo4j / embedding 相關套件：

```bash
python3 -m pip install neo4j torch transformers sentence-transformers scikit-learn
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=...
export NEO4J_PASSWORD=...
```

需要本機 batch inference 時再裝 vLLM：

```bash
python3 -m pip install vllm
```

API key 不在 git 內。`code/llm_client.py` 會先讀環境變數，再 fallback 到 gitignored 的 `api_key/`；交接或 code review 時不要把 `api_key/` 當成 source code 內容。

## Current Datasets

| dataset | n | options | status | notes |
| --- | ---: | --- | --- | --- |
| `1273` | 1273 | A-E | ready | MedQA test 全集；已有 core frozen methods including `medrag` / `walker_interval` |
| `329` | 329 | A-L | ready | 從整個 MedQA 篩出的 duration-critical 分層；和 1273 testing set 只有少量重疊；parser 必須支援到 L |
| `medbullets` | 308 | A-E | ready except MedRAG | 新增 dataset；`temporal_critical.json` 目前 88 題 critical；已有 KG/no-KG frozen 與 `__revised` |
| `mmlu` | 272 | A-D | ready except MedRAG | MMLU professional medicine；`temporal_critical.json` 目前 51 題 critical；已有 KG/no-KG frozen 與 `__revised` |
| `medmcqa` | 143 | A-D | dataset only | 新增 dataset；目前只有 benchmark/durations/temporal labels，尚未有 seeds/symptoms/query_entities/frozen |

`datasets/_medmcqa_OLD_inherited_filter/` 是舊的 inherited filter 版本，現在不要拿來當主結果。

`datasets/<ds>/temporal_critical.json` 是人工判讀後的 temporal-critical label；`verification/verified_subsets.json` 是 counterfactual/open-ended verification 實驗輸出，兩者不是同一種標籤，不要混在同一個表格解讀。

## Methods

| method | what it is | Stage A dependency |
| --- | --- | --- |
| `vanilla` | no KG, direct answer | none |
| `cot` | no KG, minimal chain-of-thought | none |
| `raw_1hop` | seed concepts 的 raw 1-hop KG neighbors | Neo4j |
| `raw_2hop` | seed concepts 的 raw 2-hop KG neighbors | Neo4j |
| `tog` | Think-on-Graph baseline | Neo4j + LLM |
| `hykge` | HyKGE-style hypothetical KG expansion | Neo4j + LLM |
| `medrag` | textbook BM25/RAG baseline | local textbook cache; currently only ready for 1273/329 |
| `walker` | ours, `score = cos + 0.3*bc - 0.08*hop` | Neo4j + LLM duration/KG cache |
| `walker_interval` | walker ablation, bc from interval sampling instead of overlap | Neo4j + LLM duration/KG cache |

MedRAG 對 `medbullets` / `mmlu` 目前先視為 on hold：`run_stage_a.sh` 已把 MedRAG retrieval 註解掉，因為 corpus provenance、舊 import path 和 duration path 還沒整理好。新 dataset 的 full comparison 不應假裝有 MedRAG cell。

## Prompt Naming

目前統一只保留三個 walker prompt 名稱：

| name | file / usage | meaning |
| --- | --- | --- |
| original prompt | `prompts.WALKER`, `prompts.WALKER_ORIGINAL`, or frozen `<method>.json` | 最一開始學長的 prompt，使用 `<a>X</a>` answer format |
| prompt v1 | `prompts.WALKER_V1` | 使用者貼的較長 structured prompt，含 `[QUESTION]` / `[OPTIONS]` / `[SYMPTOM_DURATION]` / `[RETRIEVED INFORMATION]` |
| prompt v2 | `prompts.WALKER_V2`, `prompt_revised.txt`, frozen `<method>__revised.json` | 我們後來微調後的短 reasoning prompt；之後看到 `__revised` 就當作 prompt v2 |

`__revised` 只應用在 KG-bearing methods：`walker`, `walker_interval`, `raw_1hop`, `raw_2hop`, `tog`, `hykge`。`vanilla` / `cot` 沒有 KG block，不要硬套 KG prompt v2。

### Frozen prompt vs `prompts.py`

Reader 實際送給模型的 prompt 通常已經存在 `frozen/<dataset>/<method>.json` 的每個 `item["prompt"]` 裡。也就是說，`walker__revised.json` 的 `item["prompt"]` 本身就已經是 prompt v2；如果 runner 使用 `PROMPT_VARIANT=frozen_legacy`，就會直接送這段 frozen prompt，不再套 `prompts.py` 的 template。

`prompts.py` 只有在兩種情況會影響 reader prompt：第一是用 `rerender_prompts.py` 重新產生 `__revised` frozen files；第二是 runner 指定 runtime prompt surgery，例如 `small_model_block`、`walker_v1`、`walker_v2`。特別注意：如果拿 `walker__revised` 再搭配 `PROMPT_VARIANT=small_model_block`，實際 prompt 會變成「frozen 裡的 v2 prompt」再額外 append `[Output Format]` 區塊，這不是乾淨的 v2 prompt。

重新 render prompt v2，不重跑 retrieval：

```bash
bash rerender_all_kg.sh 329 revised prompt_revised.txt
bash rerender_all_kg.sh 1273 revised prompt_revised.txt
```

或只 render 單一 method：

```bash
DATASET=329 METHOD=walker VARIANT=revised TEMPLATE=WALKER_V2 python3 rerender_prompts.py
```

## Run Existing Frozen Files

API reader，適合 gpt/gemini/together endpoint：

```bash
DATASET=329 METHOD=walker MODEL=gpt-5.4-mini N_RUNS=3 WORKERS=12 \
  RESULTS_DIR=results/old_prompt python3 run_reader.py

DATASET=329 METHOD=walker__revised MODEL=gpt-5.4-mini N_RUNS=3 WORKERS=12 \
  RESULTS_DIR=results/revised_prompt python3 run_reader.py
```

Small/medium open models 如果常出現格式錯誤，可以用 block prompt reader：

```bash
DATASET=329 RUN=1 MODEL=together:Qwen/Qwen3.5-9B WORKERS=6 \
  METHODS=vanilla,cot,raw_1hop,raw_2hop,tog,hykge,walker,walker_interval \
  RESULTS_DIR=results/qwen_block python3 run_reader_block.py
```

本機 vLLM offline batch，適合 Qwen / GPT-OSS 這類模型。`BATCH_SIZE` 是一次送進 vLLM 的 prompt 數；`VLLM_MAX_NUM_SEQS` 沒設時預設等於 `BATCH_SIZE`。

```bash
DATASETS=329,1273 \
METHODS=walker,walker_interval \
MODEL=openai/gpt-oss-20b \
RUNS=1 BATCH_SIZE=64 PROMPT_VARIANTS=frozen_legacy \
VLLM_REASONING_EFFORT=low VLLM_MAX_TOKENS=512 \
RESULTS_DIR=results/vllm_gptoss_original \
python3 run_reader_vllm_batch.py
```

Qwen 範例：

```bash
DATASETS=329,1273 \
METHODS=vanilla,cot,raw_1hop,raw_2hop,tog,hykge,walker,walker_interval \
MODEL=Qwen/Qwen3.5-9B \
RUNS=1 BATCH_SIZE=32 PROMPT_VARIANTS=small_model_block \
RESULTS_DIR=results/vllm_qwen_block \
python3 run_reader_vllm_batch.py
```

vLLM runner 的 `PROMPT_VARIANTS` 可用：

| value | meaning |
| --- | --- |
| `frozen_legacy` / `original` / `legacy` | 直接使用 frozen file 裡存好的 prompt；如果 method 是 `walker__revised`，這就等於 prompt v2 |
| `small_model_block` | 對 frozen prompt 做 runtime 加工並 append `[Output Format]`；可降低 unparseable，但若 frozen file 已是 `__revised`，實際 prompt 會是 v2 + 額外 output instruction，不是乾淨 v2 |
| `walker_v1` / `prompt_v1` | runtime 套 `prompts.WALKER_V1`；只適用 `walker` / `walker_interval` |
| `walker_v2` / `prompt_v2` / `prompt_ver3` | runtime 套 `prompts.WALKER_V2`；只適用 `walker` / `walker_interval` |

建議比較乾淨 prompt v2 時優先用 frozen `__revised` 檔，並搭配 `PROMPT_VARIANTS=frozen_legacy`：

```bash
DATASETS=329,1273 METHODS=walker__revised,walker_interval__revised \
MODEL=openai/gpt-oss-20b RUNS=1 BATCH_SIZE=64 PROMPT_VARIANTS=frozen_legacy \
RESULTS_DIR=results/vllm_gptoss_v2 python3 run_reader_vllm_batch.py
```

## Stage A Retrieval

如果 dataset 已經有 frozen files，通常不需要跑 Stage A。真的要重建 retrieval 時才跑：

```bash
bash run_stage_a.sh medbullets
bash run_stage_a.sh mmlu
```

Stage A 是刻意 serial 執行；walker worker 會吃大量 SapBERT matrix memory，同時跑多個 retrieval job 容易把 38GB RAM 機器打滿。

新 dataset 若還缺 `seeds.json` / `symptoms.json` / `query_entities.json`，先跑：

```bash
DATASET=medmcqa MODEL=gpt-5.4-mini WORKERS=8 python3 prepare_dataset_inputs.py
```

## Metrics

`metrics.py` 的定義是目前統一算法：

| metric | definition |
| --- | --- |
| accuracy | `correct / n`；unparseable 算錯 |
| micro recall | 單標籤 multiple-choice 下等於 accuracy |
| parseable precision / micro precision | `correct / (n - unparseable)` |
| macro precision | 對固定 answer letters 做 `tp/(tp+fp)` 後平均，zero division = 0 |
| macro recall | 對固定 answer letters 做 `tp/(tp+fn)` 後平均，也就是 balanced accuracy |
| weighted precision/recall/F1 | 依 gold support 加權；329 類別極不均時比 macro 穩 |
| unparseable | parser 無法抽出有效選項字母的筆數；仍計入 n 且算錯 |

注意：這裡的 precision/recall 是「答案字母」層級，不是診斷概念層級，也不是 retrieval recall。不要和 KG retrieval recall 放在同一個欄位下解讀。

產生 report：

```bash
RESULTS_DIR=results/vllm_gptoss_original python3 metrics_report.py 329 1273
RESULTS_DIR=results/revised_prompt python3 metrics_report.py 329 medbullets mmlu
```

一般 `metrics_report.py` 會使用 result JSON 裡已存好的 `predicted`。如果 parser 修過、想從 `raw_response` 重新抽答案，請加 `REPARSE=1`：`REPARSE=1 RESULTS_DIR=results/old_prompt python3 metrics_report.py 329`。

## What To Test Now

目前最符合這份 repo 的 workload：

| dataset | methods | prompt |
| --- | --- | --- |
| `329`, `1273` | all core methods including `medrag` | original prompt for historical comparison |
| `329`, `1273` | `walker`, `walker_interval` | prompt v2 / `__revised` for current prompt setting |
| `medbullets`, `mmlu` | `vanilla`, `cot`, `raw_1hop`, `raw_2hop`, `tog`, `hykge`, `walker`, `walker_interval` | no-KG controls use original; KG methods can use original and `__revised` |
| `medmcqa` | not ready for full reader comparison | build inputs + Stage A first; no frozen methods currently |

老師若只說「把 baseline 和 method 放到小模型跑」，最小可交付就是用 vLLM batch 跑上表可用的 frozen methods，並輸出 `acc`, `macro precision`, `macro recall`, `unparseable`。對新 dataset 不要填 MedRAG，除非先把 MedRAG pipeline 修好並明確寫 corpus choice。

## Files

| file | role |
| --- | --- |
| `prompts.py` | 所有 canonical prompts；`WALKER_V1/WALKER_V2` 也在這裡 |
| `prompt_revised.txt` | prompt v2 的文字檔版本，供 `rerender_prompts.py TEMPLATE_FILE=...` 使用 |
| `rerender_prompts.py` | 用同一份 frozen evidence 重建 prompt wording，不跑 Neo4j/LLM |
| `run_reader.py` | API-based reader，支援 N runs 與 metrics |
| `run_reader_block.py` | small-model API reader，把輸出格式改為 `[Reasoning]` / `[Answer]` |
| `run_reader_vllm_batch.py` | 本機 vLLM offline batch reader |
| `answer_extract.py` | 共用 parser，支援 A-L 並用每題 options 過濾無效字母 |
| `metrics.py` | accuracy / precision / recall / F1 定義 |
| `metrics_report.py` | 從 results JSON 畫 aggregated metrics 表 |
| `build_new_dataset.py` | 建立 medbullets / mmlu / medmcqa benchmark |
| `prepare_dataset_inputs.py` | 產生 seeds / symptoms / query_entities / durations |
| `run_stage_a.sh` | 對單一 dataset sequential 跑 retrieval/freeze |

## Known Caveats

- `__revised` 是 prompt v2，不是新的 retrieval；它和原 method 共用同一份 `kg_block` evidence。
- ToG/HyKGE 的 frozen prompt 與 `kg_block` 格式不一定是逐字 substring，所以 reader canary 對它們用 method-level exception，避免誤判 evidence missing。
- 329 有 A-K/L option letters；舊的 A-J parser 會把 K/L 永遠算 unparseable，現在已修成 A-L 並以每題 options 過濾。
- `medbullets` / `mmlu` 目前沒有 MedRAG frozen file；比較表中 MedRAG 應標成 N/A。
- `medmcqa` 目前是 dataset-only 狀態，不能直接跑 reader；缺 frozen retrieval artifacts。
- `results/`、`logs/`、`cache/` 通常不進 git。要交接結果時請交 result JSON 或 markdown report，不要只貼 console 最後一行。

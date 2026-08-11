# Evaluation 檔案與使用說明

---

## 1. 三個評測程式

| 檔案 | 行數 | 用途 | 對象模型 |
|---|---|---|---|
| **`run_reader.py`** | 74 | 主評測器：跑 N 次、輸出 mean±std + 完整逐題記錄 | 大模型（gpt / gemini） |
| **`run_reader_block.py`** | 116 | 同上，但把答案格式換成 ReinRAG `[Answer]` 區塊 | 小模型 / 開源模型（Qwen 等） |
| **`rerender_prompts.py`** | 100 | 換 prompt 措辭、沿用同一份 KG（不呼叫 LLM、不碰 Neo4j） | — |

三者都**不做 retrieval** —— 只讀 `frozen/<ds>/<method>.json`，證據已經烘焙在裡面。

---

## 2. 資料流

```
frozen/<ds>/<method>.json          ← 輸入（每題已填好的 prompt + kg_block）
   {uid, gold, route, kg_block, prompt}
            │
            ├── run_reader.py         （大模型，N 次）
            └── run_reader_block.py   （小模型，換答案格式）
            │
            ▼
results/<RESULTS_DIR>/<ds>_<method>_<model>.json    ← 輸出
   {n, runs_correct, mean_acc, std_acc, runs:[{run, results:[逐題...]}]}
```

---

## 3. `run_reader.py` — 主評測器

### 用法
```bash
DATASET=329 METHOD=walker MODEL=gpt-5.4-mini N_RUNS=5 WORKERS=10 \
  RESULTS_DIR=results/round2_intentfree python3 pipeline/run_reader.py
```

### 環境變數

| 變數 | 預設 | 說明 |
|---|---|---|
| `DATASET` | `329` | `329` 或 `1273` |
| `METHOD` | `walker` | 對應 `frozen/<ds>/<METHOD>.json`；可用 `vanilla cot raw_1hop raw_2hop medrag tog hykge walker walker_interval`，或 rerender 出來的 `walker__myv1` |
| `MODEL` | `gpt-5.4-mini` | 依名稱路由（見 §6） |
| `N_RUNS` | `1` | **建議 ≥3**；reader 有隨機性（temperature 1.0） |
| `WORKERS` | `12` | 並行請求數 |
| `RESULTS_DIR` | `results` | 輸出目錄；用它把不同輪次分開，不要蓋到舊結果 |

### 每題做的事
1. `prompt = prompts.to_big_model_prompt(frozen_prompt)` —— 移除已失效的 path legend
2. **canary**：assert `kg_block` 的每一行內容都出現在 prompt 裡（曾有 bug 讓證據整批消失卻沒人發現）
3. 呼叫 LLM → `extract_letter()` 擷取答案 → 與 gold 比對

### 答案擷取順序
```
<a>X</a>  →  \boxed{X}  →  **Answer: X**  →  final answer: X  →  answer is X  →  the answer is X
```
擷取不到回傳 `None`，一律計為答錯（會顯示在 `unparseable`）。

---

## 4. `run_reader_block.py` — 小模型用

### 用法
```bash
VLLM_BASE_URL=http://localhost:8000/v1 \
DATASET=1273 MODEL=vllm:Qwen3.5-9B METHODS=walker,cot RUN=1 WORKERS=8 \
  python3 pipeline/run_reader_block.py
```

| 變數 | 預設 | 說明 |
|---|---|---|
| `METHODS` | 六個方法 | **逗號分隔可一次跑多個**（與 run_reader 的單一 `METHOD` 不同） |
| `RUN` | `1` | 輪次編號，寫進檔名 |
| 其餘 | — | 同 run_reader |

### 與 run_reader 的差別
送的是**同一份 frozen prompt**，但 `prompts.to_small_model_prompt()` 會：
1. 移除 `<a></a>` 那句答案指示
2. 移除已失效的 path legend
3. 接上 ReinRAG 的 `[Output Format] / [Reasoning] / [Answer]` 區塊

擷取順序改為 `[Answer] X` 優先。**可續跑**（重跑會跳過已完成的 uid），並記錄 `n_unparseable`。

⚠ 小模型的 `n_unparseable` 偏高通常是**被限流**（Together 有 dynamic rate limit），不是模型變差。

---

## 5. `rerender_prompts.py` — 只換 prompt、不重跑 retrieval

```bash
DATASET=1273 METHOD=walker VARIANT=myv1 TEMPLATE_FILE=my_prompt.txt \
  python3 pipeline/rerender_prompts.py        # → frozen/1273/walker__myv1.json
DATASET=1273 METHOD=walker__myv1 MODEL=vllm:Qwen3.5-9B N_RUNS=3 python3 pipeline/run_reader.py
```
可用 placeholder：`{question}` `{options_block}` `{kg_block}` `{patient_dur_str}`。
省略 `{kg_block}` 就是「同樣題目、不給證據」的對照組。
輸出會印 `kg_block injected: N/總數` 確認證據有進去。

---

## 6. `MODEL` 路由（`code/llm_client.py`）

| 前綴 | 走哪裡 |
|---|---|
| `gpt-*`, `o*` | OpenAI |
| `gemini-3*`, `gemini-2.5-flash`, `gemma*`, `gpt-oss*` | NetDB litellm proxy |
| `google:<m>` | Google AI Studio 直連 |
| `together:<m>` | Together AI |
| `vllm:<m>`, `qwen*` | 本地 vLLM（讀 `VLLM_BASE_URL`，預設 `http://localhost:8000/v1`） |
| 其他 | Gemini 直連 |

---

## 7. 輸出檔結構

`results/round2_intentfree/329_walker_gpt-5.4-mini.json`：

```jsonc
{
  "dataset": "329", "method": "walker", "model": "gpt-5.4-mini", "n": 329,
  "runs_correct": [277, 275, 278, 277, 283],   // 每輪正確題數
  "mean_correct": 278.0, "std_correct": 2.68,
  "mean_acc": 84.50, "std_acc": 0.82,
  "runs": [                                     // 完整逐題記錄（每輪一份）
    { "run": 1, "results": [
        { "uid": "aud_002", "gold": "C", "predicted": "C", "is_correct": true,
          "route": "walker_kg",
          "kg_block": "...(749 chars)",         // 送進去的證據
          "prompt":   "...(2975 chars)",        // 實際送出的完整 prompt
          "raw_response": "...(969 chars)" }    // 模型的原始輸出
      ] }
  ]
}
```

**每題都存 prompt 與原始輸出** —— 所以任何數字都可以事後稽核，不用重跑。

### 目前的評測輸出（10 個）
```
results/round2_intentfree/
  1273_{walker, walker_interval, raw_1hop, raw_2hop, hykge}_gpt-5.4-mini.json
   329_{walker, walker_interval, raw_1hop, raw_2hop, hykge}_gpt-5.4-mini.json
```
其餘方法（vanilla / cot / medrag / tog）沿用舊結果，見 EXPERIMENT_LOG_zh.md §1(c)、§10.3。

---

## 8. 常用分析片段

```python
import json, statistics
d = json.load(open('pipeline/results/round2_intentfree/329_walker_gpt-5.4-mini.json'))

# 1) 逐輪題數
print(d['runs_correct'], d['mean_acc'], d['std_acc'])

# 2) 逐題多數決（去掉單次隨機性）
runs = [{r['uid']: r['is_correct'] for r in run['results']} for run in d['runs']]
uids = sorted(runs[0])
maj  = {u: sum(1 for r in runs if r[u]) >= len(runs)//2+1 for u in uids}
print(sum(maj.values()), '/', len(uids))

# 3) 依 route 分層
from collections import defaultdict
by = defaultdict(list)
for r in d['runs'][0]['results']: by[r['route']].append(r['is_correct'])
for k, v in by.items(): print(k, f"{100*sum(v)/len(v):.2f}%  n={len(v)}")

# 4) 看某題實際送出的 prompt 與模型輸出
r = next(x for x in d['runs'][0]['results'] if x['uid']=='aud_002')
print(r['prompt']); print('---'); print(r['raw_response'])

# 5) 兩方法配對比較（同題）
def perq(m, ds='329'):
    dd = json.load(open(f'pipeline/results/round2_intentfree/{ds}_{m}_gpt-5.4-mini.json'))
    rs = [{r['uid']: r['is_correct'] for r in run['results']} for run in dd['runs']]
    return {u: sum(1 for r in rs if r[u])/len(rs) for u in rs[0]}
A, B = perq('walker'), perq('walker_interval')
from scipy.stats import wilcoxon
u = sorted(A); print(wilcoxon([A[x] for x in u], [B[x] for x in u]).pvalue)
```

---

## 9. 注意事項

1. **reader 有隨機性**（temperature 1.0）→ `N_RUNS≥3`，回報 mean±std。單次數字可差數 pp。
2. **`RESULTS_DIR` 一定要設**，否則會寫進 `results/` 根目錄與舊結果混在一起。
3. **不需要 Neo4j** —— 評測階段只讀 frozen 檔 + 呼叫 LLM。
4. **canary 會擋下證據遺失**：若 `kg_block` 的內容沒出現在 prompt 裡會直接 assert 失敗，
   不會安靜地跑出一個沒有證據的分數。
5. **`unparseable` 要看**：偏高代表擷取失敗（或被限流），那些題一律算錯，會壓低 accuracy。
6. 換 prompt 用 `rerender_prompts.py`，**不要**重跑 `build_kg.py`（那是 retrieval，要 Neo4j 且慢）。

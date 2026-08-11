# 各個 prompt 放在哪裡（索引）

分三層：**模板**（固定文字，要改就改這裡）、**凍結檔**（實際送出去的、每題填好的 prompt）、
以及**渲染好的範例**。

- 模板 → `pipeline/prompts.py`（reader prompt 的唯一來源）
- 凍結後的實際 prompt → `pipeline/frozen/<ds>/<method>.json`，每題的 `"prompt"` 欄位
- 每個方法各一個範例（同一題貫穿全部方法）→ `pipeline/PROMPTS_1273.md`

---

## A. Reader / 回答用 prompt（回答模型真正看到的內容，也就是被凍結的東西）


| 方法                              | 模板                                   | 行號                        | 凍結檔（`frozen/<ds>/…`）                 |
| ------------------------------- | ------------------------------------ | ------------------------- | ------------------------------------ |
| vanilla                         | `prompts.VANILLA`                    | 7                         | `vanilla.json`                       |
| cot                             | `prompts.COT_MINIMAL`                | 17                        | `cot.json`                           |
| raw_1hop / raw_2hop             | `prompts.RAW_KG`                     | 25                        | `raw_1hop.json` / `raw_2hop.json`    |
| medrag                          | `prompts.MEDRAG`                     | 46                        | `medrag.json`                        |
| **walker（我們的方法）**               | `prompts.WALKER`                     | 68                        | `walker.json`                        |
| walker_interval                 | `prompts.WALKER`（同一份）                | 68                        | `walker_interval.json`               |
| 無 duration / 無 seeds 的 fallback | `prompts.NO_KG`                      | 99                        | （在 walker.json 內，route≠walker_kg 的題） |
| tog                             | `tog_baseline.ANSWER`                | code/tog_baseline.py:71   | `tog.json`                           |
| hykge                           | `hykge_baseline.ANSWER`（論文 P_Reader） | code/hykge_baseline.py:94 | `hykge.json`                         |


`prompts.WALKER` 內同時包含「病人 duration 那一行」與 `cos / bc` 的說明圖例。
當某題沒有可用的 duration 時，reader prompt 會用 `prompts.NO_DURATION_STR`（第 149 行）
取代 duration 文字，且 bc 全部為 0。

**重點**：walker 與 walker_interval 用的是**同一份模板**，
差別只在 kg_block 裡的 bc 數值與候選排序 —— 這正是 ablation 該有的乾淨對照。

## B. Retrieval 期間的 prompt（build 過程中呼叫 LLM，不會被凍結）


| 使用者                         | prompt 名稱                  | 位置                                         |
| --------------------------- | -------------------------- | ------------------------------------------ |
| tog — relation prune        | `REL_PRUNE`                | code/tog_baseline.py:49                    |
| tog — entity prune          | `ENT_PRUNE`                | code/tog_baseline.py:57                    |
| tog — 充分性判斷                 | `SUFF`                     | code/tog_baseline.py:65                    |
| hykge — hypothesis output   | `HO_PROMPT`（論文 P_HO）       | code/hykge_baseline.py:63                  |
| walker — on-demand duration | `ROLE_PROMPTS`             | code/dkr_policy/bc_ondemand.py:39          |
| seeds（每題）                   | `PROMPT`                   | code/extract_multitype_seeds.py:17         |
| 病人 duration 抽取              | `_build_extraction_prompt` | scripts/extract_patient_duration_llm.py:68 |


*(*`prompts.py` *裡另有參考用的副本* `TOG_REL_PRUNE`*(116) 與* `HYKGE_HYPOTHESIS`*(124)，
但 baseline 實際使用的是上表這些檔案內的版本。)*

## C. 小模型 / 開源模型的答案格式

`run_reader_block.py` 送的是**同一份凍結 prompt**，只換掉答案指示：
把 `<a></a>` 那句移除，改接 `prompts.REINRAG_OUTPUT`（第 136 行），
也就是 `[Reasoning]` / `[Answer]` 區塊。
轉換函式：`prompts.to_big_model_prompt()` / `to_small_model_prompt()`。

---



## 快速查看某個 prompt

```bash
# 看固定模板
sed -n '68,97p' pipeline/prompts.py                       # 例：WALKER

# 看某一題實際填好的 prompt
python3 -c "import json; d=json.load(open('pipeline/frozen/1273/walker.json'))['items']; \
print(next(i['prompt'] for i in d if i['uid']=='test_0001'))"

# 所有方法、同一題，並排比較
open pipeline/PROMPTS_1273.md
```

---



## 兩個資料集目前的狀態


|                                               | 329         | 1273        |
| --------------------------------------------- | ----------- | ----------- |
| vanilla, cot, raw_1hop, raw_2hop, medrag, tog | ✅           | ✅           |
| hykge                                         | ✅（chains 版） | ✅（chains 版） |
| **walker（我們的方法）**                             | ✅           | ✅           |
| **walker_interval**                           | ❌ **未建立**   | ✅           |


`frozen/1273/` 九個方法全部齊全且是最新版（intent-free、no-duration 已改為走 KG、
path legend 已移除）。
`frozen/329/` 有 7 個；缺的兩個就是 walker 與 walker_interval，
需要跑 RUN_PIPELINE_zh.md 的步驟 2、把 `DATASET` 換成 `329`（約各 30–40 分鐘，
因為 duration cache 的 bug 已修好、cache 是熱的）。
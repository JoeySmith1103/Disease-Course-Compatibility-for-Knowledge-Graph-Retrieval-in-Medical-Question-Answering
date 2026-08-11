# Where every usable prompt lives

Three layers: the **template** (the fixed wording, edit here), the **frozen** per-question files
(the concrete filled prompts actually sent), and a **rendered example** of each.

- Templates → `pipeline/prompts.py` (single source of truth for reader prompts)
- Frozen filled prompts → `pipeline/frozen/<ds>/<method>.json`, field `"prompt"` per question
- One rendered example of every method (same question) → `pipeline/PROMPTS_1273.md`

---

## A. Reader / answer prompts (what the answering model sees — these are what's frozen)

| method | template | line | frozen file (`frozen/<ds>/…`) |
|---|---|---|---|
| vanilla | `prompts.VANILLA` | 7 | `vanilla.json` |
| cot | `prompts.COT_MINIMAL` | 17 | `cot.json` |
| raw_1hop / raw_2hop | `prompts.RAW_KG` | 25 | `raw_1hop.json` / `raw_2hop.json` |
| medrag | `prompts.MEDRAG` | 46 | `medrag.json` |
| **walker (ours)** | `prompts.WALKER` | 68 | `walker.json` |
| walker_interval | `prompts.WALKER` (same) | 68 | `walker_interval.json` |
| no-duration / no-seeds fallback | `prompts.NO_KG` | 99 | (inside walker.json, route≠walker_kg) |
| tog | `tog_baseline.ANSWER` | code/tog_baseline.py:71 | `tog.json` |
| hykge | `hykge_baseline.ANSWER` (paper P_Reader) | code/hykge_baseline.py:94 | `hykge.json` |

`prompts.WALKER` also holds the duration line + the `cos/bc` legend. When a walker question has no
usable duration, the reader prompt substitutes `prompts.NO_DURATION_STR` (line 149) for the
duration and bc stays 0.

## B. Retrieval-time prompts (LLM calls made DURING a build, not frozen)

| used by | prompt | location |
|---|---|---|
| tog — relation prune | `REL_PRUNE` | code/tog_baseline.py:49 |
| tog — entity prune | `ENT_PRUNE` | code/tog_baseline.py:57 |
| tog — sufficiency check | `SUFF` | code/tog_baseline.py:65 |
| hykge — hypothesis output | `HO_PROMPT` (paper P_HO) | code/hykge_baseline.py:63 |
| walker — on-demand duration | `ROLE_PROMPTS` | code/dkr_policy/bc_ondemand.py:39 |
| seeds (per question) | `PROMPT` | code/extract_multitype_seeds.py:17 |
| patient duration | `_build_extraction_prompt` | scripts/extract_patient_duration_llm.py:68 |

*(prompts.py also carries reference copies `TOG_REL_PRUNE` (116) and `HYKGE_HYPOTHESIS` (124), but
the LIVE ones the baselines actually use are in the baseline files above.)*

## C. Small / open-model answer format

`run_reader_block.py` sends the SAME frozen prompt but swaps the answer instruction: it strips the
`<a></a>` line and appends `prompts.REINRAG_OUTPUT` (line 136) — the `[Reasoning]` / `[Answer]`
block. Transform functions: `prompts.to_big_model_prompt()` / `to_small_model_prompt()`.

---

## Quick ways to read a concrete prompt

```bash
# the fixed template
sed -n '68,97p' pipeline/prompts.py                       # e.g. WALKER

# an actual filled prompt for a given question
python3 -c "import json; d=json.load(open('pipeline/frozen/1273/walker.json'))['items']; \
print(next(i['prompt'] for i in d if i['uid']=='test_0001'))"

# every method, one shared question, side by side
open pipeline/PROMPTS_1273.md
```

Both datasets: `frozen/1273/` is complete (all 9 methods). `frozen/329/` has 7 — `walker` there
is still the old (stale) version and `walker_interval` is not built (those two need the Neo4j
walker builds in RUN_PIPELINE.md step 2 with `DATASET=329`).

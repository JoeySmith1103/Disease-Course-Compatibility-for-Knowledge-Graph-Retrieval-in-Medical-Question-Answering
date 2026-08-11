#!/usr/bin/env bash
# Re-render every KG-bearing method of a dataset with one template, so all methods are compared
# under identical wording.
#
# vanilla/cot are skipped by design, not by oversight: the revised template announces
# "Supplementary evidence from a clinical knowledge graph:" and then interpolates the block, so
# rendering it for a method that has no block produces a prompt that promises evidence and shows
# none. A no-KG control needs its own template, not this one.
#
# Costs nothing but seconds — kg_block is stored separately from prompt in the frozen files, so
# no Neo4j and no LLM calls are involved. Evidence is byte-identical to the original freeze;
# only the wording changes, which is what makes prompt variants a controlled comparison.
#
# Usage:  bash pipeline/rerender_all_kg.sh medbullets [VARIANT] [TEMPLATE_FILE]
set -eu
DS="${1:?usage: rerender_all_kg.sh <dataset> [variant] [template_file]}"
VAR="${2:-revised}"
TPL="${3:-pipeline/prompt_revised.txt}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for M in walker walker_interval raw_1hop raw_2hop tog hykge; do
  f="pipeline/frozen/$DS/$M.json"
  [ -f "$f" ] || { echo "skip $M (no $f)"; continue; }
  DATASET=$DS METHOD=$M VARIANT=$VAR TEMPLATE_FILE=$TPL \
    python3 pipeline/rerender_prompts.py | grep -E "re-rendered|kg_block injected|kg transform"
done

echo
python3 - "$DS" "$VAR" <<'PY'
import json, sys, glob, os
ds, var = sys.argv[1], sys.argv[2]
print(f"frozen/{ds}/*__{var}.json :")
for f in sorted(glob.glob(f"pipeline/frozen/{ds}/*__{var}.json")):
    try: d = json.load(open(f))
    except Exception: continue
    if not isinstance(d, dict) or "items" not in d: continue
    it = d["items"]
    kg = sum(1 for i in it if (i.get("kg_block") or "").strip())
    # the freeze is only useful if the evidence actually reached the prompt
    bad = [i["uid"] for i in it if (i.get("kg_block") or "").strip()
           and i["kg_block"].strip() not in i["prompt"]]
    flag = f"  ⚠ {len(bad)} 題 kg_block 未進 prompt" if bad else ""
    print(f"  {os.path.basename(f):30s} n={len(it):4d}  kg_block {kg:4d} ({100*kg/len(it):5.1f}%){flag}")
PY

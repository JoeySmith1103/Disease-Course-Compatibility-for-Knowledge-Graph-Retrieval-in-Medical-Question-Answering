#!/usr/bin/env python3
"""Local vLLM offline reader for frozen multiple-choice prompts.

Reads frozen/<dataset>/<method>.json and writes one JSON per dataset/method/run. This script is a
local helper for the README vLLM workflow; it is intentionally ignored by git in this clone.

Examples:
  DATASETS=329,1273 METHODS=walker__revised,walker_interval__revised \
  MODEL=openai/gpt-oss-20b RUNS=1,2,3 BATCH_SIZE=64 PROMPT_VARIANTS=frozen_legacy \
  RESULTS_DIR=results/vllm_gptoss_v2 python3 run_reader_vllm_batch.py

  DATASETS=329 METHODS=vanilla,cot,raw_1hop__revised,raw_2hop__revised,walker__revised \
  MODEL=Qwen/Qwen3.5-9B RUNS=1 BATCH_SIZE=32 PROMPT_VARIANTS=small_model_block \
  VLLM_MAX_TOKENS=8192 RESULTS_DIR=results/vllm_qwen_block python3 run_reader_vllm_batch.py
"""
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(PIPE / "code"))
sys.path.insert(0, str(PIPE / "code" / "dkr_policy"))

spec = importlib.util.spec_from_file_location("P", PIPE / "prompts.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

DATASET = os.environ.get("DATASET", "329")
DATASETS = [x for x in os.environ.get("DATASETS", DATASET).split(",") if x]
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
MODEL_NAME = MODEL[len("vllm:"):] if MODEL.startswith("vllm:") else MODEL
RUN = os.environ.get("RUN", "1")
RUNS = [x for x in os.environ.get("RUNS", RUN).split(",") if x]
METHODS = [x for x in os.environ.get(
    "METHODS",
    "vanilla,cot,raw_1hop__revised,raw_2hop__revised,tog__revised,hykge__revised,"
    "walker__revised,walker_interval__revised",
).split(",") if x]
PROMPT_VARIANT = os.environ.get("PROMPT_VARIANT", "small_model_block")
PROMPT_VARIANTS = [x for x in os.environ.get("PROMPT_VARIANTS", PROMPT_VARIANT).split(",") if x]
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "0"))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "42"))
RES = PIPE / os.environ.get("RESULTS_DIR", "results")
RES.mkdir(parents=True, exist_ok=True)

ANSWER_LETTERS = "ABCDEFGHIJKL"
OPTS_CACHE = {}


def _env_bool(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _tag(text):
    base = Path(str(text).rstrip("/")).name or str(text)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base.replace(":", "_"))


def _base_method(method):
    return method.split("__", 1)[0]


def _options_for_uid(uid):
    if DATASET not in OPTS_CACHE:
        bench = json.load(open(PIPE / f"datasets/{DATASET}/benchmark.json"))
        OPTS_CACHE[DATASET] = {b["uid"]: b.get("options", {}) for b in bench}
    return OPTS_CACHE[DATASET].get(uid, {})


def _valid_letters(options=None):
    return set(str(k).upper() for k in options) if options else set(ANSWER_LETTERS)


def _extract_letter(raw, options=None):
    """Local A-L extractor so 329 K/L options do not depend on the repo parser state."""
    raw = raw or ""
    valid = _valid_letters(options)
    cls = "".join(sorted(valid))
    patterns = [
        rf"\[Answer\]\s*:?\s*\**([{cls}])\b",
        rf"ANSWER:\s*\**([{cls}])\b",
        rf"<a>\s*([{cls}])\s*</a>",
        rf"\\boxed\{{\s*([{cls}])\s*\}}",
        rf"<\s*[a-z]\s*>\s*([{cls}])\s*<\s*/\s*[a-z]?\s*>",
        rf"final answer[^A-Za-z\n]{{0,12}}<\s*([{cls}{cls.lower()}])\s*>",
        rf"final answer[^A-Za-z\n]{{0,12}}<?\s*([{cls}])\s*<",
        rf"\*\*?Answer:?\*?\*?\s*:?\s*\*?\*?([{cls}])\b",
        rf"final answer[:\s]+(?:is\s+)?(?:\\boxed\{{)?\*?\*?([{cls}])\b",
        rf"answer is[:\s]+(?:\\boxed\{{)?\*?\*?([{cls}])\b",
        rf"\bthe answer\s*(?:is|:)\s*\*?\*?([{cls}])\b",
        rf"(?:^|\n)\s*\**\s*([{cls}])\s*\**\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.I)
        if m:
            lab = m.group(1).upper()
            if lab in valid:
                return lab
    if options:
        m = re.search(r"<a>\s*(.+?)\s*</a>", raw, re.I | re.S)
        if m:
            txt = m.group(1).strip().lower()
            hit = [str(k).upper() for k, v in options.items() if (v or "").strip().lower() == txt]
            if len(hit) == 1:
                return hit[0]
    return None


def _evidence_lines(text):
    return {re.sub(r"^\s*(?:\d+\.|[-*])\s*", "", ln).strip()
            for ln in (text or "").splitlines() if ln.strip()}


def iter_batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_vllm():
    from vllm import LLM, SamplingParams

    kwargs = {
        "model": MODEL_NAME,
        "trust_remote_code": _env_bool("VLLM_TRUST_REMOTE_CODE", True),
        "tensor_parallel_size": int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
        "max_num_seqs": int(os.environ.get("VLLM_MAX_NUM_SEQS", str(BATCH_SIZE))),
    }
    optional = {
        "dtype": os.environ.get("VLLM_DTYPE"),
        "quantization": os.environ.get("VLLM_QUANTIZATION"),
        "max_model_len": os.environ.get("VLLM_MAX_MODEL_LEN"),
        "max_num_batched_tokens": os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS"),
    }
    for key, value in optional.items():
        if value:
            kwargs[key] = int(value) if key in ("max_model_len", "max_num_batched_tokens") else value

    llm = LLM(**kwargs)
    max_tokens_default = "1" if _env_bool("VLLM_FORCE_CHOICE", False) else "2048"
    sampling_kwargs = {
        "temperature": float(os.environ.get("VLLM_TEMPERATURE", "1.0")),
        "max_tokens": int(os.environ.get("VLLM_MAX_TOKENS", max_tokens_default)),
        "seed": int(os.environ.get("VLLM_SEED", "42")),
    }
    if _env_bool("VLLM_FORCE_CHOICE", False):
        tokenizer = llm.get_tokenizer()
        allowed = []
        for letter in ANSWER_LETTERS:
            for text in (letter, " " + letter):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    allowed.append(ids[0])
        sampling_kwargs["allowed_token_ids"] = sorted(set(allowed))
    try:
        sampling = SamplingParams(**sampling_kwargs)
    except TypeError:
        sampling_kwargs.pop("seed", None)
        sampling = SamplingParams(**sampling_kwargs)
    return llm, sampling


def apply_chat_template(llm, prompt):
    if not _env_bool("VLLM_APPLY_CHAT_TEMPLATE", True):
        return prompt
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {"enable_thinking": _env_bool("VLLM_ENABLE_THINKING", False)}
    effort = os.environ.get("VLLM_REASONING_EFFORT")
    if effort:
        template_kwargs["reasoning_effort"] = effort
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **template_kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _between(text, start_label, end_labels):
    i = (text or "").find(start_label)
    if i < 0:
        return ""
    i += len(start_label)
    if i < len(text) and text[i] == "\n":
        i += 1
    end = len(text)
    for label in end_labels:
        j = text.find(label, i)
        if j >= 0 and j < end:
            end = j
    return text[i:end].strip()


def _extract_question_options(frozen_prompt):
    question = _between(frozen_prompt, "Question:", ("\n\nOptions:",))
    options = _between(
        frozen_prompt,
        "Options:",
        ("\n\nSupplementary evidence", "\n\nRetrieved textbook", "\n\nThink step by step:",
         "\n\n[Output Format]"),
    )
    return question, "\n".join(line.strip() for line in options.splitlines() if line.strip())


def _extract_duration_text(frozen_prompt):
    m = re.search(r"Patient symptom duration:\s*([^\n]+)", frozen_prompt or "")
    return m.group(1).strip() if m else "not stated"


def _duration_to_hours_text(duration_text):
    raw = (duration_text or "").strip()
    if not raw:
        return "not stated"
    low = raw.lower()
    if "not stated" in low or "not applicable" in low:
        return "not stated"
    unit_re = r"seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?"
    range_m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:-|to|-)\s*([0-9]+(?:\.[0-9]+)?)\s*(" + unit_re + r")", low)
    if range_m:
        value = (float(range_m.group(1)) + float(range_m.group(2))) / 2.0
        unit = range_m.group(3)
    else:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(" + unit_re + r")", low)
        if not m:
            return "not stated"
        value = float(m.group(1))
        unit = m.group(2)
    if unit.startswith(("second", "sec")):
        hours = value / 3600.0
    elif unit.startswith(("minute", "min")):
        hours = value / 60.0
    elif unit.startswith(("hour", "hr")):
        hours = value
    elif unit.startswith("day"):
        hours = value * 24.0
    elif unit.startswith("week"):
        hours = value * 24.0 * 7.0
    elif unit.startswith("month"):
        hours = value * 24.0 * 30.0
    elif unit.startswith("year"):
        hours = value * 24.0 * 365.0
    else:
        return "not stated"
    return str(int(round(hours))) if abs(hours - round(hours)) < 1e-9 else f"{hours:.2f}".rstrip("0").rstrip(".")


def _canonical_prompt_variant(name):
    aliases = {
        "original": "frozen_legacy",
        "legacy": "frozen_legacy",
        "walker_original": "frozen_legacy",
        "prompt_v1": "walker_v1",
        "prompt_ver1": "walker_v1",
        "prompt_v2": "walker_v2",
        "prompt_ver2": "walker_v2",
        "prompt_ver3": "walker_v2",
        "walker_v3": "walker_v2",
        "walker_v4_brief_reasoning": "walker_v2",
    }
    return aliases.get(name, name)


def _small_model_prompt(frozen_prompt):
    prompt = P.to_small_model_prompt(frozen_prompt) if hasattr(P, "to_small_model_prompt") else frozen_prompt
    return re.sub(r"A single capital letter \(A-[A-Z]\)\s*[^\n]*", "A single capital letter from the provided options.", prompt)


def _walker_template(name):
    if name == "walker_v1" and hasattr(P, "WALKER_V1"):
        return P.WALKER_V1
    if name == "walker_v2" and hasattr(P, "WALKER_V2"):
        return P.WALKER_V2
    if name == "walker_v2" and (PIPE / "prompt_revised.txt").exists():
        return (PIPE / "prompt_revised.txt").read_text()
    raise ValueError(f"{name} is not available in prompts.py or prompt_revised.txt")


def build_reader_prompt(item, method):
    frozen_prompt = item.get("prompt") or ""
    variant = _canonical_prompt_variant(PROMPT_VARIANT)
    if variant == "small_model_block":
        return _small_model_prompt(frozen_prompt)
    if variant == "frozen_legacy":
        return P.to_big_model_prompt(frozen_prompt) if hasattr(P, "to_big_model_prompt") else frozen_prompt

    if variant not in ("walker_v1", "walker_v2"):
        raise ValueError(f"Unknown PROMPT_VARIANT={PROMPT_VARIANT}")
    if _base_method(method) not in ("walker", "walker_interval", "walker_criticality", "walker_adaptive"):
        raise ValueError("walker prompt variants are only valid for walker-style methods")
    question, options = _extract_question_options(frozen_prompt)
    if not question or not options:
        raise ValueError(f"Could not parse question/options for {item.get('uid')}")
    duration_text = _extract_duration_text(frozen_prompt)
    kg = item.get("kg_block") or "No retrieved information available."
    vals = {
        "question": question,
        "options": options,
        "options_block": options,
        "symptom_duration": _duration_to_hours_text(duration_text),
        "patient_duration": duration_text,
        "patient_dur_str": duration_text,
        "retrieved_information": kg,
        "kg_block": kg,
        "retrieval": kg,
        "evidence": kg,
    }
    return _walker_template(variant).format(**vals)


def make_record(item, prompt, raw):
    pred = _extract_letter(raw, _options_for_uid(item["uid"]))
    return {
        "uid": item["uid"],
        "gold": item["gold"],
        "predicted": pred,
        "is_correct": pred == item["gold"],
        "route": item.get("route"),
        "kg_block": item.get("kg_block") or "",
        "prompt": prompt,
        "raw_response": raw,
    }


def save(out_path, method, results):
    n = len(results)
    c = sum(1 for x in results if x["is_correct"])
    none = sum(1 for x in results if x["predicted"] is None)
    json.dump({
        "dataset": DATASET,
        "method": method,
        "model": "vllm:" + MODEL_NAME,
        "engine": "vllm_offline",
        "run": RUN,
        "prompt": PROMPT_VARIANT,
        "prompt_variant": PROMPT_VARIANT,
        "limit": LIMIT,
        "sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
        "batch_size": BATCH_SIZE,
        "force_choice": _env_bool("VLLM_FORCE_CHOICE", False),
        "n": n,
        "n_correct": c,
        "runs_correct": [c],
        "mean_correct": c,
        "std_correct": 0.0,
        "n_unparseable": none,
        "runs_unparseable": [none],
        "mean_unparseable": none,
        "accuracy": 100 * c / max(1, n),
        "mean_acc": 100 * c / max(1, n),
        "std_acc": 0.0,
        "results": results,
        "runs": [{"run": RUN, "results": results}],
    }, open(out_path, "w"), indent=1)


def run_dataset(method, llm, sampling, model_tag, prompt_part, summary_kg_methods):
    fp = PIPE / f"frozen/{DATASET}/{method}.json"
    if not fp.exists():
        print(f"[{DATASET}/{method}] no frozen file, skip", flush=True)
        return

    items = json.load(open(fp))["items"]
    if SAMPLE_SIZE > 0:
        pool = sorted(items, key=lambda it: it["uid"])
        rng = random.Random(f"{SAMPLE_SEED}:{DATASET}")
        chosen = {it["uid"] for it in rng.sample(pool, min(SAMPLE_SIZE, len(pool)))}
        items = [it for it in items if it["uid"] in chosen]
    if LIMIT > 0:
        items = items[:LIMIT]

    sample_part = f"_sample{SAMPLE_SIZE}s{SAMPLE_SEED}" if SAMPLE_SIZE > 0 else ""
    out_path = RES / f"{DATASET}_{method}_{model_tag}{prompt_part}{sample_part}_vllm_batch_run{RUN}.json"
    done = {}
    if out_path.exists():
        try:
            done = {r["uid"]: r for r in json.load(open(out_path)).get("results", [])}
        except Exception:
            done = {}

    todo = [it for it in items if it["uid"] not in done]
    results = list(done.values())
    initial_done = len(results)
    started = time.monotonic()
    print(f"[{DATASET}/{method}] vllm run{RUN} prompt={PROMPT_VARIANT} todo {len(todo)}/{len(items)}", flush=True)

    prepared = []
    for it in todo:
        prompt = build_reader_prompt(it, method)
        kg = it.get("kg_block") or ""
        if kg and _base_method(method) not in summary_kg_methods:
            missing = _evidence_lines(kg) - _evidence_lines(prompt)
            if missing:
                raise AssertionError(f"BUG: kg_block evidence missing from prompt for {it['uid']}")
        prepared.append((it, prompt, apply_chat_template(llm, prompt)))

    for chunk in iter_batches(prepared, BATCH_SIZE):
        try:
            outputs = llm.generate([x[2] for x in chunk], sampling)
            for (it, prompt, _), out in zip(chunk, outputs):
                raw = out.outputs[0].text.strip() if out.outputs else ""
                results.append(make_record(it, prompt, raw))
        except Exception as e:
            print(f"  [{DATASET}/{method}] batch failed ({len(chunk)} items): {repr(e)[:160]}", flush=True)
            for single in chunk:
                try:
                    out = llm.generate([single[2]], sampling)[0]
                    raw = out.outputs[0].text.strip() if out.outputs else ""
                except Exception as inner:
                    print(f"  [{DATASET}/{method}] single failed {single[0]['uid']}: {repr(inner)[:160]}", flush=True)
                    raw = ""
                results.append(make_record(single[0], single[1], raw))

        elapsed = time.monotonic() - started
        generated = max(1, len(results) - initial_done)
        save(out_path, method, results)
        c = sum(1 for x in results if x["is_correct"])
        none = sum(1 for x in results if x["predicted"] is None)
        print(f"  [{DATASET}/{method}] {len(results)}/{len(items)} acc={100*c/len(results):.1f}% "
              f"unparseable={none} elapsed={elapsed:.1f}s sec/item={elapsed/generated:.2f}", flush=True)

    save(out_path, method, results)
    c = sum(1 for x in results if x["is_correct"])
    none = sum(1 for x in results if x["predicted"] is None)
    elapsed = time.monotonic() - started
    generated = max(1, len(results) - initial_done)
    print(f"[{DATASET}/{method}] run{RUN} DONE {c}/{len(results)} = {100*c/max(1,len(results)):.2f}% "
          f"(unparseable={none}, elapsed={elapsed:.1f}s, sec/item={elapsed/generated:.2f})", flush=True)


def main():
    global DATASET, RUN, PROMPT_VARIANT
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be >= 1")
    llm, sampling = build_vllm()
    model_tag = _tag(MODEL_NAME)
    summary_kg_methods = set(x for x in os.environ.get("SUMMARY_KG_METHODS", "tog,hykge").split(",") if x)

    for prompt_variant in PROMPT_VARIANTS:
        PROMPT_VARIANT = prompt_variant
        prompt_part = "" if PROMPT_VARIANT == "small_model_block" else f"_{_tag(PROMPT_VARIANT)}"
        for run in RUNS:
            RUN = run
            for dataset in DATASETS:
                DATASET = dataset
                for method in METHODS:
                    run_dataset(method, llm, sampling, model_tag, prompt_part, summary_kg_methods)


if __name__ == "__main__":
    main()

"""LLM API client — the single home for `call_llm` and every provider it dispatches to.

`call_llm(prompt, model)` is the one entry point the rest of the pipeline uses. Routing by
model-name prefix:
  'together:<model>'            → Together AI  (small/medium open models, e.g. Qwen3.5-9B)
  'litellm:<model>' / 'gemma*'  → NetDB litellm proxy
  'google:<model>'              → direct Google AI Studio (bypass litellm)
  'gemini-3*' / 'gemini-2.5-flash' / 'gpt-oss*' → NetDB litellm allowlist
  'vllm:<model>' / 'qwen*'      → local vLLM serve endpoint
  'gpt-*' / 'o*'                → OpenAI
  else                          → Gemini (direct)

API keys are read from ../api_key/ (or ~/Desktop/Master_thesis/api_key/) or the matching env var.
"""
import os
import time
import requests


def get_api_key() -> str:
    """Gemini (Google AI Studio) key."""
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, '..', 'api_key', 'own_gemini_api'),
        os.path.join(here, '..', '..', 'api_key', 'own_gemini_api'),
        os.path.expanduser('~/Desktop/Master_thesis/api_key/own_gemini_api'),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("No Gemini API key found. Tried: " + ', '.join(candidates))
    return key


def call_openai(api_key: str, prompt: str, model: str = "gpt-4.1-mini",
                temperature: float = 1.0, max_tokens: int = 2048,
                seed: int = None) -> str:
    """Call OpenAI API. Handles parameter-naming differences between model families.

    seed: passed when supported (best-effort determinism). gpt-5 / o-series
    accept it; older chat models also accept it. Without a seed, default
    sampling produces ~3-5pp run-to-run variance on bench325.
    """
    if seed is None:
        seed = int(os.environ.get("OPENAI_SEED", "42"))
    url = "https://api.openai.com/v1/chat/completions"
    is_gpt5 = model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3') \
              or model.startswith('o4') or model.startswith('o5')
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "seed": seed,
    }
    if is_gpt5:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=180)
            resp.raise_for_status()
            body = resp.json()
            return body["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [OpenAI attempt {attempt+1}] {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


def call_gemini(api_key: str, prompt: str, model: str = "gemini-2.5-flash",
                temperature: float = 1.0, max_tokens: int = 8192) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    gen_config = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if 'gemini-2.5' in model or 'gemini-3' in model:
        gen_config["thinkingConfig"] = {"thinkingBudget": 2048}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            body = resp.json()
            candidates = body.get("candidates", [])
            if not candidates:
                raise ValueError("no candidates")
            parts = candidates[0].get("content", {}).get("parts", [])
            # For thinking models, skip thought parts and return the actual answer
            for part in reversed(parts):
                if "text" in part and not part.get("thought", False):
                    return part["text"].strip()
            raise ValueError("no non-thought text part")
        except Exception as e:
            print(f"  [Gemini attempt {attempt+1}] {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


def get_openai_api_key() -> str:
    here = os.path.dirname(__file__)
    for path in (
        os.path.join(here, '..', 'api_key', 'openai_api'),
        os.path.join(here, '..', '..', 'api_key', 'openai_api'),
        os.path.expanduser('~/Desktop/Master_thesis/api_key/openai_api'),
    ):
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
    return os.environ.get("OPENAI_API_KEY", "")


def call_vllm_openai(prompt: str, model: str = "qwen35-27b",
                       base_url: str = None,
                       temperature: float = 1.0,
                       max_tokens: int = 2048,
                       enable_thinking: bool = False) -> str:
    """OpenAI-compatible call to a local vLLM serve endpoint.

    Used for NCHC-hosted Qwen3.5-27B etc. Set VLLM_BASE_URL env
    (default http://localhost:8000/v1).

    For Qwen3.x models, `enable_thinking=False` skips the chain-of-thought
    block and returns the answer directly — required for MCQ where we
    expect <a>X</a> in the output.
    """
    base_url = base_url or os.environ.get("VLLM_BASE_URL",
                                            "http://localhost:8000/v1")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model.startswith("qwen3") or "qwen3" in model.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    headers = {"Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [vllm attempt {attempt+1}] {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


def _litellm_token() -> str:
    """Resolve the NetDB litellm token live (it rotates).
    Priority: env LITELLM_API_KEY → api_key/gemini_netdb_route.py → empty.

    There is no hardcoded fallback. One used to live here, which meant a real credential was
    sitting in a tracked source file; a stale key that fails loudly is strictly better than a live
    one that ships to anyone who clones the repo.
    """
    import re
    t = os.environ.get("LITELLM_API_KEY")
    if t:
        return t
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, '..', 'api_key', 'gemini_netdb_route.py'),
              os.path.expanduser('~/Desktop/Master_thesis/api_key/gemini_netdb_route.py')):
        try:
            m = re.search(r'api_key\s*=\s*["\'](sk-[A-Za-z0-9_\-]+)["\']', open(p).read())
            if m:
                return m.group(1)
        except Exception:
            pass
    return ""


def _together_key() -> str:
    t = os.environ.get("TOGETHER_API_KEY")
    if t: return t
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, '..', 'api_key', 'together_ai_api'),
              os.path.expanduser('~/Desktop/Master_thesis/api_key/together_ai_api')):
        try: return open(p).read().strip()
        except Exception: pass
    return ""


def call_together(prompt: str, model: str, temperature: float = 1.0, max_tokens: int = 8192) -> str:
    """OpenAI-compatible call to Together AI (small/medium open models, e.g. Qwen).
    max_tokens defaults to 8192: Qwen3.x are REASONING models whose <think> block eats the budget —
    at 2048 they truncate (finish_reason=length) with EMPTY content. Also retries on empty."""
    import openai
    # timeout so a hung call fails fast instead of blocking the whole batch; max_retries=0 (our loop retries)
    client = openai.OpenAI(api_key=_together_key(), base_url="https://api.together.xyz/v1",
                           timeout=90.0, max_retries=0)
    # NOTE: Qwen3.x default to THINKING mode (slow ~90s/call but reliable format). enable_thinking=False
    # is fast but Together applies it inconsistently and the answer format often breaks — so we keep
    # thinking ON (with the 8192 budget) for reliable parseable answers.
    extra = {}
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens, **extra)
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
            # empty content — for reasoning models it may be in reasoning_content
            rc = getattr(resp.choices[0].message, "reasoning_content", None)
            if rc and rc.strip():
                return rc.strip()
            print(f"  [together attempt {attempt+1}] empty response, retrying")
        except Exception as e:
            print(f"  [together attempt {attempt+1}] {e}")
        if attempt < 4: time.sleep(1.5 * (attempt + 1))
    return ""


def call_litellm(prompt: str, model: str,
                  base_url: str = "https://litellm.netdb.csie.ncku.edu.tw",
                  api_key: str = None,
                  temperature: float = 1.0,
                  max_tokens: int = 2048) -> str:
    """OpenAI-compatible call routed via NetDB litellm proxy.
    Used for Gemma 4 (gemma4:31b-IT-NVFP4) and other models served on
    NetDB infra without going through NCHC.
    """
    import openai
    client = openai.OpenAI(api_key=api_key or _litellm_token(), base_url=base_url)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  [litellm attempt {attempt+1}] {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


def call_llm(prompt: str, model: str, api_key: str = None,
             temperature: float = None, max_tokens: int = None) -> str:
    """Dispatch to the correct API based on model name. See module docstring for routing.

    temperature / max_tokens are optional; when None each provider keeps its own default, so
    existing callers are unaffected. They exist so a baseline can reproduce a paper's decoding
    settings (e.g. HyKGE generates its hypothesis at temperature=0.6, max_tokens=300).
    """
    kw = {}
    if temperature is not None: kw['temperature'] = temperature
    if max_tokens is not None:  kw['max_tokens'] = max_tokens

    if model.startswith('together:'):
        return call_together(prompt, model=model[len('together:'):], **kw)
    if model.startswith('litellm:'):
        return call_litellm(prompt, model=model[len('litellm:'):], **kw)
    # 'google:<model>' prefix → direct Google AI Studio API (bypass NetDB litellm)
    if model.startswith('google:'):
        key = api_key or get_api_key()
        return call_gemini(key, prompt, model=model[len('google:'):], **kw)
    # NetDB litellm allowlist: gemma4, gpt-oss:*, gemini-3-*, gemini-2.5-flash
    if (model.startswith('gemma') or model.startswith('gpt-oss')
            or model.startswith('gemini-3') or model == 'gemini-2.5-flash'):
        return call_litellm(prompt, model=model, **kw)
    if model.startswith('vllm:'):
        return call_vllm_openai(prompt, model=model[len('vllm:'):], **kw)
    if model.startswith('qwen') or model in ('qwen35-27b', 'qwen3.5-27b'):
        return call_vllm_openai(prompt, model=model, **kw)
    if model.startswith('gpt-') or model.startswith('o'):
        key = api_key or get_openai_api_key()
        return call_openai(key, prompt, model=model, **kw)
    else:
        key = api_key or get_api_key()
        return call_gemini(key, prompt, model=model, **kw)

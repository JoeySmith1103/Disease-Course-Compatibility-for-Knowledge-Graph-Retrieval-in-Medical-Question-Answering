"""Text encoders for the UMLS embedding matrix.

Extracted from the old spectrum_mdn.py, which was 819 lines of MDN training code of which only
these encoders were still reachable — the MDN itself was dropped when bc moved to all-LLM
durations (see dkr_policy/bc_llm_direct.py). `make_encoder` is what `duration_kg_rag._load_umls_emb`
uses to rebuild the encoder named inside each embedding pickle (ours say `encoder='sapbert'`).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

# module-level constants the encoders reference (from the original spectrum_mdn.py)
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SAPBERT_DIM = 768
OPENAI_SMALL_DIM = 1536
OPENAI_LARGE_DIM = 3072

class SapBERTEncoder:
    """Lazy wrapper that encodes strings to normalized SapBERT [CLS] embeddings.

    For training we precompute embeddings for the disease list once. At
    inference we may also encode option strings on the fly.
    """

    OUT_DIM = SAPBERT_DIM

    def __init__(self, model_name: str = SAPBERT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = None
        self._mdl = None

    def _load(self):
        if self._mdl is None:
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._mdl = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._mdl.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)
            out = self._mdl(**enc)
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, dim=1)
            embs.append(cls.cpu().numpy().astype(np.float32))
        return np.concatenate(embs, axis=0)


class OpenAIEmbedder:
    """Encode strings via OpenAI's text-embedding-3 endpoint.

    Use case: SapBERT was pretrained for entity linking (UMLS synonyms) and
    is sometimes weak at distinguishing compositional phrases like "X during
    acute episode" vs "X in chronic course". OpenAI embeddings are trained on
    a much broader corpus and may pick up the descriptor more cleanly.

    Embeddings are L2-normalised after the API call so cosine similarity ≈
    dot product, matching SapBERTEncoder semantics. API is only hit when an
    embedding is not in the local cache; recommended workflow is to
    precompute once into a pickle (same as SapBERT) and reuse offline.
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    OUT_DIM_BY_MODEL = {
        "text-embedding-3-small": OPENAI_SMALL_DIM,
        "text-embedding-3-large": OPENAI_LARGE_DIM,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.OUT_DIM = self.OUT_DIM_BY_MODEL.get(model, OPENAI_SMALL_DIM)
        if api_key is None:
            if api_key_path is None:
                api_key_path = str(
                    Path(__file__).resolve().parent.parent / "api_key" / "openai_api"   # pipeline/api_key
                )
            with open(api_key_path) as f:
                api_key = f.read().strip()
        self._api_key = api_key
        self._client = None

    def _load(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "openai package required for OpenAIEmbedder; pip install openai"
                ) from exc
            self._client = OpenAI(api_key=self._api_key)

    def encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        """Returns L2-normalised embeddings, shape (n, OUT_DIM)."""
        self._load()
        embs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = self._client.embeddings.create(model=self.model, input=batch)
            arr = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            # Normalize for cosine compatibility with downstream MLP head.
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embs.append(arr / norms)
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, self.OUT_DIM), dtype=np.float32)


def make_encoder(name: str, **kwargs):
    """Factory: 'sapbert' | 'openai-small' | 'openai-large'."""
    n = name.lower()
    if n in ("sapbert", "sap"):
        return SapBERTEncoder(**{k: v for k, v in kwargs.items() if k in ("model_name", "device")})
    if n in ("openai", "openai-small", "openai-3-small"):
        return OpenAIEmbedder(model="text-embedding-3-small")
    if n in ("openai-large", "openai-3-large"):
        return OpenAIEmbedder(model="text-embedding-3-large")
    raise ValueError(f"unknown encoder: {name}")

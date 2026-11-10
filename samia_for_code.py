#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    f1_score,
    precision_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Paths / reproducibility
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "samia_outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
PLOT_DIR = OUTPUT_DIR / "plots"
RAW_DIR = OUTPUT_DIR / "raw_scores"
TABLE_DIR = OUTPUT_DIR / "tables"

DEFAULT_SEED = 42

# Official SAMIA generation hyperparameters (sampling.py).
SAMIA_TEMPERATURE = 1.0
SAMIA_TOP_K = 50
SAMIA_TOP_P = 1.0
# Official repo uses max_length=1024 (prompt+completion). On CPU we cap
# newly generated tokens; --paper_max_length restores 1024.
DEFAULT_MAX_NEW_TOKENS = 64

# OpenRouter slugs for commercial / hosted models (queried as requested; not swapped).
OPENROUTER_API_KEY_DEFAULT = (
    "Your API key here (or set OPENROUTER_API_KEY in your environment)."
)
OPENROUTER_SLUGS = {
    "Gemini-2.5-Flash": "google/gemini-2.5-flash",
    "Mistral-Small-24B-Instruct": "mistralai/mistral-small-24b-instruct-2501",
    "DeepSeek-V3": "deepseek/deepseek-chat-v3-0324",
    "LLaMA-3.3-70B-Instruct": "meta-llama/llama-3.3-70b-instruct",
    "GPT-5": "openai/gpt-5-chat",
}

# Local Hugging Face causal LMs (free, downloadable).
HF_MODELS = {
    "GPT-2": "gpt2",
    "DistilGPT-2": "distilgpt2",
    "DialoGPT-medium": "microsoft/DialoGPT-medium",
    "DialoGPT-large": "microsoft/DialoGPT-large",
}

MODEL_TABLE_ORDER = [
    "DialoGPT-medium",
    "DialoGPT-large",
    "GPT-2",
    "DistilGPT-2",
    "Gemini-2.5-Flash",
    "Mistral-Small-24B-Instruct",
    "DeepSeek-V3",
    "LLaMA-3.3-70B-Instruct",
    "GPT-5",
]


# ===========================================================================
# 1. Official SAMIA scoring (src/eval_samia.py + src/sampling.py)
# ===========================================================================

def tokenize_words(text: str) -> list[str]:
    """Whitespace tokenization used by the official SAMIA implementation."""
    return [w for w in str(text).split() if w != ""]


def get_prefix(text: str, prefix_ratio: float) -> str:
    """Official get_prefix: first prefix_ratio fraction of whitespace tokens."""
    words = tokenize_words(text)
    n = int(len(words) * prefix_ratio)
    n = max(1, min(n, max(0, len(words) - 1))) if words else 0
    return " ".join(words[:n])


def get_suffix_words(text: str, prefix_ratio: float, text_length: int | None = None) -> list[str]:
    """
    Official get_suffix (eval_samia.py): drop the first
    round(text_length * prefix_ratio) whitespace tokens.

    When text_length is None, use the actual word count (needed for code
    snippets that are not pre-truncated to WikiMIA lengths {32,64,128,256}).
    """
    words = tokenize_words(text)
    n = len(words) if text_length is None else text_length
    cut = round(n * prefix_ratio)
    return words[cut:]


def truncate_to_word_length(text: str, text_length: int) -> str:
    """WikiMIA-style length control: keep at most `text_length` whitespace tokens."""
    words = tokenize_words(text)
    if len(words) <= text_length:
        return " ".join(words)
    return " ".join(words[:text_length])


def ngrams(sequence: list[str], n: int):
    """Official ngrams()."""
    return zip(*[sequence[i:] for i in range(n)])


def rouge_n(candidate: list[str], reference: list[str], n: int = 1) -> float:
    """
    Official ROUGE-N. Returns *recall* (overlap / |reference|), not F1.
    This recall is the SAMIA surface-similarity used as pseudo-likelihood.
    """
    if not candidate or not reference:
        return 0.0
    candidate_ngrams = list(ngrams(candidate, n))
    reference_ngrams = list(ngrams(reference, n))
    if not candidate_ngrams or not reference_ngrams:
        return 0.0
    ref_count = Counter(reference_ngrams)
    cand_count = Counter(candidate_ngrams)
    overlap = ref_count & cand_count
    return sum(overlap.values()) / len(reference)


def clean_generated_text(text: str, model_name: str) -> str:
    """Strip common chat/causal special tokens (extends official clean_text)."""
    text = str(text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    text = re.sub(r"</s>", "", text)
    text = re.sub(r"<s>", "", text)
    text = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", text)
    # Chat models often echo the prompt; drop a duplicated prefix if present.
    return text.strip()


def samia_score_from_generations(
    original_text: str,
    generations: list[str],
    prefix_ratio: float,
    text_length: int,
    model_name: str,
    use_zlib: bool = False,
) -> dict[str, Any]:
    """
    Score one example the way eval_samia.py does:

      score = mean_i ROUGE-1_recall(suffix(gen_i), suffix(original))
      optional: multiply each ROUGE by zlib length of the generated suffix

    Higher score => the model more often regenerates the original suffix
    => treated as more member-like.
    """
    suffix_ref = get_suffix_words(original_text, prefix_ratio, text_length)
    per_sample = []
    for gen in generations:
        cleaned = clean_generated_text(gen, model_name)
        suffix_cand = get_suffix_words(cleaned, prefix_ratio, text_length)
        r = rouge_n(suffix_cand, suffix_ref, n=1)
        if use_zlib:
            payload = " ".join(suffix_cand).encode("utf-8")
            r = r * len(zlib.compress(payload)) if payload else 0.0
        per_sample.append(float(r))
    score = float(np.mean(per_sample)) if per_sample else float("nan")
    return {
        "samia_score": score,
        "rouge_per_sample": per_sample,
        "n_valid_generations": len(per_sample),
        "suffix_ref_len": len(suffix_ref),
    }


# ===========================================================================
# 2. Metrics (AUC, TPR@5%FPR from the ROC curve, Precision, F1)
# ===========================================================================

def tpr_at_target_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float = 0.05) -> float:
    """
    TPR at a target FPR taken from the ROC curve.

    Uses the largest TPR among ROC points with FPR <= target (standard MIA
    reporting). If every point has FPR > target (tiny samples), interpolate.
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if len(fpr) == 0:
        return float("nan")
    leq = fpr <= target + 1e-12
    if np.any(leq):
        return float(np.max(tpr[leq]))
    return float(np.interp(target, fpr, tpr))


def threshold_for_target_fpr(fpr, tpr, thresholds, target: float = 0.05) -> float:
    """Score threshold corresponding to the TPR@target-FPR operating point."""
    fpr = np.asarray(fpr, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    leq = np.where(fpr <= target + 1e-12)[0]
    if len(leq) == 0:
        return float(thresholds[int(np.argmin(np.abs(fpr - target)))])
    # Among points with FPR<=target, take the one with highest TPR (last such
    # sklearn point is typically the most permissive still under the cap).
    idx = int(leq[-1])
    thr = thresholds[idx]
    if np.isinf(thr):
        finite = thresholds[np.isfinite(thresholds)]
        return float(finite[0]) if len(finite) else 0.0
    return float(thr)


def compute_metrics(labels: list[int], scores: list[float], fpr_target: float = 0.05) -> dict[str, Any]:
    """Compute AUC, TPR@5%FPR, Precision, F1. Scores: higher => predict member."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    out = {
        "n": int(len(y)),
        "n_member": int((y == 1).sum()),
        "n_nonmember": int((y == 0).sum()),
        "auc": None,
        "tpr_at_5fpr": None,
        "precision": None,
        "f1": None,
        "threshold_at_5fpr": None,
        "note": "",
    }
    if len(y) < 4 or len(set(y.tolist())) < 2:
        out["note"] = "Need both classes and >=4 finite scores to compute ROC metrics."
        return out
    fpr, tpr, thr = roc_curve(y, s)
    out["auc"] = float(auc(fpr, tpr))
    out["tpr_at_5fpr"] = tpr_at_target_fpr(fpr, tpr, fpr_target)
    t = threshold_for_target_fpr(fpr, tpr, thr, fpr_target)
    out["threshold_at_5fpr"] = t
    pred = (s >= t).astype(int)
    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["fpr"] = fpr.tolist()
    out["tpr"] = tpr.tolist()
    return out


def format_metric(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    return f"{v:.{digits}f}"


def format_tpr_percent(v) -> str:
    """TPR@5%FPR printed as a percent (e.g. 9.24), matching the requested table."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    return f"{100.0 * float(v):.2f}"


# ===========================================================================
# 3. Dataset loading
# ===========================================================================

def _looks_like_python(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return False
    keys = ("def ", "import ", "class ", "return ", "print(", "if ", "for ", ":\n")
    t = text[:4000]
    return sum(k in t for k in keys) >= 2


def _extract_text_fields(item: dict) -> str:
    for key in ("answer", "text", "content", "code", "body", "input"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _parse_date(item: dict) -> datetime | None:
    for key in (
        "creation_date",
        "created_at",
        "date",
        "timestamp",
        "closed_date",
        "last_activity_date",
        "year",
    ):
        val = item.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            # Unix seconds or milliseconds.
            ts = float(val)
            if ts > 1e12:
                ts /= 1000.0
            try:
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (OSError, ValueError):
                year = int(val)
                if 1990 <= year <= 2100:
                    return datetime(year, 1, 1, tzinfo=timezone.utc)
        if isinstance(val, str) and val.strip():
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    return datetime.strptime(val[: len(fmt) + 5], fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            m = re.search(r"(19|20)\d{2}", val)
            if m:
                return datetime(int(m.group(0)), 1, 1, tzinfo=timezone.utc)
    return None


def find_file(name: str, extra: list[Path] | None = None) -> Path | None:
    candidates = [
        ROOT / name,
        ROOT / "data" / name,
        ROOT / "dataset" / name,
        ROOT / "Project_Submissions" / name,
        ROOT / "Project_submissions" / name,
        Path.cwd() / name,
        Path.cwd() / "Project_Submissions" / name,
        Path.cwd() / "Project_submissions" / name,
    ]
    if extra:
        candidates.extend(extra)
    for p in candidates:
        if p.is_file():
            return p
    return None


def find_dir(name: str) -> Path | None:
    names = [
        name,
        "Project_Submissions",
        "Project_submissions",
        "Project_submissiions",
        "student_submissions",
    ]
    for base in (ROOT, Path.cwd(), ROOT / "data"):
        for n in names:
            p = Path(base) / n
            if p.is_dir():
                return p
    return None


def load_json_or_jsonl(path: Path) -> list[dict]:
    """qa.en.python.json in this repo is JSONL (one object per line) or a JSON list."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        head = f.read(256)
        f.seek(0)
        stripped = head.lstrip()
        if stripped.startswith("["):
            data = json.load(f)
            if isinstance(data, list):
                rows = [x for x in data if isinstance(x, dict)]
            return rows
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_stackoverflow_samples(
    path: Path | None,
    max_samples: int,
    text_length: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load Stack Overflow Python Q/A.

    Membership construction (documented):
      * If a usable timestamp exists: WikiMIA-style temporal split.
        Pre-cutoff (default 2021-01-01) => member, post-cutoff => non-member.
        This matches the paper's seen/unseen construction (time-separated
        text from the same domain) as closely as the file allows.
      * If timestamps are absent: original SO answers are treated as
        *putative members* (public web text that many LLMs may have seen)
        and paired with unique synthetic Python *putative non-members*.
        This is a PROXY, not verified training-set membership.
    """
    meta = {
        "source_path": str(path) if path else None,
        "available": False,
        "n_raw": 0,
        "fields": [],
        "label_protocol": None,
        "cutoff": "2021-01-01",
        "warning": "",
    }
    if path is None or not path.is_file():
        meta["warning"] = (
            "qa.en.python.json was not found. Stack Overflow evaluation cannot "
            "use the requested file. Place it in the project root (JSONL with "
            "an 'answer' field) and re-run."
        )
        return [], meta

    # File can be ~2GB JSONL. Stream until we have enough valid Python answers
    # (plus a buffer for shuffling) instead of loading every row.
    want = max(max_samples * 8, 80)
    raw: list[dict] = []
    n_scanned = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        head = f.read(256)
        f.seek(0)
        if head.lstrip().startswith("["):
            data = json.load(f)
            raw = [x for x in data if isinstance(x, dict)][:want]
            n_scanned = len(data) if isinstance(data, list) else 0
        else:
            for line in f:
                n_scanned += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                text = _extract_text_fields(item)
                if not _looks_like_python(text) and len(tokenize_words(text)) < 16:
                    continue
                raw.append(item)
                if len(raw) >= want:
                    break
    meta["n_raw"] = n_scanned
    if raw:
        meta["fields"] = sorted({k for row in raw[:50] for k in row.keys()})

    samples = []
    n_with_date = 0
    for i, item in enumerate(raw):
        text = _extract_text_fields(item)
        if not _looks_like_python(text) and len(tokenize_words(text)) < 16:
            continue
        text = truncate_to_word_length(text, text_length)
        if len(tokenize_words(text)) < 12:
            continue
        dt = _parse_date(item)
        if dt is not None:
            n_with_date += 1
        samples.append(
            {
                "id": f"so_{item.get('answer_id', item.get('question_id', i))}",
                "text": text,
                "date": dt.isoformat() if dt else None,
                "title": item.get("title", ""),
                "origin": "stackoverflow",
            }
        )

    rng = random.Random(seed)
    rng.shuffle(samples)
    meta["available"] = True
    cutoff = datetime(2021, 1, 1, tzinfo=timezone.utc)

    dated = [s for s in samples if s["date"]]
    if n_with_date >= 20 and len({s["date"][:4] for s in dated}) >= 2:
        meta["label_protocol"] = "temporal_wikimia_style"
        members, nonmembers = [], []
        for s in dated:
            dt = datetime.fromisoformat(s["date"])
            rec = dict(s)
            if dt < cutoff:
                rec["label"] = 1
                members.append(rec)
            else:
                rec["label"] = 0
                nonmembers.append(rec)
        rng.shuffle(members)
        rng.shuffle(nonmembers)
        k = min(max_samples, len(members), len(nonmembers))
        labeled = members[:k] + nonmembers[:k]
        meta["warning"] = (
            f"Temporal split at {meta['cutoff']}: {k} members (pre-cutoff) and "
            f"{k} non-members (post-cutoff). This is WikiMIA-style, not a "
            f"verified dump of any one model's training set."
        )
        return labeled, meta

    # No reliable dates: putative members = SO answers; non-members = synthetic.
    meta["label_protocol"] = "so_as_putative_member_plus_synthetic_nonmember"
    members = samples[:max_samples]
    for s in members:
        s["label"] = 1
    nonmembers = make_synthetic_nonmembers([m["text"] for m in members], rng)
    labeled = members + nonmembers
    meta["warning"] = (
        "No usable timestamps in qa.en.python.json, so labels are a PROXY: "
        "original SO Python answers = putative members; unique synthetic "
        "Python (nonce identifiers, not in public dumps) = putative non-members. "
        "True membership of each target LLM is unknown. DialoGPT in particular "
        "was trained on Reddit, not Stack Overflow."
    )
    return labeled, meta


def make_synthetic_nonmembers(member_texts: list[str], rng: random.Random) -> list[dict[str, Any]]:
    """
    Construct code that is extremely unlikely to appear in pretraining:
    unique nonce identifiers + control-flow that is not a paraphrase of SO.
    Surface form remains Python so the comparison is still code-vs-code.
    """
    templates = [
        (
            "import math\n"
            "NONCE = '{nonce}'\n"
            "def samia_holdout_{nonce_fn}(xs):\n"
            "    acc = 0\n"
            "    for i, x in enumerate(xs):\n"
            "        acc += (x * x + i) ^ (i & 3)\n"
            "        if acc % 7 == 0:\n"
            "            acc = acc // 2\n"
            "    return acc, NONCE, math.sqrt(abs(acc) + 1)\n"
            "if __name__ == '__main__':\n"
            "    print(samia_holdout_{nonce_fn}(list(range(12))))\n"
        ),
        (
            "from collections import deque\n"
            "MARKER = '{nonce}'\n"
            "class Holdout_{nonce_fn}:\n"
            "    def __init__(self, seed):\n"
            "        self.seed = seed\n"
            "        self.q = deque([seed])\n"
            "    def step(self, n):\n"
            "        for k in range(n):\n"
            "            v = (self.q[-1] * 1103515245 + k) & 0xFFFFFFFF\n"
            "            self.q.append(v)\n"
            "        return list(self.q), MARKER\n"
            "print(Holdout_{nonce_fn}(17).step(9))\n"
        ),
        (
            "def zigzag_{nonce_fn}(n, tag='{nonce}'):\n"
            "    grid = []\n"
            "    for r in range(n):\n"
            "        row = [(r - c) * (c + 3) for c in range(n)]\n"
            "        grid.append(row[::-1] if r % 2 else row)\n"
            "    return tag, grid\n"
            "print(zigzag_{nonce_fn}(6))\n"
        ),
    ]
    out = []
    for i, _src in enumerate(member_texts):
        nonce = hashlib.sha1(f"samia-holdout-{i}-{rng.random()}".encode()).hexdigest()[:12]
        nonce_fn = "n" + nonce[:8]
        tmpl = templates[i % len(templates)]
        text = tmpl.format(nonce=nonce, nonce_fn=nonce_fn)
        out.append(
            {
                "id": f"synth_nonmember_{i}_{nonce}",
                "text": text,
                "date": None,
                "title": "synthetic_holdout",
                "origin": "synthetic_nonmember",
                "label": 0,
            }
        )
    return out


def extract_notebook_python(nb_path: Path) -> str:
    """Read an .ipynb and concatenate Python code cells. Does not write back."""
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = src.strip()
        if not src:
            continue
        # Keep code; skip IPython magics / shell.
        lines = []
        for line in src.splitlines():
            st = line.strip()
            if st.startswith("!") or st.startswith("%"):
                continue
            lines.append(line)
        body = "\n".join(lines).strip()
        if body:
            chunks.append(body)
    return "\n\n".join(chunks)


def chunk_code_windows(text: str, text_length: int, max_chunks: int = 10) -> list[str]:
    """Split long notebooks into overlapping word windows for SAMIA scoring."""
    words = tokenize_words(text)
    if len(words) < 12:
        return []
    if len(words) <= text_length:
        return [" ".join(words)]
    step = max(text_length // 2, 12)
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + text_length]
        if len(window) < 12:
            break
        chunks.append(" ".join(window))
        if len(chunks) >= max_chunks:
            break
    return chunks


def infer_student_group(path: Path, submissions_dir: Path, notebook_groups: dict[str, str]) -> str:
    rel_parts = path.relative_to(submissions_dir).parts
    if len(rel_parts) >= 2:
        top = rel_parts[0]
        m = re.search(r"group\s*(\d+)", top, re.I)
        if m:
            return f"Group {int(m.group(1))}"
        if "group" in top.lower():
            return top
    if path.name in notebook_groups:
        return notebook_groups[path.name]
    m = re.search(r"(\d+)", path.stem)
    if m:
        return f"Group {int(m.group(1))}"
    return "Group 1"


def load_student_samples(
    submissions_dir: Path | None,
    text_length: int,
    max_chunks_per_file: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Recursively load student/group Python and titanic_analysis notebooks.

    Membership: student coursework is treated as *non-member candidates*
    (private / recent classroom work). ROC metrics pair each group against
    Stack Overflow putative members (cross-domain proxy).
    """
    meta = {
        "source_path": str(submissions_dir) if submissions_dir else None,
        "available": False,
        "n_files": 0,
        "notebooks": [],
        "groups": [],
        "warning": "",
    }
    if submissions_dir is None or not submissions_dir.is_dir():
        meta["warning"] = (
            "Folder Project_Submissions was not found. Student-assignment "
            "evaluation is skipped."
        )
        return [], meta

    skip_names = {"qa.en.python.json"}
    skip_suffix = {".json", ".jsonl", ".csv", ".md", ".txt", ".zip"}
    notebooks = sorted(
        p for p in submissions_dir.rglob("*.ipynb")
        if "titanic_analysis" in p.name.lower() or True
    )
    # Prefer titanic_analysis notebooks; if none, all notebooks.
    titanic = [p for p in notebooks if "titanic_analysis" in p.name.lower()]
    ordered = titanic if titanic else notebooks
    notebook_groups = {p.name: f"Group {i}" for i, p in enumerate(ordered, 1)}

    samples = []
    for path in sorted(submissions_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in skip_names or path.suffix.lower() in skip_suffix:
            continue
        rel = str(path.relative_to(submissions_dir))
        text = ""
        kind = ""
        if path.suffix.lower() == ".ipynb":
            try:
                text = extract_notebook_python(path)
            except Exception as e:
                print(f"  Could not parse notebook {rel}: {e}")
                continue
            kind = "titanic_notebook" if "titanic_analysis" in path.name.lower() else "notebook"
            if "titanic_analysis" in path.name.lower():
                meta["notebooks"].append(rel)
        elif path.suffix.lower() == ".py":
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            kind = "python"
        else:
            continue
        windows = chunk_code_windows(text, text_length, max_chunks=max_chunks_per_file)
        if not windows:
            continue
        group = infer_student_group(path, submissions_dir, notebook_groups)
        for ci, win in enumerate(windows):
            samples.append(
                {
                    "id": f"student_{group.replace(' ', '')}_{path.stem}_{ci}",
                    "text": win,
                    "date": None,
                    "title": rel,
                    "origin": "student_assignment",
                    "group": group,
                    "kind": kind,
                    "label": 0,
                }
            )
        meta["n_files"] += 1

    meta["groups"] = sorted({s["group"] for s in samples}, key=lambda g: (len(g), g))
    meta["available"] = bool(samples)
    if not samples:
        meta["warning"] = (
            f"No usable Python/notebook code under {submissions_dir}."
        )
    else:
        meta["warning"] = (
            "Each student group is scored as non-member vs Stack Overflow "
            "putative members (cross-domain proxy). Original files are not modified. "
            f"Groups: {meta['groups']}"
        )
    return samples, meta


def load_fallback_human_python(max_samples: int, text_length: int, seed: int) -> list[dict[str, Any]]:
    """Repo-local human Python (dataset/human) used only if SO file is missing."""
    human_dir = ROOT / "dataset" / "human"
    if not human_dir.is_dir():
        return []
    files = sorted(human_dir.glob("*.py"))
    rng = random.Random(seed)
    rng.shuffle(files)
    out = []
    for p in files[:max_samples]:
        text = truncate_to_word_length(p.read_text(encoding="utf-8", errors="replace"), text_length)
        if len(tokenize_words(text)) < 12:
            continue
        out.append(
            {
                "id": f"human_{p.stem}",
                "text": text,
                "date": None,
                "title": p.name,
                "origin": "dataset/human_fallback",
                "label": 1,
            }
        )
    return out


# ===========================================================================
# 4. Model backends (Hugging Face local + OpenRouter APIs)
# ===========================================================================

def get_openrouter_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if key and key.strip() not in ("", "PUT YOUR API KEY", "PUT_API"):
        return key.strip()
    if OPENROUTER_API_KEY_DEFAULT.strip():
        return OPENROUTER_API_KEY_DEFAULT.strip()
    return None


def cache_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace"))
        h.update(b"|")
    return h.hexdigest()[:24]


def cache_load(key: str) -> Any | None:
    path = CACHE_DIR / f"{key}.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def cache_save(key: str, obj: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


class HuggingFaceSampler:
    def __init__(self, model_id: str, display_name: str, max_new_tokens: int, paper_max_length: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.display_name = display_name
        self.max_new_tokens = max_new_tokens
        self.paper_max_length = paper_max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading {display_name} ({model_id}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def generate_n(self, prompt: str, n: int) -> list[str]:
        import torch

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        gen_kwargs = dict(
            do_sample=True,
            temperature=SAMIA_TEMPERATURE,
            top_k=SAMIA_TOP_K,
            top_p=SAMIA_TOP_P,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.paper_max_length:
            gen_kwargs["max_length"] = 1024
        else:
            gen_kwargs["max_new_tokens"] = self.max_new_tokens

        outs = []
        for _ in range(n):
            with torch.no_grad():
                tokens = self.model.generate(**enc, **gen_kwargs)
            # Official decoding is the *full* sequence (prompt+completion).
            text = self.tokenizer.decode(tokens[0], skip_special_tokens=False)
            outs.append(text)
        return outs

    def close(self):
        del self.model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class OpenRouterSampler:
    def __init__(self, slug: str, display_name: str, api_key: str, max_new_tokens: int):
        self.slug = slug
        self.display_name = display_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.unavailable_reason: str | None = None

    def generate_n(self, prompt: str, n: int) -> list[str]:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "SAMIA-Journal",
        }
        # Prefix continuation: the prefix is the user message (no extra instruction
        # that would change the distribution away from the model's own sampling).
        payload = {
            "model": self.slug,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": SAMIA_TEMPERATURE,
            "top_p": SAMIA_TOP_P,
            "max_tokens": self.max_new_tokens,
            "n": 1,
        }
        outs = []
        for i in range(n):
            last_err = None
            for attempt in range(4):
                try:
                    r = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=90,
                    )
                except Exception as e:
                    last_err = str(e)
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code == 200:
                    data = r.json()
                    choices = data.get("choices") or []
                    if not choices:
                        last_err = "empty choices"
                        break
                    msg = choices[0].get("message") or {}
                    content = msg.get("content") or ""
                    # Reconstruct full text ≈ prompt + continuation (SAMIA eval
                    # then drops the prefix via get_suffix).
                    full = prompt + " " + str(content)
                    outs.append(full)
                    last_err = None
                    break
                if r.status_code == 429:
                    wait = 20 * (attempt + 1)
                    print(f"    OpenRouter 429 on {self.slug}; wait {wait}s")
                    time.sleep(wait)
                    last_err = "rate limited"
                    continue
                if r.status_code in (401, 402, 403, 404):
                    last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                    self.unavailable_reason = last_err
                    raise RuntimeError(last_err)
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(2 ** attempt)
            if last_err and len(outs) == i:
                raise RuntimeError(last_err or "generation failed")
            time.sleep(0.4)
        return outs

    def close(self):
        return


def probe_openrouter_models(api_key: str) -> set[str]:
    import requests

    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"  OpenRouter model catalog HTTP {r.status_code}")
            return set()
        ids = {m.get("id") for m in r.json().get("data", []) if m.get("id")}
        return ids
    except Exception as e:
        print(f"  Could not list OpenRouter models: {e}")
        return set()


# ===========================================================================
# 5. Experiment runner
# ===========================================================================

def score_dataset(
    sampler,
    display_name: str,
    samples: list[dict[str, Any]],
    num_samples: int,
    prefix_ratio: float,
    text_length: int,
    use_zlib: bool,
) -> list[dict[str, Any]]:
    rows = []
    for i, ex in enumerate(samples, 1):
        prefix = get_prefix(ex["text"], prefix_ratio)
        ck = cache_key(
            display_name,
            ex["id"],
            prefix,
            num_samples,
            SAMIA_TEMPERATURE,
            SAMIA_TOP_K,
            SAMIA_TOP_P,
        )
        cached = cache_load(ck)
        if cached and len(cached.get("generations", [])) >= num_samples:
            gens = cached["generations"][:num_samples]
            print(f"    [{i}/{len(samples)}] {ex['id']} (cache hit)")
        else:
            print(f"    [{i}/{len(samples)}] {ex['id']} sampling n={num_samples}")
            try:
                gens = sampler.generate_n(prefix, num_samples)
            except Exception as e:
                err = str(e)
                print(f"      generation failed: {e}")
                if any(x in err for x in ("HTTP 401", "HTTP 402", "HTTP 403", "HTTP 404")):
                    print("      Aborting this model (auth/availability error).")
                    remaining = [
                        {
                            "id": ex2["id"],
                            "label": ex2["label"],
                            "origin": ex2.get("origin"),
                            "samia_score": None,
                            "error": err,
                            "model": display_name,
                        }
                        for ex2 in samples[i - 1 :]
                    ]
                    rows.extend(remaining)
                    return rows
                rows.append(
                    {
                        "id": ex["id"],
                        "label": ex["label"],
                        "origin": ex.get("origin"),
                        "samia_score": None,
                        "error": err,
                        "model": display_name,
                    }
                )
                continue
            cache_save(ck, {"generations": gens, "prefix": prefix})
        scored = samia_score_from_generations(
            ex["text"], gens, prefix_ratio, text_length, display_name, use_zlib
        )
        rows.append(
            {
                "id": ex["id"],
                "label": ex["label"],
                "origin": ex.get("origin"),
                "title": ex.get("title"),
                "samia_score": scored["samia_score"],
                "rouge_per_sample": scored["rouge_per_sample"],
                "n_valid_generations": scored["n_valid_generations"],
                "model": display_name,
                "prefix": prefix[:200],
            }
        )
    return rows


def save_raw(track: str, model_name: str, rows: list[dict]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    path = RAW_DIR / f"{track}_{safe}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def plot_roc(track: str, per_model_metrics: dict[str, dict], title: str) -> Path | None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    drew = False
    for name, m in per_model_metrics.items():
        if not m.get("fpr") or m.get("auc") is None:
            continue
        ax.plot(m["fpr"], m["tpr"], label=f"{name} (AUC={m['auc']:.3f})")
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.axvline(0.05, color="gray", ls=":", alpha=0.7, label="5% FPR")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    path = PLOT_DIR / f"roc_{track}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_table(title: str, rows: list[dict[str, Any]], name_key: str = "model") -> None:
    print("\n" + title)
    label = "Group" if name_key == "group" else "Model"
    header = f"{label:<28} {'AUC':>8} {'TPR@5%FPR':>10} {'Precision':>10} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r.get(name_key, '')):<28} {format_metric(r.get('auc')):>8} "
            f"{format_tpr_percent(r.get('tpr_at_5fpr')):>10} "
            f"{format_metric(r.get('precision')):>10} {format_metric(r.get('f1')):>8}"
        )


def run_track(
    track_name: str,
    samples: list[dict[str, Any]],
    models_plan: list[dict[str, Any]],
    args,
    extra_samples: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict]]]:
    table_rows = []
    roc_store = {}
    extra_by_model: dict[str, list[dict]] = {}
    if not samples:
        for spec in models_plan:
            table_rows.append(
                {
                    "model": spec["name"],
                    "auc": None,
                    "tpr_at_5fpr": None,
                    "precision": None,
                    "f1": None,
                    "status": "skipped_no_data",
                }
            )
        return table_rows, extra_by_model

    labels_present = {s["label"] for s in samples}
    if labels_present != {0, 1}:
        print(
            f"  Track {track_name}: labels={labels_present}. "
            "ROC/AUC/TPR/Precision/F1 require both member (1) and non-member (0)."
        )

    for spec in models_plan:
        name = spec["name"]
        status = spec.get("status", "ok")
        reason = spec.get("reason", "")
        row = {
            "model": name,
            "auc": None,
            "tpr_at_5fpr": None,
            "precision": None,
            "f1": None,
            "status": status,
            "reason": reason,
        }
        if status != "ok":
            print(f"\n  -- {name}: SKIPPED ({reason})")
            table_rows.append(row)
            continue
        print(f"\n  -- {name}")
        sampler = None
        try:
            sampler = spec["factory"]()
            scored = score_dataset(
                sampler,
                name,
                samples,
                args.num_samples,
                args.prefix_ratio,
                args.text_length,
                args.zlib,
            )
            save_raw(track_name, name, scored)
            y = [r["label"] for r in scored if r.get("samia_score") is not None]
            s = [r["samia_score"] for r in scored if r.get("samia_score") is not None]
            metrics = compute_metrics(y, s, fpr_target=0.05)
            row.update(
                {
                    "auc": metrics["auc"],
                    "tpr_at_5fpr": metrics["tpr_at_5fpr"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "n": metrics["n"],
                    "n_member": metrics["n_member"],
                    "n_nonmember": metrics["n_nonmember"],
                    "note": metrics.get("note", ""),
                    "status": "ok",
                }
            )
            roc_store[name] = metrics
            print(
                f"     AUC={format_metric(row['auc'])}  "
                f"TPR@5%FPR={format_tpr_percent(row['tpr_at_5fpr'])}  "
                f"P={format_metric(row['precision'])}  F1={format_metric(row['f1'])}"
            )
            if extra_samples:
                print(f"     scoring {len(extra_samples)} student windows...")
                extra_by_model[name] = score_dataset(
                    sampler,
                    name,
                    extra_samples,
                    args.num_samples,
                    args.prefix_ratio,
                    args.text_length,
                    args.zlib,
                )
                save_raw("student_windows", name, extra_by_model[name])
        except Exception as e:
            row["status"] = "error"
            row["reason"] = str(e)
            print(f"     ERROR: {e}")
            traceback.print_exc()
        finally:
            if sampler is not None:
                sampler.close()
        table_rows.append(row)

    plot_roc(track_name, roc_store, f"SAMIA ROC — {track_name}")
    return table_rows, extra_by_model


def build_model_plan(args) -> list[dict[str, Any]]:
    plan = []
    requested = args.models
    or_key = get_openrouter_key()
    catalog: set[str] = set()
    needs_api = any(n in requested for n in OPENROUTER_SLUGS)
    if needs_api and or_key:
        print("Probing OpenRouter model catalog...")
        catalog = probe_openrouter_models(or_key)
        if catalog:
            print(f"  {len(catalog)} models visible on this API key.")
    elif needs_api:
        print("OPENROUTER_API_KEY not set (and no key found in ectel.py).")

    hf_wanted = [n for n in HF_MODELS if n in requested]
    api_wanted = [n for n in OPENROUTER_SLUGS if n in requested]

    for name in hf_wanted:
        mid = HF_MODELS[name]
        plan.append(
            {
                "name": name,
                "kind": "huggingface",
                "status": "ok",
                "factory": lambda mid=mid, name=name: HuggingFaceSampler(
                    mid, name, args.max_new_tokens, args.paper_max_length
                ),
            }
        )

    for name in api_wanted:
        slug = OPENROUTER_SLUGS[name]
        if not or_key:
            plan.append(
                {
                    "name": name,
                    "kind": "openrouter",
                    "status": "skipped",
                    "reason": "No OpenRouter API key (set OPENROUTER_API_KEY).",
                }
            )
            continue
        if catalog and slug not in catalog:
            # Do not silently swap. Record the mismatch; still attempt the
            # requested slug once, because catalog filters can hide models.
            print(
                f"  Note: {slug} was not in the catalog listing; will still "
                f"query the requested id (no substitute model)."
            )
        plan.append(
            {
                "name": name,
                "kind": "openrouter",
                "status": "ok",
                "factory": lambda slug=slug, name=name, key=or_key: OpenRouterSampler(
                    slug, name, key, args.max_new_tokens
                ),
            }
        )
    return plan


def _merge_table_rows(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Keep prior model results when a later run covers only a subset of models."""
    by_name = {r.get("model"): r for r in existing if r.get("model")}
    for r in incoming:
        name = r.get("model")
        if not name:
            continue
        prev = by_name.get(name)
        incoming_has_metrics = r.get("auc") is not None
        if incoming_has_metrics or prev is None or prev.get("auc") is None:
            by_name[name] = r
    order = MODEL_TABLE_ORDER
    rest = [n for n in by_name if n not in order]
    return [by_name[n] for n in order + rest if n in by_name]


def evaluate_student_groups(
    so_samples: list[dict[str, Any]],
    st_samples: list[dict[str, Any]],
    extra_by_model: dict[str, list[dict]],
) -> list[dict[str, Any]]:
    """
    Table 3: one row per student group.
    Members = SO putative members (Track A scores).
    Non-members = that group's code windows.
    Sample score = mean SAMIA across models that produced a score.
    """
    members = [s for s in so_samples if s.get("label") == 1]
    if not members or not st_samples:
        return []

    per_model_by_id: dict[str, dict[str, float]] = {}

    # Track A member/non-member scores
    for path in RAW_DIR.glob("stackoverflow_*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("samia_score") is None:
                continue
            per_model_by_id.setdefault(row["id"], {})[row.get("model", path.stem)] = float(row["samia_score"])

    for name, rows in extra_by_model.items():
        for row in rows:
            if row.get("samia_score") is None:
                continue
            per_model_by_id.setdefault(row["id"], {})[name] = float(row["samia_score"])

    def mean_score(sample_id: str) -> float | None:
        vals = list(per_model_by_id.get(sample_id, {}).values())
        return float(np.mean(vals)) if vals else None

    groups = sorted({s["group"] for s in st_samples}, key=lambda g: (len(g), g))
    table = []
    for group in groups:
        g_samples = [s for s in st_samples if s["group"] == group]
        y, s = [], []
        for ex in members:
            sc = mean_score(ex["id"])
            if sc is not None:
                y.append(1)
                s.append(sc)
        for ex in g_samples:
            sc = mean_score(ex["id"])
            if sc is not None:
                y.append(0)
                s.append(sc)
        metrics = compute_metrics(y, s, fpr_target=0.05)
        table.append(
            {
                "group": group,
                "auc": metrics["auc"],
                "tpr_at_5fpr": metrics["tpr_at_5fpr"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "n": metrics["n"],
                "n_member": metrics["n_member"],
                "n_nonmember": metrics["n_nonmember"],
                "status": "ok" if metrics["auc"] is not None else "insufficient",
                "note": metrics.get("note", ""),
            }
        )
        print(
            f"  {group}: AUC={format_metric(table[-1]['auc'])}  "
            f"TPR@5%FPR={format_tpr_percent(table[-1]['tpr_at_5fpr'])}  "
            f"P={format_metric(table[-1]['precision'])}  "
            f"F1={format_metric(table[-1]['f1'])}"
        )
    return table


def write_tables(so_rows, st_rows, extra_meta: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    def dump(name, rows, id_col: str):
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        cols = [id_col, "auc", "tpr_at_5fpr", "precision", "f1", "status", "reason", "n", "note"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        keep = [c for c in cols if c in df.columns]
        df = df[keep]
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
        (TABLE_DIR / f"{name}.json").write_text(
            df.to_json(orient="records", indent=2), encoding="utf-8"
        )

    dump("stackoverflow", so_rows, "model")
    dump("student_assignments", st_rows, "group" if st_rows and "group" in st_rows[0] else "model")
    (TABLE_DIR / "run_meta.json").write_text(
        json.dumps(extra_meta, indent=2, default=str), encoding="utf-8"
    )


def parse_args():
    p = argparse.ArgumentParser(description="SAMIA membership inference for Python code")
    p.add_argument("--so_path", type=str, default="", help="Path to qa.en.python.json")
    p.add_argument("--students_dir", type=str, default="", help="Path to Project_submissiions")
    p.add_argument("--text_length", type=int, default=64, help="WikiMIA-style word cap (32/64/128/256)")
    p.add_argument("--num_samples", type=int, default=3, help="Generations per example (paper default 10)")
    p.add_argument("--prefix_ratio", type=float, default=0.5)
    p.add_argument("--max_samples", type=int, default=20, help="Per class")
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--paper_max_length", action="store_true", help="Use official max_length=1024")
    p.add_argument("--zlib", action="store_true", help="Official zlib-scaled ROUGE variant")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--models",
        nargs="+",
        default=MODEL_TABLE_ORDER,
    )
    p.add_argument(
        "--hf_only",
        action="store_true",
        help="Skip API models (useful for a CPU-only Cursor run)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="DistilGPT-2 only, 6+6 examples, 2 generations (smoke test)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.models = ["DistilGPT-2"]
        args.max_samples = 6
        args.num_samples = 2
        args.max_new_tokens = 40
        args.text_length = 48
    if args.hf_only:
        args.models = [m for m in args.models if m in HF_MODELS]

    print("=" * 72)
    print("SAMIA — Sampling-based Pseudo-Likelihood (Kaneko et al., 2024)")
    print("https://github.com/nlp-titech/samia")
    print("=" * 72)
    print(
        f"Config: text_length={args.text_length} words, prefix_ratio={args.prefix_ratio}, "
        f"N_gen={args.num_samples}, max_per_class={args.max_samples}, zlib={args.zlib}"
    )
    print(
        "Score: mean ROUGE-1 recall(generated suffix, original suffix). "
        "Higher => more member-like."
    )

    so_path = Path(args.so_path) if args.so_path else find_file("qa.en.python.json")
    st_dir = Path(args.students_dir) if args.students_dir else find_dir("Project_Submissions")

    so_samples, so_meta = load_stackoverflow_samples(
        so_path, args.max_samples, args.text_length, args.seed
    )
    print("\n[Stack Overflow]")
    print(f"  path={so_meta['source_path']}  available={so_meta['available']}")
    print(f"  fields={so_meta['fields']}")
    print(f"  protocol={so_meta['label_protocol']}")
    print(f"  {so_meta['warning']}")

    used_fallback = False
    if not so_samples:
        fb = load_fallback_human_python(args.max_samples, args.text_length, args.seed)
        if fb:
            rng = random.Random(args.seed)
            non = make_synthetic_nonmembers([x["text"] for x in fb], rng)
            so_samples = fb + non
            used_fallback = True
            so_meta["label_protocol"] = "fallback_dataset_human_plus_synthetic"
            so_meta["warning"] += (
                " FALLBACK: using dataset/human/*.py as putative members because "
                "qa.en.python.json is missing."
            )
            print("  Using dataset/human fallback.")

    st_samples, st_meta = load_student_samples(st_dir, args.text_length, max_chunks_per_file=8)
    print("\n[Student assignments]")
    print(f"  path={st_meta['source_path']}  n_windows={len(st_samples)}  files={st_meta['n_files']}")
    print(f"  titanic notebooks={st_meta['notebooks']}")
    print(f"  groups={st_meta.get('groups')}")
    print(f"  {st_meta['warning']}")

    plan = build_model_plan(args)
    print("\n[Models]")
    for spec in plan:
        extra = f" — {spec.get('reason')}" if spec.get("status") != "ok" else ""
        print(f"  {spec['name']}: {spec.get('status', 'ok')}{extra}")

    print("\n=== Track A: Stack Overflow Python (qa.en.python.json) ===")
    so_rows, extra_by_model = run_track(
        "stackoverflow", so_samples, plan, args, extra_samples=st_samples or None
    )
    if used_fallback:
        for r in so_rows:
            r["note"] = (r.get("note") or "") + " [FALLBACK corpus]"

    print("\n=== Track B: Student groups (Table 3) ===")
    if st_samples and any(s.get("label") == 1 for s in so_samples):
        st_rows = evaluate_student_groups(so_samples, st_samples, extra_by_model)
    else:
        st_rows = []
        print("  Skipped: need student notebooks and SO members.")

    so_by = {r.get("model"): r for r in so_rows}
    so_rows = [so_by[n] for n in MODEL_TABLE_ORDER if n in so_by] + [
        r for r in so_rows if r.get("model") not in MODEL_TABLE_ORDER
    ]

    write_tables(so_rows, st_rows, {"so": so_meta, "student": st_meta, "args": vars(args)})

    print_table("A. Stack Overflow Python Dataset" + (" (FALLBACK)" if used_fallback else ""), so_rows)
    print_table("B. Student group submissions", st_rows, name_key="group")
    print(f"\nRaw scores: {RAW_DIR}")
    print(f"Tables:     {TABLE_DIR}")
    print(f"Plots:      {PLOT_DIR}")
    print("Done. Re-run with cached generations to recompute metrics without new queries.")


if __name__ == "__main__":
    main()

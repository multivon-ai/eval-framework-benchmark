"""
Deterministic RAGTruth loader.

RAGTruth (Niu et al., 2024) is a real RAG-trace hallucination dataset:
human annotators marked hallucination spans inside model-generated
summaries with the source document attached. The labels are
fine-grained (Evident Conflict, Subtle Conflict, Evident Introduction
of Baseless Info, etc.), but for binary faithfulness comparison we
collapse to ``faithful`` (no labels) or ``hallucinated`` (any label).

The 200-case pilot is drawn with seed=42 from the Summary task's test
split. Balanced 100/100 by construction. The selection is
deterministic across machines, and the order is fixed so all three
frameworks see the same cases in the same sequence.

Why RAGTruth: the v1 benchmark used HaluEval Summarization, which
multivon-eval's calibration was measured on (the circularity caveat
in the live blog post). RAGTruth is a dataset multivon's threshold
table has never seen — so the cross-dataset comparison is honest.
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from data.loader import Case


DataDir = Path(__file__).resolve().parent

RESPONSES_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"
SOURCES_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"

RESPONSES_FULL = DataDir / "ragtruth_responses_full.jsonl"
SOURCES_FULL = DataDir / "ragtruth_sources_full.jsonl"


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as r, dest.open("wb") as f:
        f.write(r.read())


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_ragtruth_summary(n: int = 200, *, seed: int = 42,
                          refresh_pilot: bool = False) -> list[Case]:
    """Return the deterministic n-case RAGTruth Summary test pilot.

    First call downloads the full files (~36 MB total) into ``data/``
    and caches a balanced n-case JSON. Subsequent calls read the cache.
    """
    if n % 2:
        raise ValueError(f"n must be even (got {n})")
    pilot_path = DataDir / f"ragtruth_sum_pilot_{n}.json"
    if pilot_path.exists() and not refresh_pilot:
        return [Case(**row) for row in json.loads(pilot_path.read_text())]

    # If a larger cached pilot exists, slice deterministically from it
    # (cheaper than re-downloading + re-sampling for sub-sizes).
    for candidate in (200, 500, 1000):
        if candidate > n:
            larger = DataDir / f"ragtruth_sum_pilot_{candidate}.json"
            if larger.exists():
                rows = json.loads(larger.read_text())[:n]
                pilot_path.write_text(json.dumps(rows, indent=2))
                return [Case(**row) for row in rows]

    _download_if_missing(RESPONSES_URL, RESPONSES_FULL)
    _download_if_missing(SOURCES_URL, SOURCES_FULL)

    # Build source lookup.
    sources: dict[str, dict] = {}
    for row in _read_jsonl(SOURCES_FULL):
        if row.get("task_type") == "Summary":
            sources[str(row["source_id"])] = row

    # Filter responses: Summary task, test split, with a usable source.
    test_summaries: list[tuple[dict, dict]] = []
    for resp in _read_jsonl(RESPONSES_FULL):
        sid = str(resp.get("source_id"))
        src = sources.get(sid)
        if src is None:
            continue
        if resp.get("split") != "test":
            continue
        test_summaries.append((resp, src))

    faithful = [(r, s) for r, s in test_summaries if not r.get("labels")]
    hallucinated = [(r, s) for r, s in test_summaries if r.get("labels")]

    rng = random.Random(seed)
    half = n // 2
    if len(faithful) < half or len(hallucinated) < half:
        raise RuntimeError(
            f"Insufficient cases: need {half}+{half} but have "
            f"{len(faithful)} faithful and {len(hallucinated)} hallucinated"
        )
    picked_f = rng.sample(faithful, half)
    picked_h = rng.sample(hallucinated, half)

    cases: list[Case] = []
    for resp, src in picked_f:
        cases.append(Case(
            id=f"ragtruth_sum_{resp['id']}_faithful",
            task="sum",
            context=_source_text(src),
            question="",
            answer=resp["response"],
            label="faithful",
            source_index=int(resp["id"]),
        ))
    for resp, src in picked_h:
        cases.append(Case(
            id=f"ragtruth_sum_{resp['id']}_hallucinated",
            task="sum",
            context=_source_text(src),
            question="",
            answer=resp["response"],
            label="hallucinated",
            source_index=int(resp["id"]),
        ))

    # Sort by id for deterministic display order.
    cases.sort(key=lambda c: c.id)

    pilot_path.write_text(json.dumps([c.to_dict() for c in cases], indent=2))
    return cases


def _source_text(src: dict) -> str:
    """Extract the source document text from a RAGTruth source record.

    Quirk: ``source`` is the *dataset name* (e.g. "CNN/DM"), not the
    article body. The article body lives in ``source_info`` (a string
    for Summary tasks, a dict for QA/DataToText). For Summary tasks
    we read ``source_info`` directly; for the other task types we fall
    back to the ``prompt`` (which has the body inlined after the
    instruction line).
    """
    info = src.get("source_info")
    if isinstance(info, str) and info.strip():
        return info
    # Some task types store source_info as a dict; flatten if so.
    if isinstance(info, dict):
        for k in ("article", "passage", "context", "document", "text"):
            if k in info and isinstance(info[k], str):
                return info[k]
    # Last resort: the prompt body. Strip the "Summarize this:" prefix
    # by splitting on the first double newline.
    prompt = src.get("prompt") or ""
    if isinstance(prompt, str) and "\n" in prompt:
        return prompt.split("\n", 1)[1].strip()
    return prompt if isinstance(prompt, str) else ""


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    cases = load_ragtruth_summary(n=args.n, seed=args.seed,
                                  refresh_pilot=args.refresh)
    n_f = sum(1 for c in cases if c.label == "faithful")
    n_h = sum(1 for c in cases if c.label == "hallucinated")
    print(f"RAGTruth Summary test: {len(cases)} cases  "
          f"({n_f} faithful / {n_h} hallucinated)")
    print(f"  Example faithful: {next(c for c in cases if c.label == 'faithful').answer[:120]}...")

"""
Deterministic HaluEval loader.

HaluEval ships each source row with both a faithful and a hallucinated
candidate (QA: ``right_answer`` / ``hallucinated_answer``;
Summarization: ``right_summary`` / ``hallucinated_summary``). We expand
each row into two test cases — one labeled "faithful" and one labeled
"hallucinated" — so the resulting dataset is balanced by construction.

The 100-case pilot is drawn with seed=42 from this expanded pool. The
selection is deterministic across machines; the order is also fixed so
all three frameworks see the same cases in the same sequence.

Cached datasets live in ``data/`` and are git-ignored above 1 MB.
"""
from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal

DataDir = Path(__file__).resolve().parent

QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
SUM_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json"

QA_FULL = DataDir / "halueval_qa_full.json"
SUM_FULL = DataDir / "halueval_sum_full.json"

QA_PILOT = DataDir / "halueval_qa_pilot_100.json"
SUM_PILOT = DataDir / "halueval_sum_pilot_100.json"


Task = Literal["qa", "sum"]


@dataclass(frozen=True)
class Case:
    id: str
    task: Task
    context: str
    question: str  # empty for sum tasks
    answer: str
    label: Literal["faithful", "hallucinated"]
    source_index: int  # row in the full HaluEval file we came from

    def to_dict(self) -> dict:
        return asdict(self)


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as f:
        f.write(resp.read())


def _read_jsonl(path: Path) -> Iterable[dict]:
    """HaluEval's data files are NDJSON despite the ``.json`` extension."""
    with path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


def _expand_qa(rows: Iterable[dict]) -> list[Case]:
    """Each row yields one faithful + one hallucinated case."""
    out: list[Case] = []
    for i, row in enumerate(rows):
        ctx = row["knowledge"]
        q = row["question"]
        out.append(Case(
            id=f"qa_{i:05d}_faithful",
            task="qa",
            context=ctx,
            question=q,
            answer=row["right_answer"],
            label="faithful",
            source_index=i,
        ))
        out.append(Case(
            id=f"qa_{i:05d}_hallucinated",
            task="qa",
            context=ctx,
            question=q,
            answer=row["hallucinated_answer"],
            label="hallucinated",
            source_index=i,
        ))
    return out


def _expand_sum(rows: Iterable[dict]) -> list[Case]:
    out: list[Case] = []
    for i, row in enumerate(rows):
        doc = row["document"]
        out.append(Case(
            id=f"sum_{i:05d}_faithful",
            task="sum",
            context=doc,
            question="",
            answer=row["right_summary"],
            label="faithful",
            source_index=i,
        ))
        out.append(Case(
            id=f"sum_{i:05d}_hallucinated",
            task="sum",
            context=doc,
            question="",
            answer=row["hallucinated_summary"],
            label="hallucinated",
            source_index=i,
        ))
    return out


def _stratified_sample(cases: list[Case], n: int, seed: int = 42) -> list[Case]:
    """Half faithful, half hallucinated, deterministic order."""
    if n % 2:
        raise ValueError(f"n must be even (got {n}) so the sample is balanced")
    faithful = [c for c in cases if c.label == "faithful"]
    hallucinated = [c for c in cases if c.label == "hallucinated"]
    rng = random.Random(seed)
    picked_f = rng.sample(faithful, n // 2)
    picked_h = rng.sample(hallucinated, n // 2)
    # Sort by id for deterministic display order; rng has already done the
    # statistical work of picking which cases.
    merged = sorted(picked_f + picked_h, key=lambda c: c.id)
    return merged


def load_qa(n: int = 100, *, seed: int = 42, refresh_pilot: bool = False) -> list[Case]:
    """Return the deterministic n-case HaluEval-QA pilot subset.

    First call downloads the full file (~5 MB) and writes a deterministic
    pilot subset to ``data/halueval_qa_pilot_{n}.json``.
    """
    pilot_path = DataDir / f"halueval_qa_pilot_{n}.json"
    if pilot_path.exists() and not refresh_pilot:
        return [Case(**row) for row in json.loads(pilot_path.read_text())]
    _download_if_missing(QA_URL, QA_FULL)
    full = _expand_qa(_read_jsonl(QA_FULL))
    pilot = _stratified_sample(full, n=n, seed=seed)
    pilot_path.write_text(json.dumps([c.to_dict() for c in pilot], indent=2))
    return pilot


def load_sum(n: int = 100, *, seed: int = 42, refresh_pilot: bool = False) -> list[Case]:
    """Return the deterministic n-case HaluEval-Summarization pilot subset."""
    pilot_path = DataDir / f"halueval_sum_pilot_{n}.json"
    if pilot_path.exists() and not refresh_pilot:
        return [Case(**row) for row in json.loads(pilot_path.read_text())]
    _download_if_missing(SUM_URL, SUM_FULL)
    full = _expand_sum(_read_jsonl(SUM_FULL))
    pilot = _stratified_sample(full, n=n, seed=seed)
    pilot_path.write_text(json.dumps([c.to_dict() for c in pilot], indent=2))
    return pilot


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["qa", "sum", "both"], default="both")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    if args.task in ("qa", "both"):
        qa = load_qa(n=args.n, seed=args.seed, refresh_pilot=args.refresh)
        print(f"qa: {len(qa)} cases, {sum(c.label == 'faithful' for c in qa)} faithful / "
              f"{sum(c.label == 'hallucinated' for c in qa)} hallucinated")
    if args.task in ("sum", "both"):
        sm = load_sum(n=args.n, seed=args.seed, refresh_pilot=args.refresh)
        print(f"sum: {len(sm)} cases, {sum(c.label == 'faithful' for c in sm)} faithful / "
              f"{sum(c.label == 'hallucinated' for c in sm)} hallucinated")

"""
Benchmark orchestrator.

Runs each framework over the same case set ``runs`` times. Results are
streamed to ``results/raw/{framework}_{task}_run{i}.jsonl`` so a partial
run can be inspected (or resumed after a crash).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from data.loader import Case, load_qa, load_sum
from frameworks.base import FrameworkResult, FrameworkRunner


def _safe_runner_factories(judge_model: str) -> list[tuple[str, Callable[[], FrameworkRunner]]]:
    """Return import-safe factories so a missing optional dep doesn't kill
    the whole run."""

    def _mk_multivon() -> FrameworkRunner:
        from frameworks.multivon import MultivonFaithfulness
        return MultivonFaithfulness(judge_model=judge_model)

    def _mk_deepeval() -> FrameworkRunner:
        from frameworks.deepeval import DeepEvalFaithfulness
        return DeepEvalFaithfulness(judge_model=judge_model)

    def _mk_ragas() -> FrameworkRunner:
        from frameworks.ragas import RagasFaithfulness
        return RagasFaithfulness(judge_model=judge_model)

    return [
        ("multivon-eval", _mk_multivon),
        ("deepeval", _mk_deepeval),
        ("ragas", _mk_ragas),
    ]


def _stream_run(
    runner: FrameworkRunner,
    cases: list[Case],
    out_path: Path,
    *,
    workers: int,
    progress_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(runner.run, c): c for c in cases}
        for i, fut in enumerate(as_completed(futures), start=1):
            result: FrameworkResult = fut.result()
            f.write(json.dumps(result.to_dict()) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(cases):
                err_flag = " (last had error)" if result.error else ""
                print(f"  {progress_label}: {i}/{len(cases)}{err_flag}", file=sys.stderr)


def run_benchmark(
    *,
    task: str,
    n: int,
    runs: int,
    judge_model: str,
    only: list[str] | None,
    workers: int,
    out_dir: Path,
) -> None:
    if task == "qa":
        cases = load_qa(n=n)
    elif task == "sum":
        cases = load_sum(n=n)
    else:
        raise ValueError(task)

    factories = _safe_runner_factories(judge_model)
    if only:
        factories = [(name, fn) for name, fn in factories if name in only]
        if not factories:
            raise SystemExit(f"No matching frameworks in {only!r}")

    for name, factory in factories:
        try:
            runner = factory()
        except ImportError as exc:
            print(f"[skip] {name} not installed ({exc}); pip install -r requirements.txt", file=sys.stderr)
            continue
        for run_idx in range(runs):
            out_path = out_dir / "raw" / f"{name}_{task}_run{run_idx}.jsonl"
            if out_path.exists():
                print(f"[skip] {out_path} already exists; delete to re-run", file=sys.stderr)
                continue
            t0 = time.time()
            _stream_run(
                runner, cases, out_path,
                workers=workers,
                progress_label=f"{name} {task} run {run_idx + 1}/{runs}",
            )
            elapsed = time.time() - t0
            print(f"[done] {name} {task} run {run_idx + 1}/{runs} → {out_path} ({elapsed:.0f}s)", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["qa", "sum", "both"], default="both")
    p.add_argument("--n", type=int, default=100, help="pilot size; must be even")
    p.add_argument("--runs", type=int, default=5, help="repeated runs to measure variance")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--only", nargs="*", default=None, choices=["multivon-eval", "deepeval", "ragas"])
    p.add_argument("--workers", type=int, default=4, help="concurrent judge calls")
    p.add_argument("--out", default="results", help="output directory root")
    p.add_argument("--smoke", action="store_true", help="run on n=4 cases × 1 run for a fast smoke test")
    args = p.parse_args()

    if args.smoke:
        args.n = 4
        args.runs = 1

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        print("WARN: OPENAI_API_KEY not set; framework calls will fail.", file=sys.stderr)

    tasks = ["qa", "sum"] if args.task == "both" else [args.task]
    for t in tasks:
        run_benchmark(
            task=t, n=args.n, runs=args.runs,
            judge_model=args.judge_model, only=args.only,
            workers=args.workers, out_dir=out_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

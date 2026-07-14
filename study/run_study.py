"""Cell-addressed study runner (plan §6, PREREG_ADDENDUM.md §7.1 escalated design).

One invocation runs exactly ONE cell:

    python study/run_study.py --task ragtruth-sum --split test \\
        --framework ragas --judge claude-haiku-4-5 --run 2

Design (the §6 table recomputed in PREREG_ADDENDUM.md §7.1 with RAGTruth-Sum
test escalated to n=500):

    dev   (100 items/task):  R=1, both judges, all 3 tasks      -> 30 cells
    test  ragtruth-sum 500:  R=5, both judges                   -> 50 cells
    test  halueval-sum 300:  R=3 gpt-4o-mini, R=1 claude-haiku  -> 20 cells
    test  halueval-qa  300:  R=3 gpt-4o-mini, R=1 claude-haiku  -> 20 cells
                                                          total   120 cells
                                                                 40,000 evals

The strong-judge ablation (gpt-5.5-2026-04-23, RAGTruth-Sum n=150 subset,
750 evals; addendum §10/§11) is addressed as its own split:

    python study/run_study.py --task ragtruth-sum --split ablation \\
        --framework ragas --judge gpt-5.5 --run 0

Subset rule: study/items/ragtruth-sum_ablation_150.json = the FIRST 150
items of the committed ragtruth-sum_test_500.json (deterministic prefix,
no new sampling decisions; labels stay hidden). Only that exact cell shape
(ragtruth-sum x gpt-5.5 x run 0, all 5 frameworks) is accepted — any other
ablation address, or any non-ablation use of gpt-5.5, is refused. The
record "judge" field and output directory use the series tag "gpt-5.5";
adapters receive the pinned snapshot id gpt-5.5-2026-04-23. Output:
study/runs/raw/gpt-5.5/ragtruth-sum_ablation/{framework}_run0.jsonl.

Output: study/runs/raw/{judge}/{task}_{split}/{framework}_run{R}.jsonl —
one JSON record per item (schema in _record()).

RESUME SEMANTICS — designed against run.py's stale-file failure mode
(run.py skips a whole run file if it merely EXISTS, silently reusing stale
or partial data). Here instead:

  * Output is opened in APPEND mode. On restart the existing file is read,
    every record is VALIDATED against the cell address (task/split/
    framework/judge/run and item-id membership); any mismatch aborts loudly
    (the file is stale or misplaced — never silently reused).
  * Already-recorded item ids are skipped and a loud
    "resuming: N/M done" line is printed. A record with error != null still
    counts as done (errors-as-failures is primary, plan §7 P4); re-running
    errored items is an analysis-time decision, not a silent runner one.
  * A whole cell is never skipped silently: complete cells print
    "cell already complete" and exit 0.
  * Overwriting requires the explicit --fresh flag, which moves the old
    file to a timestamped .bak (never deletes data).
  * --limit N (validation only) stamps "limit": N into every record it
    writes, so limited partial cells are detectable forever.

Blinding: items are loaded ONLY via sample_items.load_study_items()
(study/items/, label sentinel "hidden"). data/labels_hidden/ is never read.

Judge snapshot + tokens + cost: every judge HTTP call is intercepted at the
shared httpx transport (same choke point as study/smoke.py) with THREAD-
LOCAL attribution — each eval runs synchronously in one worker thread
(RAGAS's asyncio.run() loop also lives in the calling thread), so the calls
observed on a thread between eval start and end belong to that eval. The
response-reported model id is recorded as judge_snapshot; provider usage
tokens and pinned-price cost are aggregated per eval. If a framework ever
issues calls from a foreign thread they are simply not attributed (fields
stay null) — attribution never blocks the run.

Determinism knobs: temperature 0 is hardcoded inside every adapter; seed 42
is passed to any adapter whose constructor accepts a ``seed`` kwarg
(inspected at runtime — none of the five pinned adapters currently do, so
the record's "seed" field is null unless an adapter grows support).

Other modes:
    --cell-list      print all 120 cells with pending/partial/done status
    --estimate       remaining evals + projected $ from the §8 measured table
    --repair-errors  ONE documented repair pass for a cell: re-attempt ONLY
                     records with error != null, replacing them in place.
                     The pre-repair file is moved to a .bak (nothing
                     deleted), every replacement is logged to
                     study/runs/logs/repairs.jsonl, and each replacement
                     record carries a "repaired" audit stamp. A cell that
                     already contains repaired records is REFUSED — the
                     repair pass is single-shot by design (triage protocol
                     in PREREG_ADDENDUM.md §12: transient errors only;
                     systematic errors stay as data, errors-as-failures
                     being the preregistered primary).

Env:  set -a; . ~/Documents/.env.local; set +a
Run:  .venv-study/bin/python study/run_study.py ...
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "study"))

import httpx  # noqa: E402

from sample_items import load_study_items  # noqa: E402

# ── Design constants (plan §6 + PREREG_ADDENDUM.md §7.1) ─────────────────────

TASKS = ("ragtruth-sum", "halueval-sum", "halueval-qa")
SPLITS = ("dev", "test", "ablation")
FRAMEWORKS = ("multivon-eval", "deepeval", "ragas", "trulens", "opik")
PRIMARY_JUDGES = ("gpt-4o-mini", "claude-haiku-4-5")
JUDGES = ("gpt-4o-mini", "claude-haiku-4-5", "gpt-5.5")
SEED = 42

# Strong-judge ablation (plan §5 judge 3, PREREG_ADDENDUM.md §10/§11):
# EXACTLY ONE cell shape is preregistered — ragtruth-sum x gpt-5.5 x run 0
# on the 150-item deterministic prefix subset. Any other --split ablation
# address, and any non-ablation use of the gpt-5.5 judge, is refused.
# The adapter receives the exact pinned snapshot id; the record's "judge"
# field and the output directory use the series tag "gpt-5.5".
ABLATION_JUDGE = "gpt-5.5"
ABLATION_SNAPSHOT = "gpt-5.5-2026-04-23"   # frozen Day 1, addendum §10
ABLATION_TASK = "ragtruth-sum"
N_ABLATION = 150


def runs_for(task: str, split: str, judge: str) -> int:
    """Preregistered run count for a (task, split, judge) — §7.1 table."""
    if split == "ablation":
        return 1                                   # strong judge: R=1
    if split == "dev":
        return 1                                   # dev: R=1, both judges
    if task == "ragtruth-sum":
        return 5                                   # test repeats, both judges
    return 3 if judge == "gpt-4o-mini" else 1      # halueval: R=3 gpt only


def items_count(task: str, split: str) -> int:
    if split == "ablation":
        return N_ABLATION
    if split == "dev":
        return 100
    return 500 if task == "ragtruth-sum" else 300  # §7.1 escalation


# Measured mean cost per eval (USD), framework x judge — hardcoded from the
# §8 smoke-measured multiplier table (PREREG_ADDENDUM.md §8; source of truth
# study/smoke/multipliers.json, "mean_cost_usd_per_eval"). Pinned prices:
# gpt-4o-mini $0.15/$0.60, claude-haiku-4-5 $1.00/$5.00 per Mtok.
MEASURED_COST_PER_EVAL = {
    "multivon-eval": {"gpt-4o-mini": 0.0007, "claude-haiku-4-5": 0.0055},
    "deepeval":      {"gpt-4o-mini": 0.0007, "claude-haiku-4-5": 0.0063},
    "ragas":         {"gpt-4o-mini": 0.0007, "claude-haiku-4-5": 0.0109},
    "trulens":       {"gpt-4o-mini": 0.0009, "claude-haiku-4-5": 0.0119},
    "opik":          {"gpt-4o-mini": 0.0002, "claude-haiku-4-5": 0.0026},
}

# Pinned prices, USD per 1M tokens (input, output) — same table as smoke.py.
PRICE_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-4-5": (1.00, 5.00),
}

RAW_ROOT = Path(__file__).resolve().parent / "runs" / "raw"
JUDGE_PATHS = ("/chat/completions", "/messages", "/responses", "/completions")


def _judge_tag(judge: str) -> str:
    return judge.replace("/", "_").replace(":", "_")


def cell_path(task: str, split: str, framework: str, judge: str, run: int) -> Path:
    return RAW_ROOT / _judge_tag(judge) / f"{task}_{split}" / f"{framework}_run{run}.jsonl"


def all_cells() -> list[tuple[str, str, str, str, int]]:
    """Every cell of the full preregistered design, in schedule order."""
    cells = []
    for split in ("dev", "test"):
        for task in TASKS:
            for judge in PRIMARY_JUDGES:
                for fw in FRAMEWORKS:
                    for r in range(runs_for(task, split, judge)):
                        cells.append((task, split, fw, judge, r))
    # Strong-judge ablation: the single preregistered cell shape x 5 fws.
    for fw in FRAMEWORKS:
        cells.append((ABLATION_TASK, "ablation", fw, ABLATION_JUDGE, 0))
    return cells


# ── httpx interception (thread-local attribution) ────────────────────────────

_TLS = threading.local()
_ORIG_SYNC_SEND = httpx.Client.send
_ORIG_ASYNC_SEND = httpx.AsyncClient.send


def _record_call(request: httpx.Request, response: httpx.Response) -> None:
    buf = getattr(_TLS, "calls", None)
    if buf is None:
        return
    url = str(request.url)
    if not any(p in url for p in JUDGE_PATHS):
        return
    entry: dict = {"status": response.status_code, "model": None,
                   "prompt_tokens": None, "completion_tokens": None}
    try:
        data = json.loads(response.content)
        entry["model"] = data.get("model")
        usage = data.get("usage") or {}
        entry["prompt_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens"))
        entry["completion_tokens"] = usage.get("completion_tokens", usage.get("output_tokens"))
    except Exception:
        pass  # streamed/opaque body — record the call anyway
    buf.append(entry)


def _sync_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
    response = _ORIG_SYNC_SEND(self, request, **kwargs)
    try:
        response.read()
    except Exception:
        pass
    _record_call(request, response)
    return response


async def _async_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
    response = await _ORIG_ASYNC_SEND(self, request, **kwargs)
    try:
        await response.aread()
    except Exception:
        pass
    _record_call(request, response)
    return response


def install_interceptor() -> None:
    httpx.Client.send = _sync_send            # type: ignore[method-assign]
    httpx.AsyncClient.send = _async_send      # type: ignore[method-assign]


def _cost_usd(judge: str, pt: int | None, ct: int | None) -> float | None:
    if pt is None or ct is None or judge not in PRICE_PER_MTOK:
        return None
    p_in, p_out = PRICE_PER_MTOK[judge]
    return round((pt * p_in + ct * p_out) / 1e6, 6)


# ── Runner construction ──────────────────────────────────────────────────────

def make_runner(framework: str, judge: str):
    """Same adapter classes as run.py's factories; additionally passes
    seed=42 to any adapter whose constructor accepts it (none of the five
    pinned adapters currently do — recorded per record as "seed")."""
    if framework == "multivon-eval":
        from frameworks.multivon import MultivonFaithfulness as Cls
    elif framework == "deepeval":
        from frameworks.deepeval import DeepEvalFaithfulness as Cls
    elif framework == "ragas":
        from frameworks.ragas import RagasFaithfulness as Cls
    elif framework == "trulens":
        from frameworks.trulens import TruLensGroundedness as Cls
    elif framework == "opik":
        from frameworks.opik import OpikHallucination as Cls
    else:
        raise ValueError(framework)
    # The ablation judge is addressed by its series tag ("gpt-5.5" — record
    # field + output directory) but the adapter must receive the exact
    # pinned snapshot id (addendum §10).
    adapter_model = ABLATION_SNAPSHOT if judge == ABLATION_JUDGE else judge
    kwargs: dict = {"judge_model": adapter_model}
    seed_passed = False
    if "seed" in inspect.signature(Cls.__init__).parameters:
        kwargs["seed"] = SEED
        seed_passed = True
    return Cls(**kwargs), seed_passed


# ── Cell state (resume + status) ─────────────────────────────────────────────

def read_cell_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"ABORT: {path}:{lineno} is not valid JSON ({exc}). The file "
                f"is corrupt (interrupted mid-write?). Repair or --fresh it "
                f"explicitly; refusing to guess.")
    return records


def validate_records(records: list[dict], *, path: Path, task: str, split: str,
                     framework: str, judge: str, run: int,
                     item_ids: set[str]) -> None:
    """Refuse to resume onto a file that doesn't match its cell address —
    the stale-file failure mode this runner exists to prevent."""
    for i, rec in enumerate(records, start=1):
        addr = (rec.get("task"), rec.get("split"), rec.get("framework"),
                rec.get("judge"), rec.get("run"))
        expect = (task, split, framework, judge, run)
        if addr != expect:
            raise SystemExit(
                f"ABORT: {path} record {i} addresses cell {addr}, expected "
                f"{expect}. This file is stale or misplaced — refusing to "
                f"resume onto it. Move it aside or rerun with --fresh.")
        if rec.get("item_id") not in item_ids:
            raise SystemExit(
                f"ABORT: {path} record {i} has item_id {rec.get('item_id')!r} "
                f"which is not in study/items/ for {task}/{split}. Stale "
                f"item set — refusing to resume. Move aside or --fresh.")


def cell_status(task: str, split: str, framework: str, judge: str, run: int) -> dict:
    path = cell_path(task, split, framework, judge, run)
    total = items_count(task, split)
    records = read_cell_records(path)
    done_ids = {r["item_id"] for r in records}
    limited = any(r.get("limit") is not None for r in records)
    errors = sum(1 for r in records if r.get("error"))
    state = ("done" if len(done_ids) >= total
             else "pending" if not done_ids else "partial")
    return {"path": path, "total": total, "done": len(done_ids),
            "errors": errors, "limited": limited, "state": state}


# ── Modes ────────────────────────────────────────────────────────────────────

def mode_cell_list() -> int:
    cells = all_cells()
    counts = {"pending": 0, "partial": 0, "done": 0}
    total_evals = done_evals = 0
    print(f"{'task':<14}{'split':<6}{'framework':<15}{'judge':<19}{'run':<5}"
          f"{'status':<9}{'done/total':<12}flags")
    for task, split, fw, judge, r in cells:
        st = cell_status(task, split, fw, judge, r)
        counts[st["state"]] += 1
        total_evals += st["total"]
        done_evals += min(st["done"], st["total"])
        flags = []
        if st["limited"]:
            flags.append("LIMIT-STAMPED")
        if st["errors"]:
            flags.append(f"{st['errors']} errors")
        print(f"{task:<14}{split:<6}{fw:<15}{judge:<19}{r:<5}"
              f"{st['state']:<9}{st['done']}/{st['total']:<10} {' '.join(flags)}")
    print(f"\n{len(cells)} cells: {counts['done']} done, {counts['partial']} "
          f"partial, {counts['pending']} pending | evals {done_evals}/{total_evals}")
    return 0


def mode_estimate() -> int:
    remaining_by = {}  # (fw, judge) -> remaining evals
    for task, split, fw, judge, r in all_cells():
        st = cell_status(task, split, fw, judge, r)
        rem = max(st["total"] - st["done"], 0)
        remaining_by[(fw, judge)] = remaining_by.get((fw, judge), 0) + rem
    total_rem = sum(remaining_by.values())
    total_usd = 0.0
    per_judge: dict[str, float] = {}
    print(f"{'framework':<15}{'judge':<19}{'remaining':<11}projected $")
    for (fw, judge), rem in sorted(remaining_by.items()):
        cost = MEASURED_COST_PER_EVAL[fw].get(judge)
        if cost is None:                 # gpt-5.5 ablation: no public price;
            print(f"{fw:<15}{judge:<19}{rem:<11}(unpriced — §8 scaled "
                  f"estimate covers the 750-eval ablation at ~$8.33 total)")
            continue
        usd = rem * cost
        total_usd += usd
        per_judge[judge] = per_judge.get(judge, 0.0) + usd
        print(f"{fw:<15}{judge:<19}{rem:<11}${usd:.2f}")
    print(f"\nremaining evals: {total_rem}")
    for judge, usd in sorted(per_judge.items()):
        print(f"  {judge}: ${usd:.2f}")
    print(f"projected remaining spend: ${total_usd:.2f}")
    print("(per-eval costs = §8 smoke-measured means, "
          "study/smoke/multipliers.json; the 750-eval strong-judge ablation "
          "has no public per-token price and is carried at the §8 scaled "
          "estimate of ~$8.33 total)")
    return 0


def make_eval_fn(runner, seed_passed: bool, *, task: str, split: str,
                 framework: str, judge: str, run: int, retries: int,
                 limit: int | None):
    """Per-item evaluation closure shared by run_cell and repair_cell."""

    def _eval_one(case) -> dict:
        attempts = 0
        while True:
            _TLS.calls = []
            t0 = time.perf_counter()
            try:
                result = runner.run(case)
            finally:
                calls, _TLS.calls = _TLS.calls, None
            latency_ms = (time.perf_counter() - t0) * 1000
            if not result.error or attempts >= retries:
                break
            attempts += 1
            time.sleep(min(2 ** attempts, 10))
        ok = [c for c in calls if 200 <= c["status"] < 300]
        models = sorted({c["model"] for c in ok if c["model"]})
        pt = (sum(c["prompt_tokens"] for c in ok)
              if ok and all(c["prompt_tokens"] is not None for c in ok) else None)
        ct = (sum(c["completion_tokens"] for c in ok)
              if ok and all(c["completion_tokens"] is not None for c in ok) else None)
        # Fall back to adapter-reported tokens when interception saw nothing.
        pt = pt if pt is not None else result.prompt_tokens
        ct = ct if ct is not None else result.completion_tokens
        rec = {
            "item_id": case.id,
            "task": task, "split": split,
            "framework": framework,
            "judge": judge,
            "judge_snapshot": (models[0] if len(models) == 1
                               else (models or None)),
            "run": run,
            "score": result.score,
            "threshold": result.threshold,
            "verdict": "hallucinated" if result.flagged_hallucinated else "faithful",
            "flagged_hallucinated": result.flagged_hallucinated,
            "error": result.error,
            "retries": attempts,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "judge_calls": len(ok) or None,
            "cost_usd": _cost_usd(judge, pt, ct),
            "seed": SEED if seed_passed else None,
            "raw": result.to_dict().get("raw"),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if limit is not None:
            rec["limit"] = limit
        return rec

    return _eval_one


def run_cell(args: argparse.Namespace) -> int:
    task, split, framework, judge, run = (
        args.task, args.split, args.framework, args.judge, args.run)

    # Strong-judge ablation guard (addendum §10/§11): exactly ONE cell shape
    # is preregistered. Refuse everything else, in both directions.
    if split == "ablation" and (task != ABLATION_TASK or judge != ABLATION_JUDGE):
        raise SystemExit(
            f"ABORT: the strong-judge ablation is preregistered ONLY as "
            f"{ABLATION_TASK} x {ABLATION_JUDGE} (snapshot "
            f"{ABLATION_SNAPSHOT}) x run 0 on the 150-item prefix subset. "
            f"Refusing {task} x {judge}.")
    if split != "ablation" and judge == ABLATION_JUDGE:
        raise SystemExit(
            f"ABORT: judge {ABLATION_JUDGE} ({ABLATION_SNAPSHOT}) is "
            f"preregistered ONLY for --split ablation (plan §5 judge 3). "
            f"Refusing {task}/{split}.")

    n_runs = runs_for(task, split, judge)
    if not (0 <= run < n_runs):
        raise SystemExit(
            f"ABORT: run {run} is outside the preregistered design for "
            f"{task}/{split} x {judge} (R={n_runs}, 0-indexed). Off-design "
            f"cells are refused to protect the spend budget (§7.1 table).")

    cases = load_study_items(task, split)     # blinded loader ONLY
    expected_n = items_count(task, split)
    assert len(cases) == expected_n, (task, split, len(cases))
    if args.limit is not None:
        cases = cases[:args.limit]
        print(f"NOTE: --limit {args.limit} — VALIDATION MODE; every record "
              f"written will carry \"limit\": {args.limit}", file=sys.stderr)

    path = cell_path(task, split, framework, judge, run)
    cell_label = f"{task}/{split} {framework} x {judge} run{run}"

    if args.fresh and path.exists():
        bak = path.with_suffix(f".jsonl.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        path.rename(bak)
        print(f"--fresh: moved existing {path.name} -> {bak.name} "
              f"(nothing deleted)", file=sys.stderr)

    all_ids = {c.id for c in load_study_items(task, split)}
    existing = read_cell_records(path)
    validate_records(existing, path=path, task=task, split=split,
                     framework=framework, judge=judge, run=run,
                     item_ids=all_ids)
    done_ids = {r["item_id"] for r in existing}
    if existing:
        lim = sorted({r["limit"] for r in existing if r.get("limit") is not None})
        lim_note = f" (contains --limit-stamped records: {lim})" if lim else ""
        print(f"### RESUMING {cell_label}: {len(done_ids)}/{expected_n} done "
              f"in {path}{lim_note}", file=sys.stderr)

    todo = [c for c in cases if c.id not in done_ids]
    if not todo:
        scope = f"first {len(cases)} (--limit)" if args.limit else f"all {expected_n}"
        print(f"### cell already complete: {cell_label} — {scope} items "
              f"present in {path}. Use --fresh to redo.", file=sys.stderr)
        return 0
    if args.stop_after is not None and len(todo) > args.stop_after:
        # Invocation batching ONLY: submit the first N pending items and
        # exit cleanly; the cell resumes on the next invocation. Unlike
        # --limit this does NOT change the item set and stamps nothing —
        # the finished cell is indistinguishable from a single-invocation
        # run (append-resume is already the design).
        print(f"NOTE: --stop-after {args.stop_after} — running "
              f"{args.stop_after}/{len(todo)} pending items this invocation",
              file=sys.stderr)
        todo = todo[:args.stop_after]

    runner, seed_passed = make_runner(framework, judge)
    install_interceptor()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    t_start = time.time()
    _eval_one = make_eval_fn(runner, seed_passed, task=task, split=split,
                             framework=framework, judge=judge, run=run,
                             retries=args.retries, limit=args.limit)

    n_err = 0
    with open(path, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_eval_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            with write_lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if rec["error"]:
                n_err += 1
            if i % 10 == 0 or i == len(todo):
                print(f"  {cell_label}: {i}/{len(todo)} new "
                      f"({len(done_ids) + i}/{expected_n} total, {n_err} errors)",
                      file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"[done] {cell_label} -> {path} (+{len(todo)} records, {n_err} "
          f"errors, {elapsed:.0f}s)", file=sys.stderr)
    if n_err:
        print(f"WARNING: {n_err}/{len(todo)} evals recorded errors "
              f"(errors-as-failures per plan §7 P4; inspect before repeats).",
              file=sys.stderr)
    return 0


def repair_cell(args: argparse.Namespace) -> int:
    """ONE documented repair pass for a cell (triage protocol,
    PREREG_ADDENDUM.md §12): re-attempt ONLY records with error != null and
    replace them in place.

    Guarantees:
      * single-shot — refuses to run if ANY record in the cell already
        carries a "repaired" stamp (a second pass would be undocumented
        selection pressure on errors);
      * nothing deleted — the pre-repair file is moved to
        <file>.jsonl.bak-repair-<ts> before the rewrite;
      * fully audited — every replacement (including still-errored
        re-attempts, which are kept as data) appends a line to
        study/runs/logs/repairs.jsonl and stamps the replacement record
        with {"repaired": {"pass": 1, "prev_error": ..., "prev_ts": ...}};
      * order-preserving — the rewritten file keeps the original record
        order, with non-errored records byte-identical.
    """
    task, split, framework, judge, run = (
        args.task, args.split, args.framework, args.judge, args.run)
    if args.limit is not None or args.fresh or args.stop_after is not None:
        raise SystemExit("ABORT: --repair-errors is incompatible with "
                         "--limit/--fresh/--stop-after.")
    path = cell_path(task, split, framework, judge, run)
    cell_label = f"{task}/{split} {framework} x {judge} run{run}"
    records = read_cell_records(path)
    if not records:
        raise SystemExit(f"ABORT: no records in {path} — nothing to repair.")
    all_ids = {c.id for c in load_study_items(task, split)}
    validate_records(records, path=path, task=task, split=split,
                     framework=framework, judge=judge, run=run,
                     item_ids=all_ids)
    stamped = [r["item_id"] for r in records if r.get("repaired")]
    if stamped:
        raise SystemExit(
            f"ABORT: {cell_label} already contains {len(stamped)} repaired "
            f"record(s) (e.g. {stamped[0]}). The repair pass is single-shot "
            f"by design; a second pass is refused.")
    errored = {r["item_id"]: r for r in records if r.get("error")}
    if not errored:
        print(f"{cell_label}: no errored records — nothing to repair.",
              file=sys.stderr)
        return 0

    by_id = {c.id: c for c in load_study_items(task, split)}
    todo = [by_id[i] for i in errored]
    print(f"### REPAIR PASS {cell_label}: re-attempting {len(todo)} errored "
          f"record(s)", file=sys.stderr)

    runner, seed_passed = make_runner(framework, judge)
    install_interceptor()
    _eval_one = make_eval_fn(runner, seed_passed, task=task, split=split,
                             framework=framework, judge=judge, run=run,
                             retries=args.retries, limit=None)

    repair_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_recs: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_eval_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            old = errored[rec["item_id"]]
            rec["repaired"] = {"pass": 1, "repair_ts": repair_ts,
                               "prev_error": (old.get("error") or "")[:300],
                               "prev_ts": old.get("ts")}
            new_recs[rec["item_id"]] = rec
            print(f"  repair {cell_label}: {i}/{len(todo)} "
                  f"({rec['item_id']}: "
                  f"{'STILL ERRORED' if rec['error'] else 'ok'})",
                  file=sys.stderr)

    # Audit log — one line per replacement, before touching the cell file.
    log_path = RAW_ROOT.parent / "logs" / "repairs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        for iid in sorted(new_recs):
            old, new = errored[iid], new_recs[iid]
            lf.write(json.dumps({
                "cell": {"task": task, "split": split, "framework": framework,
                         "judge": judge, "run": run},
                "item_id": iid, "repair_ts": repair_ts,
                "prev_error": (old.get("error") or "")[:300],
                "prev_ts": old.get("ts"),
                "new_error": (new.get("error") or None)
                             and new["error"][:300],
                "new_verdict": new["verdict"], "new_score": new["score"],
            }) + "\n")

    bak = path.with_suffix(
        f".jsonl.bak-repair-{time.strftime('%Y%m%d-%H%M%S')}")
    path.rename(bak)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(new_recs.get(r["item_id"], r)) + "\n")

    still = sum(1 for r in new_recs.values() if r["error"])
    print(f"[repair done] {cell_label}: {len(new_recs)} re-attempted, "
          f"{len(new_recs) - still} now ok, {still} still errored (kept as "
          f"data). Pre-repair file: {bak.name}; audit: {log_path}",
          file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--split", choices=SPLITS)
    p.add_argument("--framework", choices=FRAMEWORKS)
    p.add_argument("--judge", choices=JUDGES)
    p.add_argument("--run", type=int, help="0-indexed run number")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=2,
                   help="extra attempts per item when the adapter returns error")
    p.add_argument("--limit", type=int, default=None,
                   help="VALIDATION ONLY: run first N items; stamps 'limit' "
                        "into every record so partial cells are detectable")
    p.add_argument("--stop-after", type=int, default=None,
                   help="invocation batching: submit at most N pending items "
                        "then exit cleanly (cell resumes next invocation; "
                        "no stamping — item set is unchanged)")
    p.add_argument("--fresh", action="store_true",
                   help="explicitly redo the cell: existing file is moved to "
                        ".bak (default behaviour is append-mode resume)")
    p.add_argument("--repair-errors", action="store_true",
                   help="ONE documented repair pass: re-attempt ONLY records "
                        "with error != null in the addressed cell, replacing "
                        "them in place (audited in study/runs/logs/"
                        "repairs.jsonl; pre-repair file kept as .bak; a "
                        "second pass on the same cell is refused)")
    p.add_argument("--cell-list", action="store_true",
                   help="print every cell of the full design with status")
    p.add_argument("--estimate", action="store_true",
                   help="remaining evals + projected $ (§8 measured table)")
    args = p.parse_args()

    if args.cell_list:
        return mode_cell_list()
    if args.estimate:
        return mode_estimate()

    missing = [n for n in ("task", "split", "framework", "judge", "run")
               if getattr(args, n) is None]
    if missing:
        p.error(f"cell address incomplete: missing --{', --'.join(missing)} "
                f"(or use --cell-list / --estimate)")
    if args.repair_errors:
        return repair_cell(args)
    return run_cell(args)


if __name__ == "__main__":
    raise SystemExit(main())

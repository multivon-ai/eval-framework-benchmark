"""Day-1 smoke run + measured judge-call multipliers (plan §11, addendum §8).

Runs the committed BLINDED dev items — the first 2 items (id order) of
``study/items/ragtruth-sum_dev_100.json`` and
``study/items/halueval-sum_dev_100.json`` — through all five framework
adapters (the exact ``run.py`` factories, so semantics are identical to the
orchestrator) under both primary judges. run.py itself is not used because
it consumes the raw HaluEval/RAGTruth loaders, not the committed
``study/items`` files.

Measurement: every judge HTTP call is intercepted at the shared transport
choke point (``httpx.Client.send`` / ``httpx.AsyncClient.send`` — the
OpenAI SDK, Anthropic SDK, LangChain-OpenAI and LiteLLM all sit on httpx in
the study lockfile). Per eval we record: number of judge calls, prompt/
completion tokens (provider-reported ``usage``), wall latency, and cost at
the pinned prices below. Reading the response body inside the interceptor
is safe: httpx serves ``iter_bytes``/``content`` from the buffered
``_content`` after ``read()``.

Outputs (committed):
    study/smoke/raw/{framework}_{judge}.jsonl   one line per eval:
        FrameworkResult dict + "measured" block (calls, tokens, cost) +
        the per-call trace.
    study/smoke/multipliers.json                aggregated framework x judge
        multiplier table + re-issued cost projection for the full
        ESCALATED 40,750-eval design (PREREG_ADDENDUM.md §7.1: RAGTruth-Sum
        test n=500) vs the ~$780 cap (plan §11), plus per-cell wall-clock
        projection at --workers 8 with a 12-hour flag.

Resumable: a completed {framework}_{judge}.jsonl (4 lines) is skipped.
Sequential on purpose — call attribution needs one eval in flight at a time.

Run:  set -a; . ~/Documents/.env.local; set +a
      .venv-study/bin/python study/smoke.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "study"))

import httpx  # noqa: E402

from run import _judge_tag, _safe_runner_factories  # noqa: E402
from sample_items import load_study_items  # noqa: E402

JUDGES = ["gpt-4o-mini", "claude-haiku-4-5"]
TASKS = ["ragtruth-sum", "halueval-sum"]
N_PER_TASK = 2

OUT_DIR = Path(__file__).resolve().parent / "smoke"
RAW_DIR = OUT_DIR / "raw"
MULT_PATH = OUT_DIR / "multipliers.json"

# Pinned prices, USD per 1M tokens (input, output), checked 2026-07-13.
PRICE_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Full ESCALATED design (plan §6 recomputed in PREREG_ADDENDUM.md §7.1 with
# RAGTruth-Sum test n=500): 40,750 evals total.
#   gpt-4o-mini cells (dev + run1 + all repeats): 23,000 evals = 4,600/fw
#   claude-haiku-4-5 cells (dev + run1 + RAGTruth repeats): 17,000 = 3,400/fw
#   strong-judge ablation: 750 evals (gpt-5.5; price not public — projected
#   by scaling the plan's $35 line by the measured-token factor, see below)
# Cell structure per framework (addendum §7.1 table rows), used for both the
# cost projection and the per-cell wall-clock projection at --workers 8.
CELLS_PER_FW = [
    # (cell name, {judge: evals per framework})
    ("dev_100x3",              {"gpt-4o-mini": 300,  "claude-haiku-4-5": 300}),
    ("test_run1_500+300x2",    {"gpt-4o-mini": 1100, "claude-haiku-4-5": 1100}),
    ("ragtruth_repeats_500x4", {"gpt-4o-mini": 2000, "claude-haiku-4-5": 2000}),
    ("halueval_repeats_300x2x2", {"gpt-4o-mini": 1200}),
]
EVALS_PER_FW = {
    j: sum(c[1].get(j, 0) for c in CELLS_PER_FW)
    for j in ("gpt-4o-mini", "claude-haiku-4-5")
}  # -> {"gpt-4o-mini": 4600, "claude-haiku-4-5": 3400}
WORKERS = 8
WALL_CLOCK_FLAG_HOURS = 12.0
STRONG_JUDGE_PLAN_USD = 35.0
STRONG_JUDGE_ASSUMED_TOKENS = 4 * 3400.0     # plan: 4 calls x (3k in + 400 out)
A2_SMOKE_RERUNS_USD = 40.0
CONTINGENCY_USD = 400.0
BUDGET_CAP_USD = 780.0

JUDGE_PATHS = ("/chat/completions", "/messages", "/responses", "/completions")


# ── httpx interception ───────────────────────────────────────────────────────

CALLS: list[dict] = []
_ORIG_SYNC_SEND = httpx.Client.send
_ORIG_ASYNC_SEND = httpx.AsyncClient.send


def _record(request: httpx.Request, response: httpx.Response, t0: float) -> None:
    url = str(request.url)
    if not any(p in url for p in JUDGE_PATHS):
        return
    entry: dict = {
        "url": url.split("?")[0],
        "status": response.status_code,
        "call_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "model": None, "prompt_tokens": None, "completion_tokens": None,
    }
    try:
        data = json.loads(response.content)
        entry["model"] = data.get("model")
        usage = data.get("usage") or {}
        entry["prompt_tokens"] = usage.get("prompt_tokens",
                                           usage.get("input_tokens"))
        entry["completion_tokens"] = usage.get("completion_tokens",
                                               usage.get("output_tokens"))
    except Exception as exc:  # streamed/opaque body — record the call anyway
        entry["parse_error"] = f"{type(exc).__name__}: {exc}"
    CALLS.append(entry)


def _sync_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    response = _ORIG_SYNC_SEND(self, request, **kwargs)
    try:
        response.read()
    except Exception:
        pass
    _record(request, response, t0)
    return response


async def _async_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    response = await _ORIG_ASYNC_SEND(self, request, **kwargs)
    try:
        await response.aread()
    except Exception:
        pass
    _record(request, response, t0)
    return response


def install_interceptor() -> None:
    httpx.Client.send = _sync_send            # type: ignore[method-assign]
    httpx.AsyncClient.send = _async_send      # type: ignore[method-assign]


def _price_for(model: str | None, judge: str) -> tuple[float, float]:
    """Prefer the response-reported model id; fall back to the judge id."""
    for key, price in PRICE_PER_MTOK.items():
        if model and model.startswith(key):
            return price
    return PRICE_PER_MTOK[judge]


def _call_cost(entry: dict, judge: str) -> float | None:
    pt, ct = entry.get("prompt_tokens"), entry.get("completion_tokens")
    if pt is None or ct is None:
        return None
    p_in, p_out = _price_for(entry.get("model"), judge)
    return (pt * p_in + ct * p_out) / 1e6


# ── Smoke run ────────────────────────────────────────────────────────────────

def smoke_items() -> list:
    items = []
    for task in TASKS:
        items.extend(load_study_items(task, "dev")[:N_PER_TASK])
    return items


def run_smoke() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    items = smoke_items()
    assert len(items) == N_PER_TASK * len(TASKS)

    for judge in JUDGES:
        for name, factory in _safe_runner_factories(judge):
            out_path = RAW_DIR / f"{name}_{_judge_tag(judge)}.jsonl"
            if out_path.exists() and len(out_path.read_text().splitlines()) == len(items):
                print(f"[skip] {out_path.name} complete", file=sys.stderr)
                continue
            runner = factory()
            rows = []
            for case in items:
                lo = len(CALLS)
                t0 = time.perf_counter()
                result = runner.run(case)
                eval_ms = (time.perf_counter() - t0) * 1000
                trace = CALLS[lo:]
                ok = [c for c in trace if 200 <= c["status"] < 300]
                costs = [_call_cost(c, judge) for c in ok]
                measured = {
                    "judge_calls": len(ok),
                    "judge_calls_incl_errors": len(trace),
                    "prompt_tokens": (sum(c["prompt_tokens"] for c in ok)
                                      if ok and all(c["prompt_tokens"] is not None for c in ok) else None),
                    "completion_tokens": (sum(c["completion_tokens"] for c in ok)
                                          if ok and all(c["completion_tokens"] is not None for c in ok) else None),
                    "eval_latency_ms": round(eval_ms, 1),
                    "cost_usd": (round(sum(costs), 6)
                                 if costs and all(x is not None for x in costs) else None),
                }
                rows.append({**result.to_dict(), "judge_model": judge,
                             "measured": measured, "calls": trace})
                flag = "ERR" if result.error else ("FLAG" if result.flagged_hallucinated else "pass")
                print(f"  [{judge}] {name} {case.id}: score={result.score:.3f} "
                      f"{flag} calls={measured['judge_calls']} "
                      f"tok={measured['prompt_tokens']}/{measured['completion_tokens']} "
                      f"{eval_ms/1000:.1f}s", file=sys.stderr)
            out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"[done] {out_path}", file=sys.stderr)


# ── Aggregation: multiplier table + re-issued cost projection ────────────────

def aggregate() -> dict:
    frameworks = [n for n, _ in _safe_runner_factories("gpt-4o-mini")]
    table: dict[str, dict] = {}
    incomplete = []
    for judge in JUDGES:
        for name in frameworks:
            path = RAW_DIR / f"{name}_{_judge_tag(judge)}.jsonl"
            if not path.exists():
                incomplete.append(path.name)
                continue
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            n = len(rows)
            errs = [r["case_id"] for r in rows if r.get("error")]
            m = [r["measured"] for r in rows]
            def _mean(key: str) -> float | None:
                vals = [x[key] for x in m if x.get(key) is not None]
                return round(sum(vals) / len(vals), 4) if vals else None
            cell = {
                "n_evals": n,
                "n_errors": len(errs),
                "error_case_ids": errs,
                "all_verdicts_returned": len(errs) == 0 and n == N_PER_TASK * len(TASKS),
                "mean_judge_calls_per_eval": _mean("judge_calls"),
                "mean_prompt_tokens_per_eval": _mean("prompt_tokens"),
                "mean_completion_tokens_per_eval": _mean("completion_tokens"),
                "mean_latency_ms_per_eval": _mean("eval_latency_ms"),
                "mean_cost_usd_per_eval": _mean("cost_usd"),
            }
            table.setdefault(name, {})[judge] = cell
    if incomplete:
        raise SystemExit(f"smoke incomplete, missing: {incomplete}")

    # Re-issued projection (plan §11 line items, measured multipliers).
    proj: dict = {"per_judge_cells_usd": {}, "line_items_usd": {}}
    total = 0.0
    for judge, per_fw in EVALS_PER_FW.items():
        cell_cost = 0.0
        for name in frameworks:
            c = table[name][judge]["mean_cost_usd_per_eval"]
            if c is None:
                raise SystemExit(f"no measured cost for {name} x {judge}")
            cell_cost += per_fw * c
        proj["per_judge_cells_usd"][judge] = round(cell_cost, 2)
        total += cell_cost
    # Strong-judge line: gpt-5.5 pricing is not assumed here; the plan's $35
    # estimate is scaled by the measured-vs-assumed token volume factor of
    # the gpt-judge cells (calls x tokens), keeping the plan's price basis.
    gpt_tok = [table[n]["gpt-4o-mini"]["mean_prompt_tokens_per_eval"]
               + table[n]["gpt-4o-mini"]["mean_completion_tokens_per_eval"]
               for n in frameworks]
    token_factor = (sum(gpt_tok) / len(gpt_tok)) / STRONG_JUDGE_ASSUMED_TOKENS
    strong = round(STRONG_JUDGE_PLAN_USD * token_factor, 2)
    proj["line_items_usd"] = {
        "gpt-4o-mini_cells_23000_evals": proj["per_judge_cells_usd"]["gpt-4o-mini"],
        "claude-haiku-4-5_cells_17000_evals": proj["per_judge_cells_usd"]["claude-haiku-4-5"],
        "strong_judge_750_evals_scaled": strong,
        "a2_smoke_reruns_plan_line": A2_SMOKE_RERUNS_USD,
        "contingency_plan_line": CONTINGENCY_USD,
    }
    proj["strong_judge_token_factor_vs_plan_assumption"] = round(token_factor, 3)
    proj["contingency_note"] = (
        "The n->500 escalation the $400 contingency line was written for is "
        "now priced directly into the measured cells above; the contingency "
        "is retained as pure headroom (multiplier drift, retries, A2)."
    )
    total += strong + A2_SMOKE_RERUNS_USD
    proj["projected_total_excl_contingency_usd"] = round(total, 2)
    total += CONTINGENCY_USD
    proj["projected_total_usd"] = round(total, 2)
    proj["budget_cap_usd"] = BUDGET_CAP_USD
    proj["within_cap"] = total <= BUDGET_CAP_USD

    # Wall-clock per cell at --workers 8 (plan D5-7 feasibility; RAGAS was
    # ~130 s/eval in the pilot). hours = evals x mean_latency / workers.
    wall: dict[str, dict] = {}
    over: list[str] = []
    for name in frameworks:
        wall[name] = {}
        for cell_name, per_judge in CELLS_PER_FW:
            for judge, n_evals in per_judge.items():
                lat_ms = table[name][judge]["mean_latency_ms_per_eval"]
                if lat_ms is None:
                    continue
                hours = round(n_evals * (lat_ms / 1000.0) / WORKERS / 3600.0, 2)
                key = f"{cell_name} x {judge}"
                wall[name][key] = hours
                if hours > WALL_CLOCK_FLAG_HOURS:
                    over.append(f"{name} {key}: {hours}h")
    proj["wall_clock_hours_per_cell_at_workers_8"] = wall
    proj["cells_over_12h"] = over

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "design": "4 dev items (2 ragtruth-sum + 2 halueval-sum, first-by-id, "
                  "blinded) x 5 frameworks x 2 judges, sequential",
        "prices_per_mtok": PRICE_PER_MTOK,
        "multipliers": table,
        "projection": proj,
    }
    MULT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {MULT_PATH}")
    print(f"projected total ${proj['projected_total_usd']:.2f} vs cap "
          f"${BUDGET_CAP_USD:.0f} — {'WITHIN CAP' if proj['within_cap'] else 'OVER CAP - STOP'}")
    if over:
        print(f"WALL-CLOCK FLAG (> {WALL_CLOCK_FLAG_HOURS:.0f}h at workers={WORKERS}):")
        for line in over:
            print(f"  {line}")
    else:
        print(f"wall-clock: no cell exceeds {WALL_CLOCK_FLAG_HOURS:.0f}h at workers={WORKERS}")
    return out


def main() -> int:
    install_interceptor()
    run_smoke()
    out = aggregate()
    return 0 if out["projection"]["within_cap"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

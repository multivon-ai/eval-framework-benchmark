"""A2 prompt/parser attribution — the PREREGISTERED FALLBACK path (plan §8 A2).

The plan's primary A2 instrument was an OpenAI-compatible logging proxy
(base_url override) with full cross-framework replay. Its preregistered
1-day time-box is spent; per the plan's explicit fallback clause ("if the
proxy is not working within 1 day, use standalone prompt extraction + a
shared minimal parser, and report the parsing confound as a limitation")
this module implements the fallback. The decision is recorded in
PREREG_ADDENDUM.md §11. No proxy is attempted here.

Part A — exact judge payload extraction.
    For ONE fixed dev item (the first ragtruth-sum dev item by id,
    ``ragtruth_sum_0`` — the same item that leads the Day-1 smoke) each
    framework adapter is run once per primary judge with every judge HTTP
    request intercepted at the shared httpx transport choke point (the
    same machinery as study/smoke.py, extended to capture full REQUEST
    BODIES, not just usage). Output:
        study/a2/prompts/{framework}_{judge}.json
    (one file per framework x judge: the ordered list of full request
    bodies + response summaries for that single eval).

Part B — shared minimal parser comparison.
    RAGAS's canonical judge prompt (the NLI verdict call — the LAST judge
    request of its faithfulness eval) is sent ONCE per primary judge (2
    API calls total, temperature 0) and the SAME raw completion is fed to
    each framework's own parse path where that path is reachable
    standalone (an importable function taking raw completion text, no
    live LLM required). Where it is not reachable, NOT-REACHABLE is
    recorded — no reimplementation, no hacks. Output:
        study/a2/parser_comparison.json
        study/a2/A2_REPORT.md   (fingerprint table + reachability +
                                 limitation paragraph)

Spend: 10 evals (5 frameworks x 2 judges, one item) + 2 raw completions;
well under the $1 JOB-2 budget (smoke-measured per-eval costs, addendum §8).

Run:  set -a; . ~/Documents/.env.local; set +a
      .venv-study/bin/python study/a2_prompts.py [--extract] [--compare]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "study"))

import httpx  # noqa: E402

from sample_items import load_study_items  # noqa: E402

JUDGES = ("gpt-4o-mini", "claude-haiku-4-5")
FRAMEWORKS = ("multivon-eval", "deepeval", "ragas", "trulens", "opik")
FIXED_ITEM_TASK, FIXED_ITEM_SPLIT = "ragtruth-sum", "dev"   # first item by id

A2_DIR = Path(__file__).resolve().parent / "a2"
PROMPTS_DIR = A2_DIR / "prompts"
COMPARISON_PATH = A2_DIR / "parser_comparison.json"
REPORT_PATH = A2_DIR / "A2_REPORT.md"

JUDGE_PATHS = ("/chat/completions", "/messages", "/responses", "/completions")

# ── httpx interception (global buffer; sequential execution — one eval in
#    flight at a time, so a global buffer also catches frameworks that issue
#    calls from worker threads, e.g. DeepEval) ────────────────────────────────

CALLS: list[dict] = []
_ORIG_SYNC_SEND = httpx.Client.send
_ORIG_ASYNC_SEND = httpx.AsyncClient.send


def _record(request: httpx.Request, response: httpx.Response) -> None:
    url = str(request.url)
    if not any(p in url for p in JUDGE_PATHS):
        return
    entry: dict = {"url": url.split("?")[0], "status": response.status_code,
                   "request_body": None, "response_model": None,
                   "response_text": None, "usage": None}
    try:
        entry["request_body"] = json.loads(request.content)
    except Exception:
        entry["request_body"] = {"_undecodable": repr(request.content[:2000])}
    try:
        data = json.loads(response.content)
        entry["response_model"] = data.get("model")
        entry["usage"] = data.get("usage")
        if "choices" in data:            # OpenAI chat completions
            entry["response_text"] = data["choices"][0]["message"].get("content")
        elif "content" in data:          # Anthropic messages
            blocks = data["content"]
            entry["response_text"] = "".join(
                b.get("text", "") for b in blocks if isinstance(b, dict))
    except Exception:
        pass
    CALLS.append(entry)


def _sync_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
    response = _ORIG_SYNC_SEND(self, request, **kwargs)
    try:
        response.read()
    except Exception:
        pass
    _record(request, response)
    return response


async def _async_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
    response = await _ORIG_ASYNC_SEND(self, request, **kwargs)
    try:
        await response.aread()
    except Exception:
        pass
    _record(request, response)
    return response


def install_interceptor() -> None:
    httpx.Client.send = _sync_send            # type: ignore[method-assign]
    httpx.AsyncClient.send = _async_send      # type: ignore[method-assign]


# ── Part A: payload extraction ───────────────────────────────────────────────

def fixed_item():
    case = load_study_items(FIXED_ITEM_TASK, FIXED_ITEM_SPLIT)[0]
    return case


def extract_payloads() -> None:
    from run_study import make_runner  # same adapter factories as the study
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    case = fixed_item()
    print(f"fixed dev item: {case.id} ({FIXED_ITEM_TASK}/{FIXED_ITEM_SPLIT}, "
          f"first by id)", file=sys.stderr)
    for judge in JUDGES:
        for fw in FRAMEWORKS:
            out = PROMPTS_DIR / f"{fw}_{judge}.json"
            if out.exists():
                print(f"[skip] {out.name} exists", file=sys.stderr)
                continue
            runner, _ = make_runner(fw, judge)
            lo = len(CALLS)
            t0 = time.perf_counter()
            result = runner.run(case)
            secs = time.perf_counter() - t0
            trace = CALLS[lo:]
            doc = {
                "framework": fw, "judge": judge, "item_id": case.id,
                "captured": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "n_judge_calls": len(trace),
                "eval_error": result.error,
                "eval_score": result.score,
                "calls": trace,
            }
            out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            print(f"[done] {out.name}: {len(trace)} judge calls, "
                  f"score={result.score:.3f}, {secs:.1f}s"
                  + (f" ERROR: {result.error[:80]}" if result.error else ""),
                  file=sys.stderr)


# ── Part B: shared minimal parser comparison ─────────────────────────────────

def _messages_text(body: dict) -> str:
    """Flatten an OpenAI chat/Responses-API or Anthropic request body's
    prompt content to one string (chat: ``messages``; Responses API:
    ``input``; Anthropic: ``system`` + ``messages``)."""
    parts = []
    if body.get("system"):
        s = body["system"]
        parts.append(s if isinstance(s, str) else json.dumps(s))
    msgs = body.get("messages", [])
    inp = body.get("input")
    if isinstance(inp, str):
        parts.append(inp)
    elif isinstance(inp, list):
        msgs = list(msgs) + inp
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(b.get("text", "") for b in c if isinstance(b, dict))
    return "\n".join(parts)


def ragas_canonical_prompt() -> dict:
    """RAGAS's canonical judge prompt = the NLI verdict call (the LAST judge
    request of its faithfulness eval) captured for the fixed item under
    gpt-4o-mini in Part A."""
    src = PROMPTS_DIR / "ragas_gpt-4o-mini.json"
    doc = json.loads(src.read_text())
    calls = [c for c in doc["calls"] if 200 <= c["status"] < 300]
    body = calls[-1]["request_body"]
    return {"source_file": src.name, "call_index": len(doc["calls"]) - 1,
            "messages": body.get("messages"),
            "response_format": body.get("response_format"),
            "tools": body.get("tools"),
            "text": _messages_text(body)}


def _send_prompt_once(judge: str, messages: list[dict]) -> dict:
    """One plain temperature-0 call per judge with RAGAS's canonical prompt.

    The prompt text is sent verbatim; only the transport differs (OpenAI
    SDK for gpt-4o-mini, Anthropic SDK for claude-haiku-4-5 — the same two
    transports the study's primary lanes use). Structured-output /
    tool-call scaffolding is deliberately NOT replicated: the point of the
    fallback is a plain shared completion that each parser then faces.
    """
    plain = [{"role": m["role"], "content": m["content"]}
             for m in messages if m.get("role") != "system"]
    system = "\n".join(m["content"] for m in messages
                       if m.get("role") == "system" and isinstance(m.get("content"), str))
    if judge.startswith("claude"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=judge, max_tokens=2048, temperature=0.0,
            system=system or anthropic.NOT_GIVEN, messages=plain)
        return {"judge": judge, "model": resp.model,
                "text": "".join(b.text for b in resp.content
                                if getattr(b, "type", "") == "text"),
                "usage": {"input_tokens": resp.usage.input_tokens,
                          "output_tokens": resp.usage.output_tokens}}
    from openai import OpenAI
    client = OpenAI()
    msgs = ([{"role": "system", "content": system}] if system else []) + plain
    resp = client.chat.completions.create(
        model=judge, temperature=0.0, messages=msgs)
    return {"judge": judge, "model": resp.model,
            "text": resp.choices[0].message.content or "",
            "usage": {"prompt_tokens": resp.usage.prompt_tokens,
                      "completion_tokens": resp.usage.completion_tokens}}


def _try(fn, raw: str) -> dict:
    try:
        out = fn(raw)
        return {"ok": True, "parsed": repr(out)[:500]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}


def parse_paths() -> dict[str, dict]:
    """Each framework's completion->structure parse path, where reachable
    standalone (importable, no live LLM). NOT-REACHABLE is recorded where
    the parse logic is embedded in a method that requires a constructed
    provider/endpoint or an LLM-backed retry loop — we do not reimplement.
    """
    paths: dict[str, dict] = {}

    # ragas — RagasOutputParser (a langchain PydanticOutputParser subclass)
    # over the faithfulness NLI output model. Its FIRST parse attempt
    # (.parse, sync, no LLM) is standalone; the retry path
    # (.parse_output_string) needs a live LLM and is out of scope.
    try:
        from ragas.metrics._faithfulness import NLIStatementOutput
        from ragas.prompt.pydantic_prompt import RagasOutputParser
        parser = RagasOutputParser(pydantic_object=NLIStatementOutput)
        paths["ragas"] = {
            "reachable": True,
            "entry": "ragas.prompt.pydantic_prompt.RagasOutputParser"
                     "(pydantic_object=NLIStatementOutput).parse(raw) "
                     "[first attempt; LLM-backed fix-retry not exercised]",
            "fn": parser.parse,
        }
    except Exception as exc:
        paths["ragas"] = {"reachable": False,
                          "entry": f"import failed: {exc}"}

    # deepeval — trimAndLoadJson is the single JSON-recovery chokepoint all
    # DeepEval metric parse steps go through.
    try:
        from deepeval.metrics.utils import trimAndLoadJson
        paths["deepeval"] = {
            "reachable": True,
            "entry": "deepeval.metrics.utils.trimAndLoadJson(raw)",
            "fn": lambda raw: trimAndLoadJson(raw),
        }
    except Exception as exc:
        paths["deepeval"] = {"reachable": False,
                             "entry": f"import failed: {exc}"}

    # multivon-eval — module-level parse helpers used by Faithfulness:
    # _extract_json_array (claims/verdict arrays) and _parse_yes_no.
    try:
        from multivon_eval.evaluators.llm_judge import _extract_json_array
        paths["multivon-eval"] = {
            "reachable": True,
            "entry": "multivon_eval.evaluators.llm_judge._extract_json_array(raw)",
            "fn": lambda raw: _extract_json_array(raw),
        }
    except Exception as exc:
        paths["multivon-eval"] = {"reachable": False,
                                  "entry": f"import failed: {exc}"}

    # opik — the hallucination metric ships a standalone parser module.
    try:
        from opik.evaluation.metrics.llm_judges.hallucination.parser import (
            parse_model_output)
        paths["opik"] = {
            "reachable": True,
            "entry": "opik...hallucination.parser.parse_model_output(raw, name)",
            "fn": lambda raw: parse_model_output(raw, "a2"),
        }
    except Exception as exc:
        paths["opik"] = {"reachable": False,
                         "entry": f"import failed: {exc}"}

    # trulens — the groundedness parse logic is embedded inline in
    # LLMProvider.generate_score_and_reasons / groundedness_measure_with_
    # cot_reasons (structured-output JSON + provider retry), which require
    # a constructed provider+endpoint: NOT reachable standalone. Its
    # terminal regex fallback (re_configured_rating) IS importable and is
    # exercised here, explicitly labelled as fallback-only.
    try:
        from trulens.feedback.generated import re_configured_rating
        paths["trulens"] = {
            "reachable": "fallback-only",
            "entry": "trulens.feedback.generated.re_configured_rating(raw, "
                     "min_score_val=0, max_score_val=3) — terminal regex "
                     "fallback; the primary structured-output parse path is "
                     "embedded in LLMProvider methods (NOT-REACHABLE "
                     "standalone)",
            "fn": lambda raw: re_configured_rating(
                raw, min_score_val=0, max_score_val=3),
        }
    except Exception as exc:
        paths["trulens"] = {"reachable": False,
                            "entry": f"import failed: {exc}"}

    return paths


def compare_parsers() -> dict:
    canon = ragas_canonical_prompt()
    completions = {}
    if COMPARISON_PATH.exists():
        prev = json.loads(COMPARISON_PATH.read_text())
        completions = prev.get("completions", {})
        print("[resume] reusing previously fetched completions",
              file=sys.stderr)
    for judge in JUDGES:
        if judge not in completions:
            completions[judge] = _send_prompt_once(judge, canon["messages"])
            print(f"[sent] canonical prompt -> {judge} "
                  f"({completions[judge]['model']})", file=sys.stderr)

    paths = parse_paths()
    results: dict = {}
    for fw, spec in paths.items():
        row = {"reachable": spec["reachable"], "entry": spec["entry"]}
        if spec.get("fn"):
            for judge in JUDGES:
                row[f"parse_{judge}"] = _try(spec["fn"], completions[judge]["text"])
        results[fw] = row

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "design": "RAGAS canonical NLI judge prompt (fixed dev item "
                  "ragtruth_sum_0) sent once per primary judge at "
                  "temperature 0; the identical raw completion fed to each "
                  "framework's own parse path where standalone-reachable.",
        "canonical_prompt": {k: canon[k] for k in
                             ("source_file", "call_index", "response_format",
                              "tools")},
        "canonical_prompt_chars": len(canon["text"]),
        "completions": completions,
        "parser_results": results,
    }
    COMPARISON_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False)
                               + "\n", encoding="utf-8")
    print(f"wrote {COMPARISON_PATH}", file=sys.stderr)
    return out


# ── Report ───────────────────────────────────────────────────────────────────

RUBRIC_MARKERS = {
    "claims / truths extraction": ("claims", "truths"),
    "statement decomposition": ("statements",),
    "NLI verdict (0/1)": ("verdict",),
    "0-10 / 0-3 numeric rating": ("score of 0", "out of 10", "score of 3",
                                  "min_score", "0 - 3", "0 to 10"),
    "faithfulness wording": ("faithful",),
    "groundedness wording": ("grounded",),
    "hallucination wording": ("hallucinat",),
    "step-by-step / CoT ask": ("step by step", "step-by-step", "reasons",
                               "reasoning"),
    "few-shot examples embedded": ("example", "Example"),
}


def _fingerprint(doc: dict) -> dict:
    calls = [c for c in doc["calls"] if 200 <= c["status"] < 300]
    texts = [_messages_text(c["request_body"] or {}) for c in calls]
    total = "\n".join(texts)
    lower = total.lower()
    markers = [name for name, keys in RUBRIC_MARKERS.items()
               if any(k.lower() in lower for k in keys)]
    body0 = calls[0]["request_body"] if calls else {}
    return {
        "judge_calls": len(calls),
        "total_prompt_chars": len(total),
        "chars_per_call": [len(t) for t in texts],
        "uses_system_msg": any(
            (c["request_body"] or {}).get("system")
            or any(m.get("role") == "system"
                   for m in (c["request_body"] or {}).get("messages", []))
            for c in calls),
        "structured_output": any(
            (c["request_body"] or {}).get("response_format")
            or (c["request_body"] or {}).get("tools")
            or ((c["request_body"] or {}).get("text") or {}).get("format")
            for c in calls),
        "uses_responses_api": any("input" in (c["request_body"] or {})
                                  for c in calls),
        "temperature_sent": body0.get("temperature", "absent"),
        "rubric_markers": markers,
    }


def write_report(comparison: dict | None) -> None:
    lines = [
        "# A2 Prompt/Parser Attribution — Fallback Instrument Report",
        "",
        "Preregistered fallback path (plan §8 A2; decision recorded in "
        "PREREG_ADDENDUM.md §11): standalone prompt extraction + shared "
        "minimal parser comparison. The logging-proxy replay was NOT "
        "attempted — its 1-day time-box is spent.",
        "",
        "## Prompt-scaffold fingerprints (one fixed dev item, "
        "`ragtruth_sum_0`)",
        "",
        "| framework | judge | calls | prompt chars | system msg | "
        "structured output | responses API | temp sent | rubric markers |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for fw in FRAMEWORKS:
        for judge in JUDGES:
            p = PROMPTS_DIR / f"{fw}_{judge}.json"
            if not p.exists():
                lines.append(f"| {fw} | {judge} | — | — | — | — | — | — | "
                             f"(not captured) |")
                continue
            f = _fingerprint(json.loads(p.read_text()))
            lines.append(
                f"| {fw} | {judge} | {f['judge_calls']} | "
                f"{f['total_prompt_chars']:,} | "
                f"{'yes' if f['uses_system_msg'] else 'no'} | "
                f"{'yes' if f['structured_output'] else 'no'} | "
                f"{'yes' if f['uses_responses_api'] else 'no'} | "
                f"{f['temperature_sent']} | "
                f"{'; '.join(f['rubric_markers']) or '—'} |")
    lines += [
        "",
        "Full request bodies: `study/a2/prompts/{framework}_{judge}.json`.",
        "",
        "## Parse-path reachability + shared-completion comparison",
        "",
        "RAGAS's canonical NLI judge prompt was sent once per primary judge "
        "(2 calls, temperature 0); the identical raw completion was fed to "
        "each framework's own parse path where reachable standalone "
        "(`study/a2/parser_comparison.json`):",
        "",
        "| framework | parse path reachable? | entry point | "
        "parses RAGAS completion (gpt-4o-mini / claude-haiku-4-5) |",
        "|---|---|---|---|",
    ]
    res = (comparison or {}).get("parser_results", {})
    for fw in FRAMEWORKS:
        r = res.get(fw)
        if not r:
            lines.append(f"| {fw} | (not run) | | |")
            continue
        outcome = " / ".join(
            ("ok" if r.get(f"parse_{j}", {}).get("ok") else
             "FAIL" if f"parse_{j}" in r else "—") for j in JUDGES)
        lines.append(f"| {fw} | {r['reachable']} | {r['entry']} | {outcome} |")
    lines += [
        "",
        "## Limitation paragraph (for the paper)",
        "",
        "*Prompt/parser attribution (A2) used the preregistered fallback "
        "instrument, not the full logging-proxy replay: framework judge "
        "payloads were extracted verbatim at the HTTP transport for one "
        "fixed item, and a single shared completion (RAGAS's canonical NLI "
        "judge prompt answered once per judge at temperature 0) was offered "
        "to each framework's own parse path where that path is importable "
        "standalone. This design cannot fully separate prompt-scaffold "
        "effects from parser effects: frameworks whose parse logic is "
        "embedded in provider-bound methods (TruLens's structured-output "
        "path) or behind LLM-backed repair loops (RAGAS's fix-retry, "
        "DeepEval's schema re-ask) are exercised only at their first-attempt "
        "or terminal-fallback parsers, and a parser's failure on another "
        "framework's completion format demonstrates format coupling rather "
        "than parser quality. Cross-framework disagreement measured in the "
        "main study therefore remains a joint property of prompt scaffold, "
        "output-format contract, and parser — the ablation bounds, but does "
        "not decompose, their contributions.*",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true",
                    help="Part A: capture judge payloads (10 evals)")
    ap.add_argument("--compare", action="store_true",
                    help="Part B: shared minimal parser comparison (2 calls)")
    ap.add_argument("--report", action="store_true",
                    help="(re)write A2_REPORT.md from captured artifacts")
    args = ap.parse_args()
    if not any([args.extract, args.compare, args.report]):
        ap.error("choose at least one of --extract / --compare / --report")
    install_interceptor()
    if args.extract:
        extract_payloads()
    comparison = None
    if args.compare:
        comparison = compare_parsers()
    elif COMPARISON_PATH.exists():
        comparison = json.loads(COMPARISON_PATH.read_text())
    if args.report or args.compare:
        write_report(comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

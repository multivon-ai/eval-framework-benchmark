# A2 Prompt/Parser Attribution — Fallback Instrument Report

Preregistered fallback path (plan §8 A2; decision recorded in PREREG_ADDENDUM.md §11): standalone prompt extraction + shared minimal parser comparison. The logging-proxy replay was NOT attempted — its 1-day time-box is spent.

## Prompt-scaffold fingerprints (one fixed dev item, `ragtruth_sum_0`)

| framework | judge | calls | prompt chars | system msg | structured output | responses API | temp sent | rubric markers |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | gpt-4o-mini | 8 | 22,048 | no | no | no | absent | statement decomposition |
| multivon-eval | claude-haiku-4-5 | 8 | 22,059 | no | no | no | absent | statement decomposition |
| deepeval | gpt-4o-mini | 3 | 9,300 | no | yes | no | 0.0 | claims / truths extraction; NLI verdict (0/1); few-shot examples embedded |
| deepeval | claude-haiku-4-5 | 3 | 9,557 | no | no | no | 0.0 | claims / truths extraction; NLI verdict (0/1); few-shot examples embedded |
| ragas | gpt-4o-mini | 2 | 9,694 | no | no | no | 0.01 | statement decomposition; NLI verdict (0/1); faithfulness wording; few-shot examples embedded |
| ragas | claude-haiku-4-5 | 2 | 9,701 | no | yes | no | 0.0 | statement decomposition; NLI verdict (0/1); faithfulness wording; few-shot examples embedded |
| trulens | gpt-4o-mini | 4 | 15,161 | yes | yes | yes | 0.0 | claims / truths extraction; statement decomposition; 0-10 / 0-3 numeric rating; groundedness wording; few-shot examples embedded |
| trulens | claude-haiku-4-5 | 5 | 20,101 | yes | yes | no | 0.0 | claims / truths extraction; statement decomposition; 0-10 / 0-3 numeric rating; groundedness wording; few-shot examples embedded |
| opik | gpt-4o-mini | 1 | 5,206 | yes | yes | no | 0.0 | statement decomposition; faithfulness wording; hallucination wording; step-by-step / CoT ask |
| opik | claude-haiku-4-5 | 1 | 5,206 | yes | no | no | 0.0 | statement decomposition; faithfulness wording; hallucination wording; step-by-step / CoT ask |

Full request bodies: `study/a2/prompts/{framework}_{judge}.json`.

## Parse-path reachability + shared-completion comparison

RAGAS's canonical NLI judge prompt was sent once per primary judge (2 calls, temperature 0); the identical raw completion was fed to each framework's own parse path where reachable standalone (`study/a2/parser_comparison.json`):

| framework | parse path reachable? | entry point | parses RAGAS completion (gpt-4o-mini / claude-haiku-4-5) |
|---|---|---|---|
| multivon-eval | True | multivon_eval.evaluators.llm_judge._extract_json_array(raw) | ok / ok |
| deepeval | True | deepeval.metrics.utils.trimAndLoadJson(raw) | ok / ok |
| ragas | True | ragas.prompt.pydantic_prompt.RagasOutputParser(pydantic_object=NLIStatementOutput).parse(raw) [first attempt; LLM-backed fix-retry not exercised] | ok / ok |
| trulens | fallback-only | trulens.feedback.generated.re_configured_rating(raw, min_score_val=0, max_score_val=3) — terminal regex fallback; the primary structured-output parse path is embedded in LLMProvider methods (NOT-REACHABLE standalone) | ok / ok |
| opik | True | opik...hallucination.parser.parse_model_output(raw, name) | FAIL / FAIL |

## Limitation paragraph (for the paper)

*Prompt/parser attribution (A2) used the preregistered fallback instrument, not the full logging-proxy replay: framework judge payloads were extracted verbatim at the HTTP transport for one fixed item, and a single shared completion (RAGAS's canonical NLI judge prompt answered once per judge at temperature 0) was offered to each framework's own parse path where that path is importable standalone. This design cannot fully separate prompt-scaffold effects from parser effects: frameworks whose parse logic is embedded in provider-bound methods (TruLens's structured-output path) or behind LLM-backed repair loops (RAGAS's fix-retry, DeepEval's schema re-ask) are exercised only at their first-attempt or terminal-fallback parsers, and a parser's failure on another framework's completion format demonstrates format coupling rather than parser quality. Cross-framework disagreement measured in the main study therefore remains a joint property of prompt scaffold, output-format contract, and parser — the ablation bounds, but does not decompose, their contributions.*

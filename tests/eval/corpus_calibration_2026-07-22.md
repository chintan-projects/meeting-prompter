# Corpus coverage — local rater vs cloud judge, 21-question held-out set

> ## ⚠️ INVALID AS A MEASURE OF DISTILLATION — read this first
>
> After this run was recorded, the "distilled" corpus it scored was found to be
> **corrupt**: the local backend (F-702 v1) wrote the model's *task narration*
> ("Okay, I need to write a complete, self-contained answer…") into **63 of 88
> units (72%)** instead of answers. The bug was in `lib/corpus/local.py` — the
> `<think></think>` prefill does not reliably suppress this reasoning model's
> planning, and only tagged `<think>` blocks were being stripped.
>
> Therefore:
> - **`original` = 76% (16/21) is VALID** — no distiller touched that corpus.
> - **`distilled` = 81% (17/21) is VOID.** It scores a broken corpus, so the
>   "+5 pp" delta measures the bug, not distillation.
> - The **calibration** figures (local rater vs judge) are computed across both
>   corpora, so they are contaminated too; only the `original` column can be
>   trusted for rater comparison.
>
> Fixed by a self-containment contract check that rejects narration and falls back
> to the heuristic floor, with the reject rate reported in the distill stats.
> The per-question data below is kept as the record of what was measured.
>
> **Post-fix re-run (local rater only — the judge has NOT been re-run):** see
> *"Local-rater results after the fix"* at the end of this file.

**Date:** 2026-07-22 · **Question set:** [corpus_questions.yaml](corpus_questions.yaml) (21, held out from the distiller)
**Corpora:** original `on-device-capability-playbook.md` vs the **local-backend** distilled corpus
(88 units — 77 section + 11 topic; F-702 v1 prompted LFM2.5-2.6B)
**Raters:** shipped local heuristic (`lib.corpus.readiness.heuristic_rater`) and cloud judge
(`scripts/lab/judge.py`, Opus 4.8 — offline calibration instrument only, ADR-001)

Reproduce:

```bash
python -m scripts.lab.compare_corpus --questions-file tests/eval/corpus_questions.yaml --rater local
python -m scripts.lab.compare_corpus --questions-file tests/eval/corpus_questions.yaml --rater judge
```

## Per-question ratings

| # | Question (abbrev) | local·orig | judge·orig | local·dist | judge·dist |
|---|---|---|---|---|---|
| Q01 | INT4 cost + where it degrades | wrong | partial | partial | partial |
| Q02 | three levels of distillation | partial | partial | good | partial |
| Q03 | speculative decoding lossless + speedup | partial | partial | partial | **good** |
| Q04 | prune vs quantize, do they stack | partial | good | good | good |
| Q05 | why ternary needs QAT | good | good | partial | good |
| Q06 | GPTQ / AWQ vs plain INT4 | good | good | good | good |
| Q07 | logit-level distill from DeepSeek | partial | good | partial | good |
| Q08 | R1 reasoning-trace lesson | partial | good | good | good |
| Q09 | unstructured pruning speed | partial | good | partial | good |
| Q10 | self-consistency voting + cost | partial | good | good | good |
| Q11 | KV cache size at long context | good | partial | good | partial |
| Q12 | KV architectural fixes | partial | good | good | good |
| Q13 | LoRA merge vs hot-swap | partial | good | good | good |
| Q14 | QLoRA + ternary base | partial | good | good | good |
| Q15 | RAG vs facts in weights | good | good | good | good |
| Q16 | RAG for a quantized model | good | good | good | good |
| Q17 | MTP at train + inference | partial | good | partial | good |
| Q18 | MoE total vs active params | good | good | good | good |
| Q19 | router collapse | good | good | good | good |
| Q20 | LFM2 hybrid vs Mamba | good | good | good | good |
| Q21 | test-time compute trade curve | partial | partial | partial | partial |

## Coverage (% of questions with a borrowable `good`)

| Rater | Original | Distilled (local) | Delta |
|---|---|---|---|
| local heuristic (shipped) | 38% (8/21) | 67% (14/21) | **+29 pp** |
| cloud judge (trust gate) | **76% (16/21)** | **81% (17/21)** | **+5 pp** |

## Calibration — local rater vs judge (42 corpus×question cells)

| Metric | Value |
|---|---|
| Exact 4-way agreement | 24/42 = **57%** |
| Binary agreement (`good` vs not) | 25/42 = **60%** |
| Local **harsher** than judge | 15/42 = **36%** |
| Local **softer** than judge | 3/42 = **7%** |

The error is directional, not noise: the local rater under-rates 5× more often than it
over-rates, and understates the original corpus by 38 points.

## Judge view of what distillation actually changed

- Improved: **1** question (Q03 speculative decoding, partial → good)
- Regressed: **0**
- Unchanged: 20

Split by the E-03 probe subset:

| Subset | judge · original | judge · distilled |
|---|---|---|
| Q01–Q04 (E-03's original 4-question probe) | 1/4 = 25% | 2/4 = 50% |
| Q05–Q21 (the other 17) | 15/17 = 88% | 15/17 = 88% |

E-03's "25% original" reproduces **exactly** on its own 4 questions — that probe sampled the
corpus's four hardest (table/compound) cases, not a representative slice.

---

## Local-rater results after the fix (clean corpus, same 21 questions)

The distiller was re-run with the contract check and the heading-only guard active.
Resulting corpus: **83 units, 0 narration** — of which **25 came from the model and 59
fell back to the heuristic floor (70% reject rate)**.

| Corpus | local rater | notes |
|---|---|---|
| Original | 38% (8/21), 1 gap | unchanged baseline |
| Heuristic-distilled | 71% (15/21), 0 gaps | no model involved |
| **Local-distilled (clean)** | **76% (16/21), 0 gaps** | 30% model units / 70% heuristic |
| Local-distilled (contaminated) | 67% (14/21) | the bug cost ~9 points |

**Read this carefully — it is mostly not a measurement of the model.** 70% of the units
in the "local-distilled" corpus *are* the heuristic corpus. The 25 surviving model units
are worth about +5 points over heuristic-only under this rater (71% → 76%).

## What is still unknown

1. **The judge has not been re-run on the clean corpus.** The only valid judge numbers are
   for the *original* corpus (76%). Any statement of the form "distillation is worth N
   points, judge-scored" is currently unsupported.
2. **The cloud distiller has never been scored on these 21 questions** — so the earlier
   session's 25% → 75% (cloud, 4 questions) and everything here remain non-comparable.
   This is the decisive experiment: it isolates *backend quality* from *question sample*.
3. **The local rater disagrees with the judge by 38 points on the original corpus**, so its
   76% here must not be read against the judge's 76% there — different instruments (F-707).

Commands (2 needs a credential):

```bash
python -m scripts.lab.compare_corpus --questions-file tests/eval/corpus_questions.yaml --rater judge
python -m scripts.lab.distiller --backend cloud --mode consolidated
```

---

## ✅ DECISIVE RUN — cloud distiller, judge-scored, 21 questions (2026-07-22)

The experiment that isolates *backend quality* from *question sample*. Cloud distiller
(Opus 4.8) over the same 21 held-out questions, judged by Opus 4.8.

| Corpus | judge coverage | |
|---|---|---|
| Original | **76%** (16/21; 5 partial, 0 gap) | baseline |
| **Cloud-distilled** | **95%** (20/21; 1 partial, 0 gap) | **+19 pp** |

Four questions moved `partial → good`: three levels of distillation (the table),
speculative decoding, KV-cache size, test-time compute curve. **Zero regressions.**

**Distillation works.** This confirms the direction of the earlier 4-question probe
(25% → 75%) at 5× the sample and against a much stronger baseline. The interim claim
that distillation was "worth +5 pp" was an artifact of a corrupted local corpus and is
fully retracted.

### The lone holdout is the compound question — and its fix never ran

Q01 (INT4: "how much does it hurt AND where does it degrade") is the only remaining
`partial`. Its answer spans §1.3 + §1.9, which is exactly what the **topic-level unit**
tier exists to merge — and this run emitted only **2 of 11 topic units**
(`topics_failed: 9`), Part 1 among the failures. So 95% was reached *without* the
compound-question lever. The cause was an output-budget bug, not the model:
`max_output_tokens` budgeted ~1× the input's token count for a unit that *restates* its
input, so calls truncated marginally and non-monotonically with size. Now budgeted at
~2× input tokens with a doubling retry on truncation.

**Open:** re-run cloud distill and expect `topic_units: 11, topics_failed: 0`; Q01 may
then flip, putting the ceiling at or near 100%.

### What this means for the local backend (F-702)

The gap is now measured and unambiguous:

| Backend | judge coverage | status |
|---|---|---|
| Cloud (Opus) | **95%** | proven, but ADR-001 forbids it in the product |
| Local (prompted 2.6B) | not measurable | contract-valid on only ~30% of sections |
| Heuristic | not judge-scored | the fallback floor |

ADR-001 requires distillation to run on-device, so **the forge (F-702 v2) is the only
path to shipping the 95%**. It now has a measured target, a measured baseline, and a
demonstrated reason the prompt-only route cannot get there.

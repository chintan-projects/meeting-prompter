# Corpus coverage — local rater vs cloud judge, 21-question held-out set

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

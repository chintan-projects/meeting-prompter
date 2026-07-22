# Corpus Distillation — methodology, evidence, and limits

How the "Prepare corpus" step works, why it exists, what has actually been measured,
and what is still unproven. Implementation lives in `lib/corpus/`.

Related: [ADR-001](architecture/ADR-001-local-corpus-distiller.md) (local-only decision)
· [open-decisions-log.md](architecture/open-decisions-log.md) D-08/D-09/D-11
· [corpus-prep-onboarding-spec.md](architecture/corpus-prep-onboarding-spec.md)

---

## 1. Why distillation exists

The live loop is **retrieval-first** (D-08). What appears on a card during a meeting is
a span of the user's own corpus, shown verbatim — there is no model between retrieval
and the screen. That is a deliberate trade: it buys grounding, provenance, and ~16–190 ms
latency, and it pays for them with a hard dependency.

> In a verbatim-retrieval product, **corpus quality is the ceiling on output quality.**
> No model papers over a weak source, because no model is in the path.

And most real corpora are the wrong shape. People write **explainers** — prose that
teaches a topic across paragraphs, tables, and cross-references. A meeting needs an
**answer bank** — statements a person could read aloud, standing alone, without the
surrounding page. Retrieval over an explainer returns *context*; it rarely returns
something borrowable.

Distillation reshapes the first into the second, once, offline.

```
Explainer section                    →   Answer-unit
────────────────────────────────────────────────────────────────
"## 1.3 Quantization levels          →   "Quantization has three levels: INT8
 The table below shows the three          keeps ~99% of quality at half the
 supported levels and their               size, INT4 keeps ~95% at a quarter,
 tradeoffs. As discussed above,           and INT2 is research-only. Use INT8
 the choice depends on…                   for production, INT4 when memory is
 | Level | Quality | Size |               the binding constraint."
 …"                                       _Source: playbook.md › Part 1 > 1.3_
```

Three properties define a valid unit:

1. **Self-contained** — resolves its own pronouns, names its subject, and never refers
   to the document ("this section", "as discussed above", "the text describes").
   A speaker reading it aloud must make sense to a listener who has never seen the doc.
2. **Grounded** — every claim traceable to the source section. No outside knowledge.
3. **Provenance-tagged** — each unit carries a `_Source: <doc> › <heading path>_` line,
   surfaced in the UI as expand-to-source. Text that cannot be attributed is never shown.

---

## 2. The pipeline

```
source docs                lib/corpus/incremental.py
(.md/.txt/.pdf)     ──►    distill_dir()  ── content-hash manifest,
                            │              only changed docs re-run,
                            │              orphan outputs removed
                            ▼
                     lib/corpus/distiller.py
                            │
              CompositeParser → sections
                            │
              ┌─────────────┴─────────────┐
              │                           │
     section-level units          topic-level units
     (one per section)            (one per multi-section Part)
              │                           │
              └─────────────┬─────────────┘
                            ▼
                  data/distilled/*.distilled.md
                            │
                            ▼
              lib/corpus/readiness.py  ── score + gap list
                            │
                            ▼
              lib/corpus/active.py     ── activate for live use
                       (own index DB, applies next session start)
```

### 2.1 Backends

Selected per run; all three implement the same `(heading, text, mode) → list[str]`
interface, so they are interchangeable behind the wizard.

| Backend | Model | Status | Role |
|---|---|---|---|
| `heuristic` | none | shipped | Quality **floor**. Cleans markdown, prepends the topic title, keeps the section verbatim. Grounded by construction, but cannot reshape — it provably cannot turn a table into speakable prose. |
| `local` | on-device LFM2.5-2.6B, prompted | **shipped default** (ADR-001) | Reads structure (tables included) and writes one consolidated answer per section. Falls back to the heuristic floor per section on failure. |
| `cloud` | Claude Opus 4.8 | **offline dev only — rejected by the API** | Validation instrument and training-data generator. `POST /corpus/distill` refuses `backend="cloud"`; the only entry point is the lab CLI. |

The local backend is the shipped default because of a privacy decision, not a quality
one — see [ADR-001](architecture/ADR-001-local-corpus-distiller.md). The distiller
processes the user's entire private corpus, which is the most sensitive data in the
system, so it runs on-device. Cloud may later become an **optional, consent-gated**
quality toggle, defaulting off, with the same posture as Notion export. It will never
be the default.

### 2.2 Modes

| Mode | Output | Trade |
|---|---|---|
| `atomic` | 1–5 short facts per section, capped at 60 words | Glanceable, but **fragments compound answers** — "the three levels *and* when to use each" splits across units and neither half is borrowable alone. |
| `consolidated` | one complete self-contained answer per section | Product default. Completeness over glanceability. |

**Topic-level units** are the compound-question lever. When a Part spans two or more
sections, its concatenated text is distilled into one additional unit, so a question
whose answer lives half in §1.3 and half in §1.9 still finds a single borrowable span.
These are always consolidated regardless of mode.

Their known limit: a topic unit for a large Part can run ~900 words — retrieved
correctly, but too long to read aloud. Bounding topic units to roughly 150 words is a
known, deliberately unmade change: it needs a validation run, not a guess.

### 2.3 The self-containment contract check

The local backend does not trust its own output. Each unit passes a regex contract
check (`lib/corpus/local.py`, `looks_like_meta()`) that rejects task narration and
document self-reference. A rejected unit falls back to the heuristic floor for that
section, and the run reports `local.reject_pct` in its stats.

This exists because of a real failure, and the lesson generalizes:

> An early local run wrote task narration — *"Okay, I need to write a complete,
> self-contained answer…"* — into **63 of 88 units (72%)**. Every gate was green. The
> full test suite passed. The corpus looked fine in aggregate statistics. It was
> discovered only by reading the units, and it had already invalidated a coverage
> number that was committed to the decisions log and an ADR.

Prompt hardening was tried first and produced *different* narration, which is what
established that prompt-only correction is not a viable control for this. The check is
a structural gate, not a filter of last resort. Its reject rate is reported rather than
hidden, because a corpus where the model produced 30% of the units and the heuristic
floor produced the rest **looks model-made in every summary statistic** unless the split
is stated.

### 2.4 Incremental re-distillation

`lib/corpus/incremental.py` keeps a content-hash manifest (`.distill_manifest.json`).
Only changed documents re-distill; outputs orphaned by a deleted source are removed; a
change of backend or mode invalidates everything, since mixing backends in one corpus
would make its provenance meaningless.

---

## 3. Readiness: the onboarding gate

Distillation without measurement is faith. The readiness score (D-11,
`lib/corpus/readiness.py`) answers the question a user actually has before their first
call: **can this corpus answer my meetings?**

```
questions → retrieval (top-k) → borrowable cards → rater → score + gap list
```

- **Cards** are cleaned, answer-shaped spans; multi-unit answers merge the top 2, taking
  the *minimum* of the parts' confidences and keeping per-unit provenance.
- **The shipped rater is local and heuristic** — answer-shapedness, retrieval confidence,
  and question-term overlap, with thresholds mirroring the live config's semantics
  (`GOOD_COSINE=0.60`, `PARTIAL_COSINE=0.35`, matching `rag_confidence_minimum=0.35`).
- **Output** is `{score_pct, good, partial, gap, gaps[]}`. The gap list is the point:
  it tells the user which questions their corpus cannot answer, while there is still
  time to add a document.

The rater is honest about being a proxy. The trustworthy instrument is a cloud
LLM-as-judge (`scripts/lab/judge.py`), calibrated against human ratings, and it stays
**offline** — it is a development instrument, never in the product path.

**The shipped rater is currently miscalibrated and this is tracked, not hidden**
(F-707): 57% agreement with the judge, systematically harsh — wrong in the strict
direction in 36% of cells versus 7% soft — understating a known-good corpus by 38
percentage points. It must be recalibrated against a **separate development question
set** before its number is presented to a user as a verdict.

---

## 4. What has been measured

Held-out set: 21 questions in `tests/eval/corpus_questions.yaml`. Full audit trail:
[tests/eval/corpus_calibration_2026-07-22.md](../tests/eval/corpus_calibration_2026-07-22.md).

| Configuration | Judge-scored coverage |
|---|---|
| Original corpus (well-written explainer) | 76% |
| Cloud-distilled (Opus 4.8, consolidated + topic units) | **90–95%** |
| Local-distilled (prompted 2.6B) | not meaningfully measurable — contract-valid on only ~30% of sections |

**The honest figure is 76% → 90–95%, a lift of +14 to +19 points.** Two runs of the
same configuration produced 95% and 90%: the judge is stable (the `original` column was
byte-identical across runs) while cloud generation is not, and at n=21 a single question
is worth 4.8 points. Quoting the 95% alone would be quoting the better of two draws.

Three constraints on how far this generalizes:

- **One corpus, and a well-written one.** A 76% baseline means the source was already a
  strong explainer. Value on a *messy* corpus — Notion exports, raw meeting notes, slide
  dumps — is **untested** (F-708). That is where a distiller should help most, and where
  it has not yet been tried.
- **The question set is held out from the distiller by rule.** The distiller extracts
  generally; it is never tuned against these 21 questions. Coverage that reflected a
  gamed corpus would be worthless.
- **The proven number is the cloud path, which ADR-001 forbids shipping.** This is the
  central tension: 95% is real and reachable, and the privacy decision says it must be
  reached on-device.

---

## 5. Where this goes

**F-702 v2 — forge the local specialist.** The prompted 2.6B is contract-valid on only
~30% of sections, so prompt engineering cannot close the gap; the rest falls to the
heuristic floor. The path is to fine-tune a small specialist on cloud-distilled
section→unit pairs (LEAP/forge), targeting the section → grounded-answer-unit task
including table reading. There is a pleasing symmetry to it: the product's corpus
preparation would itself be a distilled small model, demonstrating on the user's own
content exactly the value proposition the stack exists to prove.

**F-707 — recalibrate the readiness rater** against a separate dev question set, so the
score can be shown as a verdict rather than a hint.

**F-708 — validate on a messy corpus.** This gates the forge investment. On a
well-structured explainer the baseline is already 76%, so the remaining headroom is
thin; the case for a specialist model rests on corpora that are *not* well structured,
and that case is currently an assumption.

**Bounded topic units** (~150 words) so compound answers stay borrowable, and a
re-run to confirm the anti-self-reference prompt rule takes — 52% of cloud units still
carried "Section N, titled…" framing when last measured, which matters doubly because
those units are the forge's training data.

---

## 6. Working on the distiller

```bash
# via the product wizard
#   Meeting Setup → "Prepare corpus…"

# via the API
curl -sX POST localhost:8420/corpus/distill \
  -H 'content-type: application/json' -d '{"backend":"local","mode":"consolidated"}'
curl -s localhost:8420/corpus/distill/status | jq

# offline lab (the only cloud entry point — never the product path)
./scripts/lab/run.sh
python -m scripts.lab.compare_corpus --help
```

Tests: `tests/test_corpus_distiller.py`, `test_corpus_readiness.py`,
`test_corpus_incremental.py`, `test_corpus_active.py`, `test_corpus_routes.py`.

Two rules that are not negotiable when changing any of this:

1. **Never tune the distiller against `tests/eval/corpus_questions.yaml`.** It is the
   held-out measurement instrument. Use a separate development set.
2. **Never emit a unit without provenance.** Un-attributable text does not reach the
   screen — it is the whole basis of trusting a card mid-meeting.

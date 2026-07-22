# Spec — "Prepare corpus" onboarding flow

**Status:** Draft (2026-07-21) · **Priority:** V1.5 (the productization of the lab work)
**Related:** [ADR-001](ADR-001-local-corpus-distiller.md) (local distiller),
[open-decisions-log.md](open-decisions-log.md) D-08/D-09/D-10/D-11,
`scripts/lab/` (the working prototype of every component below)

## One-liner

**Bring your knowledge → we make it meeting-ready.** A one-time flow that turns a
user's raw docs into a borrowable, retrieval-first corpus — and tells them, honestly,
whether their content can actually answer their meetings before they rely on it live.

## Why this is the product, not a feature

Every generic RAG tool indexes your docs blindly and hopes. This flow **measures corpus
fitness** (the judge/coverage instrument) and gates on it: it shows *where* your content
can and can't answer your questions, then reshapes it so it can. That readiness score is
the differentiator.

## The flow (wizard)

```
1. Add sources ─▶ 2. Distill ─▶ 3. Readiness check ─▶ 4. Ready (index → live)
   folder/upload     reshape to      run likely Qs →       borrowable answers
   /Notion           answer-units    fit-for-purpose        available in calls
                     + provenance     score + gap list
```

**1 · Add sources.** Point at a folder, upload files, or pull from Notion. Reuses the
existing ingest (`context/` docs, `lib/notion/` ingest). Supports md / PDF / text.

**2 · Distill (one-time, on-device).** Each source section → self-contained, grounded
answer-unit(s) with a provenance pointer back to the source. Runs on a **local small
model** (ADR-001). Progress UI; re-runnable when docs change.

**3 · Readiness check.** Run the user's *likely questions* through the distilled corpus
and score each retrieved answer (borrowable / partial / gap). Show:
- a **fit-for-purpose score** (% of questions with a borrowable answer), and
- a **gap list** ("no borrowable answer for: pricing, SLA terms, …") so the user knows
  exactly what to add — content gap vs. a fixable shape gap.

**4 · Ready.** Index the distilled corpus; it's now the retrieval source for live calls.

## Live (what the prepared corpus powers)

Per call, retrieval-first: retrieve the borrowable answer-unit + source, glanceable,
user-gated (select-to-answer / armed window — D-02). No live generation. ~120–190ms.

## Component map — prototype → product

| Piece | Prototype (built) | Product home (to build) |
|---|---|---|
| Distiller (reshape → units + provenance) | `scripts/lab/distiller.py` (cloud + heuristic, consolidated mode, reads tables) | `lib/corpus/distiller.py` with a **local** backend (ADR-001) |
| Index | existing `lib/rag/` pipeline over the distilled corpus | same; distilled corpus is the indexed source |
| Readiness / coverage | `scripts/lab/pipeline.py::aggregate_coverage` + `judge.py` | `lib/corpus/readiness.py` (local judge or heuristic proxy) |
| Provenance | already emitted per unit (`_Source: doc › section`) | surfaced in the live card ("expand to source") |
| Live borrowable view | lab borrowable cards + retrieval-first view | `PromptsPane` / a new live view in the Tauri app |
| Wizard UI | — | new "Prepare corpus" flow (extends `MeetingSetup.tsx` / a Corpus manager) |

## Data flow

```
docs ─▶ parse (sections) ─▶ distill (local model) ─▶ answer-units (+provenance)
                                                          │
                                                    index (FTS5 + vector)
                                                          │
   likely questions ─▶ retrieve ─▶ readiness score + gaps │
                                                          ▼
                              live call: retrieve borrowable unit + source
```

## Open questions / levers (tracked in the decisions log)

- **Question set for readiness** — user-provided vs. auto-generated (auto-generating
  likely questions is itself a distillation-style step). → D-11.
- **Local judge for readiness** — ship a local judge or a heuristic proxy so the
  readiness check is also on-device. → ADR-001 follow-on.
- **Cross-section grouping** — compound questions whose answer spans two sections
  (the INT4 "how much + where" case) need topic-level grouping or multi-unit answers.
  → E-03 lever.
- **Incremental re-distill** — only re-distill changed docs.
- **Cloud opt-in** — optional consent-gated "higher quality" distill toggle (ADR-001).

## Rollout

V1.5 slice = local distiller + Prepare wizard + readiness score, wired to the existing
RAG index and the retrieval-first live view. Everything upstream of the UI is already
prototyped in `scripts/lab/`; the productization is (a) the local distiller model and
(b) the wizard UI.

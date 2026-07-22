# ADR-001 — The corpus distiller runs on a local small model

**Status:** Accepted (2026-07-21)
**Deciders:** Chintan
**Related:** [open-decisions-log.md](open-decisions-log.md) D-08 (retrieval-first), D-09
(Prepare-corpus flow), D-10 (this decision); [corpus-prep-onboarding-spec.md](corpus-prep-onboarding-spec.md)

## Context

The product is retrieval-first (D-08): in a meeting, we retrieve a borrowable span of
the user's corpus and show it — no live generation. That makes **corpus quality the
ceiling on output quality**. The lab's judge (Opus 4.8, calibrated against human
ratings) confirmed the raw source docs are *explainers*, not answer banks, so they
rarely produce a directly borrowable answer.

The **distiller** fixes this by reshaping each source section into self-contained,
grounded answer-units with a provenance pointer. Validated in the lab: distillation
lifted borrowable-answer coverage from **25% → 50%** on a 4-question probe, with the
consolidated mode + table-reading fix expected to go further.

Distillation is a **one-time, offline preparation step** (per corpus, per doc change),
not part of the live loop. It therefore has **no latency budget** — the constraint
that forces small-on-device models in the live path does not apply here.

Two hard requirements shape the decision:
- **Privacy / local-first.** The product runs all inference locally; the only
  consent-gated network egress today is Notion export. The distiller processes the
  user's *entire private corpus* — the most sensitive data in the system.
- **On-device is the core competency.** Reshaping a section into a grounded answer
  ("read this, write the borrowable statement") is squarely within a small,
  purpose-tuned model's ability — and small-model distillation is exactly what this
  stack (LEAP / forge) exists to do.

## Decision

**The shipped corpus distiller runs on a local small model, on-device. No user corpus
leaves the machine in the product.**

Cloud Claude (Opus 4.8) is used **only offline, during development** — to validate the
approach and to generate training data for the local distiller model. It is **not** in
the shipped data path.

A cloud distiller MAY be offered later as an **optional, consent-gated "higher quality"
toggle** (same posture as Notion export), but **local is the default and the promise**.

## Consequences

**Positive**
- Fully local and private — the corpus never leaves the device; consent-gated egress
  stays limited to Notion export.
- No per-document API cost; distillation is free to re-run as docs change.
- On-brand: the distiller is itself a distilled small model — the product demonstrates
  the LEAP/forge value proposition on the user's own content.

**Costs / risks**
- We must **build (forge) a local distiller model** — this is real work, not free.
- Local quality < Opus initially, so the coverage lift may be smaller until the model
  is tuned. Mitigation: bootstrap from cloud-distilled outputs as training data.
- The **readiness eval** (judge/coverage) also uses cloud today; the shipped readiness
  check should move to a local judge or a heuristic proxy over time (follow-on, not
  blocking — readiness can start cloud-gated/optional while distillation is local).
- Content whose answers live in tables/structure needs the model to read structure
  (the cloud path already does; the local model must be trained for it).

## Alternatives considered

- **Cloud distiller in the product** — rejected: sends the private corpus off-device,
  adds per-doc cost, breaks the local-first promise.
- **Heuristic-only distiller (no model)** — rejected: proven insufficient. The
  heuristic pass cannot prose-ify tables or reshape explanatory prose into answers
  (it could not recover the "three levels" table). Good for cleaning, not for
  answer-shaping.
- **Cloud as default with local opt-in** — rejected as the default posture; inverts
  the privacy promise. Cloud remains a possible *optional* quality toggle only.

## Path

1. **Validate with cloud** (done / in progress) — prove distillation moves coverage.
2. **Forge a local distiller model** — use cloud-distilled outputs as training data
   (LEAP/forge); target the section → grounded-answer-unit task, including table reading.
3. **Ship local as default**; keep cloud as an optional consent-gated quality toggle.
4. **Move readiness eval local** (local judge or heuristic) as a follow-on.

# F-503 — Trigger-router head: S1 contract

**Task (output-shape sentence):** the output of this step is **one label** ∈
`{question, alert, topic, followup, none}` for a single meeting utterance.

**Input shape:** one bare ASR turn (a short transcript utterance — lowercased,
disfluent, often unpunctuated). The head serves a bare turn; no family/subject line
is prepended, so families are *conditioning only* (no train/serve skew).

**Task class:** classification (single-label, 5-way, `none` = abstain).

**Backbone:** bidirectional **LFM2.5-Encoder-350M** (`sequence_classifier_encoder`,
mean-pool, CE). The mask question answers "yes": routing a turn correctly needs the
whole utterance in view (a tag-question cue at the *end* flips `question`→`none`), so
the read must be bidirectional — never the causal tower.

**Why this over F-510:** the F-510 frozen linear probe scored macro-F1 0.886 on 70
hand-written rows but could not clearly beat the heuristic on the one class it owns
(`question`). F-503 is the measured upgrade: a LoRA-trained encoder head on a
forge-generated, corpuscope-gated, ASR-register corpus (far better powered than 70
clean rows). It ships **only if** it beats the F-510 probe / heuristic on a frozen
held-out split **without regressing `question`** — same gate discipline as F-510.

---

## Selection rules (numbered, priority order)

Apply top-down; the first rule that fires wins. `none` is the floor.

1. **alert** — the utterance carries a heads-up the listener must act on *now*: a
   risk, a contradiction of prior claims, a confidentiality/NDA flag, a deadline
   change, a budget/latency-budget breach, a "customer said X is critical", or a
   named watch-word warning. Framing is directive/cautionary ("heads up", "careful",
   "watch out", "reminder:", "at risk", "contradicts", "flagged").

2. **question** — a genuine, answer-worthy question directed at the room, seeking
   information retrievable from docs/knowledge. Interrogative form OR embedded ask
   ("remind me what X was", "do we have benchmarks for Y").

3. **followup** — a coaching nudge / action-suggestion for *later*, not now: "we
   should circle back on X", "let's make sure to ask them about Y", "it might be
   worth mentioning Z". Deontic/suggestive, future-facing, first-person-plural.

4. **topic** — a *neutral declarative* statement of a factual claim about the
   product/domain that a reference doc could corroborate ("the hybrid RAG uses FTS5
   and vector fusion", "our model runs on-device"). No risk framing, no ask, no
   suggestion — just a fact being discussed.

5. **none** — backchannel, filler, agreement, chit-chat, or any turn with no
   routable intent ("yeah totally", "right right exactly", "okay sounds good").
   **Abstain here** rather than force a weak label.

## Exclusion zones (what each label is NOT)

- `question` excludes **rhetorical / tag / self-answering** forms: "right?", "you
  know?", "makes sense?", "isn't it?", and questions the speaker immediately answers.
  These are `none` (a tag on an otherwise-empty turn) or take the label of their
  content clause. This mirrors the F-201 rhetorical-suppression contract.
- `alert` excludes neutral facts (→ `topic`) and future suggestions (→ `followup`).
  The discriminator is *actionable-now risk*, not merely mentioning a customer/number.
- `topic` excludes anything with risk framing (→ `alert`), an ask (→ `question`), or
  a suggestion (→ `followup`). A fact stated *as a warning* is `alert`, not `topic`.
- `followup` excludes immediate heads-ups (→ `alert`) and direct questions
  (→ `question`). "let's ask them about their timeline" is `followup` (suggesting a
  future ask); "what's their timeline?" is `question` (asking now).

## Tie-breaks (the confusable pairs — these become two-sided cues)

- **question vs followup**: a direct interrogative to the room *now* → `question`;
  a suggestion to raise/ask something *later* → `followup`. Both may contain
  "ask"/"timeline"/"pricing". Cue: *interrogative-now* vs *suggestion-for-later*.
- **alert vs topic**: same subject, different framing — "the Verizon renewal is at
  risk" → `alert`; "the Verizon contract renews in Q3" → `topic`. Cue:
  *risk/heads-up framing* vs *neutral declarative*.
- **question vs none**: an answer-worthy interrogative → `question`; a tag/rhetorical
  question or backchannel → `none`. Cue: *genuine information-seeking* vs
  *tag/rhetorical/filler*.
- **topic vs followup**: stating a fact now → `topic`; suggesting to mention a fact
  later → `followup` ("worth mentioning the on-device angle"). Cue: *assert-now* vs
  *suggest-to-mention-later*.

## Abstain definition

`none` is the explicit abstain class: emit it when no routable intent is present, or
when a turn is only a tag/backchannel. Precision on the four active classes matters
more than recall — a missed trigger is silent; a wrong trigger is noise in the
operator's face. When genuinely ambiguous between an active class and `none`, prefer
`none`.

## Serving / skew notes

- Register is **voice-transcript** (ASR): lowercase, disfluencies, missing terminal
  punctuation. This is *conditioning* fed to the teacher, **not** stored in the input
  — the served turn is bare. This is the deliberate fix for the F-510 clean-text skew.
- Families (customer call · eng standup · GTM review · roadmap) are `role:
  conditioning`, rotated so no subject locks to a label.

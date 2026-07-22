"""Corpus distiller — reshape an explainer doc into borrowable answer-units (F-701).

The judge showed the source docs are *explainers* (prose about how things work),
not answer banks — so retrieval returns context, not borrowable answers. This
distiller walks a doc section by section and emits self-contained, grounded
answer-units, each carrying a provenance pointer back to its source section. The
distilled corpus is indexed like any other; retrieval and the models are unchanged.

Backends (ADR-001):
  - heuristic: clean + topic-title each section (no LLM; grounded verbatim; runnable
    with no credential — proves the pipeline).
  - cloud: Claude extracts crisp, self-contained statements from each section,
    grounded ONLY in the section text. OFFLINE validation / training-data only —
    never the shipped default (see lib.corpus.cloud).
  - local (F-702 v1): on-device prompted model (lib.corpus.local) — the shipped
    default per ADR-001; the forged fine-tuned specialist replaces the prompt
    behind the same interface. Per-section heuristic fallback = quality floor.

Modes:
  - atomic: 1-5 short facts per section (glanceable, but fragments compound answers).
  - consolidated (default in the CLI): one complete, self-contained answer per
    section, plus one topic-level unit per multi-section Part so compound questions
    whose answer spans sections still get a single borrowable unit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from lib.answer_extractor import extract_sentences
from lib.corpus import cloud
from lib.corpus.text import clean_markdown
from lib.rag.parser.composite_parser import CompositeParser
from lib.rag.types import ParsedSection

logger = logging.getLogger(__name__)

MIN_SECTION_WORDS = 12  # below this a section is navigation/heading — no borrowable claim
MAX_UNIT_WORDS = 60  # keep atomic units glanceable
MAX_SECTION_CHARS = 6000  # cap section text sent to the cloud extractor (token budget)
MAX_TOPIC_CHARS = 12000  # cap the concatenated Part text for a topic-level unit

BACKENDS = ("heuristic", "local", "cloud")

_CLOUD_SYSTEM = """You extract BORROWABLE answer-statements from a documentation section \
for a meeting-assistant corpus. A borrowable statement is one a speaker could read aloud \
to answer a question.

Rules:
- Ground every statement ONLY in the SECTION text. Do not add outside knowledge.
- Make each statement fully self-contained: resolve pronouns, name the subject, no "this"/"it"/"the above" that refers outside the statement.
- Prefer crisp factual claims (definitions, numbers, when-to-use, tradeoffs) over narration.
- Return 1-5 statements; fewer is fine. If the section is pure heading/navigation/table with no borrowable claim, return an empty list."""

# consolidated mode: ONE complete answer per section (fixes compound questions that
# atomic extraction fragments — e.g. "the three levels AND when to use each").
_CLOUD_SYSTEM_CONSOLIDATED = """You write ONE complete, self-contained answer that \
captures everything borrowable in a documentation section — a speaker could read it aloud \
to answer questions about this topic.

Rules:
- Ground it ONLY in the SECTION text. Do not add outside knowledge.
- Cover ALL the key facts in the section in one coherent answer: every item in a list, \
every level/option, every number and when-to-use — don't drop any.
- Make it fully self-contained: name the subject, resolve pronouns, no "this"/"it" that \
refers outside the answer.
- If the section is pure heading/navigation with no borrowable content, return an empty string."""

_CLOUD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"units": {"type": "array", "items": {"type": "string"}}},
    "required": ["units"],
    "additionalProperties": False,
}
_CLOUD_SCHEMA_CONSOLIDATED: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _topic_prefix(heading: str, body: str) -> str:
    """Prepend the section topic unless the body already opens with it (self-contained)."""
    topic = heading.strip().rstrip(".")
    return body if topic.lower() in body.lower()[: len(topic) + 20] else f"{topic}: {body}"


def _distill_heuristic(heading: str, text: str, mode: str = "atomic") -> list[str]:
    """No-LLM backend: clean the section and topic-title it (grounded verbatim)."""
    cleaned = clean_markdown(text)
    if len(cleaned.split()) < MIN_SECTION_WORDS:
        return []
    if mode == "consolidated":
        # Keep the whole cleaned section as one unit — completeness over glanceability.
        return [_topic_prefix(heading, cleaned)]
    # atomic: lead with the topic, then the strongest sentences up to the word cap.
    sentences = extract_sentences(cleaned) or [cleaned]
    body_words: list[str] = []
    picked: list[str] = []
    for s in sentences:
        picked.append(s)
        body_words += s.split()
        if len(body_words) >= MAX_UNIT_WORDS:
            break
    return [_topic_prefix(heading, " ".join(picked))]


def _distill_cloud(heading: str, text: str, mode: str = "atomic") -> list[str]:
    """Cloud backend (offline/opt-in): Claude reshapes the RAW section into units.

    The model gets the raw section, tables/code intact — much of the answer content
    (e.g. the "three levels" table) lives in markdown tables that clean_markdown
    strips. Claude reads tables natively and reshapes them into prose. Skip only
    true navigation/heading stubs (guard on raw length, not the stripped length,
    so a table-heavy section isn't dropped).
    """
    import anthropic

    if len(text.split()) < MIN_SECTION_WORDS:
        return []
    consolidated = mode == "consolidated"
    system = _CLOUD_SYSTEM_CONSOLIDATED if consolidated else _CLOUD_SYSTEM
    schema = _CLOUD_SCHEMA_CONSOLIDATED if consolidated else _CLOUD_SCHEMA
    user = f"SECTION: {heading}\n\n{text[:MAX_SECTION_CHARS]}"
    try:
        resp = cloud.get_client().messages.create(
            model=cloud.CLOUD_MODEL,
            max_tokens=900 if consolidated else 700,
            system=system,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        )
        txt = next(b.text for b in resp.content if b.type == "text")
        data = json.loads(txt)
        if consolidated:
            answer = str(data.get("answer", "")).strip()
            return [answer] if answer else []
        units = data.get("units", [])
        return [u.strip() for u in units if isinstance(u, str) and u.strip()]
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        # Credential problems are fatal for the whole run — re-raise so distill()
        # aborts instead of silently emitting an empty corpus, section after section.
        raise
    except (anthropic.APIError, json.JSONDecodeError, KeyError, StopIteration) as e:
        # A single section failing (rate limit blip, malformed output) degrades that
        # section only; log with context and continue.
        logger.warning("cloud extract failed on section %r: %r", heading, e)
        return []


def _part_of(section: ParsedSection) -> str:
    """The top-level Part a section belongs to (first segment of its heading path)."""
    hp = section.heading_path or section.heading or ""
    return hp.split(" > ")[0].strip() or (section.heading or "").strip()


def _group_by_part(sections: list[ParsedSection]) -> dict[str, list[ParsedSection]]:
    """Group sections under their level-1 Part, preserving order."""
    groups: dict[str, list[ParsedSection]] = {}
    for sec in sections:
        groups.setdefault(_part_of(sec), []).append(sec)
    return groups


def _emit(lines: list[str], src_name: str, heading: str, unit: str, prov: str) -> None:
    """Append one answer-unit block (heading, unit, provenance line) to the output."""
    lines.extend([f"## {heading}", "", unit, "", f"_Source: {src_name} › {prov}_", ""])


def _distill_local(heading: str, text: str, mode: str = "consolidated") -> list[str]:
    """Local on-device backend (F-702): the shipped default per ADR-001.

    Always consolidated (one complete answer per section) — the product mode;
    the ``mode`` parameter is accepted for interface parity and ignored.
    """
    from lib.corpus.local import get_local_distiller

    return get_local_distiller().distill_section(heading, text)


def _backend_fn(backend: str) -> Callable[[str, str, str], list[str]]:
    """Resolve a backend name to its section→units function (fail fast on typos)."""
    if backend == "cloud":
        hint = cloud.credential_hint()
        if hint is not None:
            # Fail fast rather than making 80 doomed API calls that each return [].
            raise RuntimeError(f"cloud backend needs a credential: {hint}")
        return _distill_cloud
    if backend == "local":
        from lib.corpus.local import get_local_distiller

        if not get_local_distiller().available():
            raise RuntimeError(
                f"local backend needs the generation model on disk "
                f"(missing: {get_local_distiller().model_path}); "
                "set MODELS_DIR/CORPUS_LOCAL_MODEL or use backend='heuristic'"
            )
        return _distill_local
    if backend == "heuristic":
        return _distill_heuristic
    raise ValueError(f"unknown distiller backend {backend!r} (choose from {BACKENDS})")


def distill(
    src: Path, out: Path, backend: str = "heuristic", mode: str = "atomic"
) -> dict[str, Any]:
    """Distill one source document into a provenance-tagged answer-unit markdown file.

    Args:
        src: source document (anything CompositeParser reads — .md, .pdf, ...).
        out: output markdown path; parent directories are created.
        backend: "heuristic" (no LLM) or "cloud" (offline/opt-in, ADR-001).
        mode: "atomic" (1-5 short facts/section) or "consolidated" (one complete
            answer/section). Topic-level units are always consolidated.

    Returns:
        Run stats: backend, mode, sections_used, topic_units, units,
        sections_empty, out.
    """
    fn = _backend_fn(backend)
    doc = CompositeParser().parse(src)
    lines: list[str] = [f"# Distilled: {src.name}", ""]
    n_units = n_sections = n_failed = n_topics = 0

    # 1. Section-level units — specific answers.
    for sec in doc.sections:
        heading = sec.heading or "(root)"
        units = fn(heading, sec.content, mode)
        if not units:
            if len(clean_markdown(sec.content).split()) >= MIN_SECTION_WORDS:
                n_failed += 1  # substantive section produced nothing (likely a failure)
            continue
        n_sections += 1
        for u in units:
            n_units += 1
            _emit(lines, src.name, heading, u, sec.heading_path or heading)

    # 2. Topic-level units — one consolidated answer per multi-section Part, so
    #    compound questions whose answer spans sub-sections (e.g. INT4 "how much"
    #    in 1.3 + "where degrades" in 1.9) get a single unit that covers both.
    for part, secs in _group_by_part(doc.sections).items():
        if len(secs) < 2:
            continue  # single-section Part == its section unit; nothing to merge
        content = "\n\n".join(s.content for s in secs)[:MAX_TOPIC_CHARS]
        for u in fn(part, content, "consolidated"):
            n_units += 1
            n_topics += 1
            _emit(lines, src.name, f"Topic — {part}", u, f"{part} (topic)")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    if n_failed:
        logger.warning("%d substantive section(s) produced no units", n_failed)
    stats: dict[str, Any] = {
        "backend": backend,
        "mode": mode,
        "sections_used": n_sections,
        "topic_units": n_topics,
        "units": n_units,
        "sections_empty": n_failed,
        "out": str(out),
    }
    if backend == "local":
        # Surface how much of the corpus the model actually produced vs how much
        # fell back to the heuristic floor — without this the corpus silently
        # looks model-made when most of it isn't (see lib/corpus/local.py).
        from lib.corpus.local import get_local_distiller

        counts = get_local_distiller().stats
        attempted = counts["model"] + counts["rejected"] + counts["empty"]
        stats["local"] = {
            **counts,
            "reject_pct": round(100 * counts["rejected"] / attempted) if attempted else 0,
        }
        if counts["rejected"]:
            logger.warning(
                "local backend: %d/%d section(s) rejected as task-narration → heuristic floor",
                counts["rejected"],
                attempted,
            )
    return stats

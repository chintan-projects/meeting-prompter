"""Corpus distiller — reshape an explainer doc into borrowable answer-units.

The judge showed the source docs are *explainers* (prose about how things work),
not answer banks — so retrieval returns context, not borrowable answers. This
distiller walks a doc section by section and emits self-contained, grounded
answer-units, each carrying a provenance pointer back to its source section. The
distilled corpus is indexed like any other; retrieval and the models are unchanged.

Two backends:
  - heuristic: clean + topic-title each section (no LLM; grounded verbatim; runnable
    with no credential — proves the pipeline).
  - cloud: Opus 4.8 extracts crisp, self-contained statements from each section,
    grounded ONLY in the section text (best quality; needs ANTHROPIC_API_KEY).

Usage:
    python -m scripts.lab.distiller <source.md> [--backend heuristic|cloud] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from lib.answer_extractor import extract_sentences
from lib.config import load_config
from lib.paths import get_docs_dir
from lib.rag.parser.composite_parser import CompositeParser
from scripts.lab.pipeline import clean_markdown

logger = logging.getLogger("lab.distiller")

MIN_SECTION_WORDS = 12  # below this a section is navigation/heading — no borrowable claim
MAX_UNIT_WORDS = 60  # keep units glanceable
MAX_SECTION_CHARS = 6000  # cap section text sent to the cloud extractor (token budget)

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
    import anthropic

    from scripts.lab import judge as _judge

    # Feed the model the RAW section, tables/code intact — much of the answer content
    # (e.g. the "three levels" table) lives in markdown tables that clean_markdown
    # strips. Opus reads tables natively and reshapes them into prose. Skip only true
    # navigation/heading stubs (guard on raw length, not the stripped length, so a
    # table-heavy section isn't dropped).
    if len(text.split()) < MIN_SECTION_WORDS:
        return []
    consolidated = mode == "consolidated"
    system = _CLOUD_SYSTEM_CONSOLIDATED if consolidated else _CLOUD_SYSTEM
    schema = _CLOUD_SCHEMA_CONSOLIDATED if consolidated else _CLOUD_SCHEMA
    user = f"SECTION: {heading}\n\n{text[:MAX_SECTION_CHARS]}"
    try:
        resp = _judge._get_client().messages.create(
            model=_judge.JUDGE_MODEL,
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


def distill(
    src: Path, out: Path, backend: str = "heuristic", mode: str = "atomic"
) -> dict[str, Any]:
    if backend == "cloud":
        from scripts.lab import judge as _judge

        hint = _judge.credential_hint()
        if hint is not None:
            # Fail fast rather than making 80 doomed API calls that each return [].
            raise RuntimeError(f"cloud backend needs a credential: {hint}")
    doc = CompositeParser().parse(src)
    fn = _distill_cloud if backend == "cloud" else _distill_heuristic
    lines: list[str] = [f"# Distilled: {src.name}", ""]
    n_units = 0
    n_sections = 0
    n_failed = 0
    for sec in doc.sections:
        heading = sec.heading or "(root)"
        words = len(clean_markdown(sec.content).split())
        units = fn(heading, sec.content, mode)
        if not units:
            if words >= MIN_SECTION_WORDS:
                n_failed += 1  # substantive section produced nothing (likely an extract failure)
            continue
        n_sections += 1
        prov = sec.heading_path or heading
        for u in units:
            n_units += 1
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(u)
            lines.append("")
            lines.append(f"_Source: {src.name} › {prov}_")
            lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    if n_failed:
        logger.warning("%d substantive section(s) produced no units", n_failed)
    return {
        "backend": backend,
        "mode": mode,
        "sections_used": n_sections,
        "units": n_units,
        "sections_empty": n_failed,
        "out": str(out),
    }


def _default_src() -> Path:
    """The single .md in the configured corpus dir, so `distiller` needs no arg."""
    docs = get_docs_dir(load_config().paths.docs_dir)
    mds = sorted(docs.glob("*.md")) if docs.is_dir() else []
    if len(mds) != 1:
        raise SystemExit(
            f"pass a source .md — the corpus dir {docs} has {len(mds)} markdown files, "
            "so the default is ambiguous."
        )
    return mds[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Distill an explainer doc into borrowable answer-units."
    )
    ap.add_argument("src", nargs="?", help="source .md (default: the configured corpus doc)")
    ap.add_argument("--backend", choices=["heuristic", "cloud"], default="heuristic")
    ap.add_argument(
        "--mode",
        choices=["atomic", "consolidated"],
        default="consolidated",
        help="atomic = 1-5 short facts/section; consolidated = one complete answer/section",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    src = Path(args.src) if args.src else _default_src()
    out = Path(args.out) if args.out else Path("data/distilled") / f"{src.stem}.distilled.md"
    logger.info("distilling %s (backend=%s, mode=%s)", src.name, args.backend, args.mode)
    stats = distill(src, out, args.backend, args.mode)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

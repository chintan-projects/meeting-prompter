"""Compatibility shim for llama-cpp-python embedded chat-template parsing.

`Llama.__init__` eagerly compiles *every* chat template stored in a model's GGUF
metadata using the Python `jinja2` package (llama.py, `Jinja2ChatFormatter`).
Newer LFM2.5 templates use Hugging Face jinja extensions (e.g. the
`{% generation %}` tag) that plain `jinja2` cannot parse — so a single template
raises `TemplateSyntaxError` and the whole model load crashes. The C++ llama.cpp
runtime uses `minja`, which *does* understand these tags, which is why the same
GGUF loads fine via the CLI.

We build ChatML prompts by hand and use raw completion, so the embedded template
is never rendered. This shim makes its compilation non-fatal: a template that
won't parse is replaced with a trivial one instead of aborting model load.

Call `install()` once before constructing any `Llama`.
"""

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_INSTALLED = False


def install() -> None:
    """Idempotently make embedded chat-template compilation non-fatal."""
    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return

    try:
        import jinja2
        from llama_cpp import llama_chat_format
    except ImportError:  # llama_cpp not present in this environment
        return

    formatter = llama_chat_format.Jinja2ChatFormatter
    # Third-party, dynamically wrapped — Any is intentional for the monkeypatch.
    original_init: Callable[..., None] = formatter.__init__

    def safe_init(self: Any, *args: Any, **kwargs: Any) -> None:
        try:
            original_init(self, *args, **kwargs)
        except jinja2.exceptions.TemplateError:
            # The embedded template is never rendered in our raw-completion path;
            # substitute a trivial one so model load succeeds.
            kwargs["template"] = "{{ '' }}"
            original_init(self, *args, **kwargs)
            logger.warning(
                "Embedded chat template failed to compile (jinja2 lacks an HF "
                "extension); replaced with a no-op — raw completion is unaffected."
            )

    formatter.__init__ = safe_init  # type: ignore[method-assign]
    _INSTALLED = True

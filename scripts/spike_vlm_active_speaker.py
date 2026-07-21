"""Stage-0 spike: can LFM2.5-VL read the active speaker off a meeting screenshot? (F-500)

The pivotal cross-platform attribution experiment. If the VLM reliably names the
highlighted active-speaker tile from a single still frame, the passive,
no-bot, app-agnostic attribution path is real. If not, we lean on the platform
SDK / acoustic fallback.

Usage:
    python scripts/spike_vlm_active_speaker.py <screenshot.png>
    python scripts/spike_vlm_active_speaker.py shot.png --roster "Chintan,Ramin,Priya"

Provide a real meeting screenshot (Zoom/Teams/Meet, gallery or speaker view) with
someone visibly highlighted as talking. Roster is optional but recommended — it
turns fragile open-vocab name OCR into a closed-set choice (the robust framing).

Uses LFM2.5-VL-450M (the light path) via transformers. No llama.cpp needed.
Requires: torch, transformers, pillow.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

MODEL_DIRNAME = "LFM2.5-VL-450M"


def resolve_model_path() -> Path:
    models_dir = os.environ.get("MODELS_DIR")
    candidates: list[Path] = []
    if models_dir:
        candidates.append(Path(models_dir).expanduser() / MODEL_DIRNAME)
    candidates.append(Path.home() / "Projects" / "_models" / MODEL_DIRNAME)
    for path in candidates:
        if path.exists():
            return path
    tried = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(f"Could not find {MODEL_DIRNAME}. Tried:\n  {tried}")


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_active_speaker_prompt(roster: list[str]) -> str:
    base = (
        "This is a screenshot of a video meeting. One participant tile is usually "
        "visually highlighted as the current active speaker — a colored border, a "
        "glowing outline, or a speaking/audio indicator. "
    )
    if roster:
        names = ", ".join(roster)
        base += (
            f"The participants are: {names}. Reply with ONLY the name of the "
            "highlighted active speaker from that list, or NONE if no tile is "
            "highlighted. Output just the name, nothing else."
        )
    else:
        base += (
            "Reply with ONLY the name shown on the highlighted active-speaker "
            "tile, or NONE if no tile is highlighted. Output just the name."
        )
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a meeting screenshot")
    parser.add_argument("--roster", default="", help="Comma-separated participant names")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        raise SystemExit(f"Screenshot not found: {image_path}")
    roster = [n.strip() for n in args.roster.split(",") if n.strip()]

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Install: pip install torch transformers pillow"
        )

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    model_path = resolve_model_path()
    device = pick_device()
    print(f"model:  {model_path}")
    print(f"device: {device}")
    print(f"image:  {image_path}")
    print(f"roster: {roster or '(none given)'}\n")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval().to(device)
    print(f"loaded in {time.perf_counter() - t0:.1f}s\n")

    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not open image {image_path}: {exc}")

    # Two probes: (1) active speaker (the real test), (2) roster extraction (L4 bonus).
    probes = {
        "active_speaker": build_active_speaker_prompt(roster),
        "visible_names": (
            "List the participant names visible in this meeting screenshot, "
            "comma-separated. If none are legible, reply NONE."
        ),
    }

    @torch.no_grad()
    def ask(prompt: str, max_new_tokens: int = 40) -> tuple[str, float]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        t = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        dt = time.perf_counter() - t
        gen = out[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return text, dt

    print("--- probes ---")
    for label, prompt in probes.items():
        answer, dt = ask(prompt)
        print(f"[{label}]  ({dt * 1000:.0f} ms)\n  {answer}\n")

    print("--- verdict ---")
    print("Eyeball it: did [active_speaker] name the person actually highlighted?")
    print("Run on 5-10 varied frames (gallery/speaker view, different apps) before")
    print("trusting it. Consistent hits → cross-platform passive path is viable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

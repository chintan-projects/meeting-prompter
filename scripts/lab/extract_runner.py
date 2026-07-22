"""Standalone 350M-Extract runner (invoked as a subprocess by the lab).

The Extract build's GGUF is incompatible with the project's pinned llama.cpp, and
its safetensors need transformers>=5 (the project pins 4.56.2). So the lab shells
out to *this* script under an interpreter that has a newer runtime — point the lab
at it via LAB_EXTRACT_PYTHON (e.g. an overlay venv).

Protocol: reads one JSON object on stdin
    {"model_dir": str, "question": str, "context": str, "max_tokens": int}
and prints one JSON object on stdout
    {"latency_ms": int, "text": str}   or   {"error": str}

Kept dependency-light and self-contained so it runs in a minimal venv.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> None:
    req = json.loads(sys.stdin.read())
    model_dir = Path(req["model_dir"])
    question = req["question"]
    context = req["context"]
    max_tokens = int(req.get("max_tokens", 160))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception:
        # Older transformers can't resolve this model's custom tokenizer class;
        # load the fast tokenizer directly from tokenizer.json + its template.
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(
            tokenizer_file=str(model_dir / "tokenizer.json"),
            bos_token="<|startoftext|>",
            eos_token="<|im_end|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>",
        )
        template = model_dir / "chat_template.jinja"
        if template.exists():
            tok.chat_template = template.read_text(encoding="utf-8")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.float32
    )
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev).eval()

    msgs = [
        {
            "role": "system",
            "content": "Answer the question using ONLY the context. Be concise. "
            "If the context lacks the answer, say so.",
        },
        {"role": "user", "content": f"CONTEXT:\n{context[:6000]}\n\nQUESTION: {question}"},
    ]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    enc = {k: v.to(dev) for k, v in enc.items() if k != "token_type_ids"}
    n_in = enc["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False)
    dt = round((time.time() - t0) * 1000)
    text = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
    print(json.dumps({"latency_ms": dt, "text": text}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — report the failure as JSON for the lab
        print(json.dumps({"error": repr(e)}))
        sys.exit(1)

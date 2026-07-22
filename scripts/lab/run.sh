#!/usr/bin/env bash
# Launch the Corpus & Retrieval Lab (offline).
#
# Retrieval + coverage work with no credential. The LLM-as-judge (cloud Opus 4.8)
# needs an Anthropic credential in THIS shell's environment — either:
#   export ANTHROPIC_API_KEY=sk-ant-...      (from console.anthropic.com)
#   # or:  ant auth login                    (OAuth; reuses your Anthropic login)
# then run this script. Your key is read by the SDK from the env; it is never
# written to disk or passed as an argument.
set -uo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source venv/bin/activate 2>/dev/null || true

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  echo "✓ ANTHROPIC_API_KEY present — LLM-judge enabled."
elif command -v ant >/dev/null 2>&1 && ant auth status >/dev/null 2>&1; then
  echo "✓ ant auth profile present — LLM-judge enabled."
else
  echo "⚠  No Anthropic credential in this shell — the LLM-judge will be disabled."
  echo "   Enable it:  export ANTHROPIC_API_KEY=sk-ant-...   then re-run this script."
  echo "   (Retrieval + borrowable answers + coverage still work without it.)"
fi

pkill -f "uvicorn scripts.lab.server" 2>/dev/null || true
sleep 1
echo "→ http://localhost:8555"
exec uvicorn scripts.lab.server:app --port 8555

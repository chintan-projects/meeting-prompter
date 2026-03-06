#!/usr/bin/env python3
"""Real-Time Meeting Intelligence Agent — CLI entry point.

Powered by LFM2.5 models running 100% locally on Apple Silicon.

Usage:
  python coach.py                              # Live meeting (BlackHole)
  python coach.py --mic                        # Test with microphone
  python coach.py --test audio.wav             # Test with audio file
  python coach.py --context meeting_context.yaml  # Load meeting context
  python coach.py --list-devices               # List audio devices
  python coach.py --create-context             # Create template context file
"""
import os

# Suppress tokenizer parallelism warning (must be before HuggingFace imports)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.audio_capture import list_audio_devices
from lib.config import load_config
from lib.conversation.meeting_context import create_meeting_template
from lib.dashboard import display_header, display_status
from lib.orchestrator import MeetingOrchestrator

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


def _test_transcription(audio_file: Path) -> None:
    """Test mode: transcribe a single audio file with full pipeline."""
    config = load_config()

    orchestrator = MeetingOrchestrator(config=config, audio_device="none")

    audio_file = audio_file.resolve()
    display_status(f"Transcribing: {audio_file}")

    text = orchestrator.lfm2.transcribe(audio_file)
    print(f"\nTranscription:\n{text}\n")

    if text and not text.startswith("["):
        rag_context, confidence, source = orchestrator.rag.query(text)
        print(f"RAG Confidence: {confidence:.0%}")
        print(f"Source: {source}")

        # Process through trigger engine
        triggers = orchestrator.buffer.add_chunk(text, 0.0)
        for trigger in triggers:
            result = orchestrator._process_trigger(trigger)
            if result:
                print(f"\n{trigger.type.emoji} {trigger.type.label}:")
                print(f"  {result.answer}")
                print(f"  ({result.method}, {result.confidence:.0%}, {result.latency_ms:.0f}ms)")
    else:
        print("No speech detected in audio file.")


def main() -> None:
    """Parse CLI args and launch the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="Real-Time Meeting Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python coach.py                              # Live meeting (BlackHole)
  python coach.py --mic                        # Microphone test mode
  python coach.py --test audio.wav             # Test with audio file
  python coach.py --context meeting_context.yaml
  python coach.py --create-context             # Create template YAML
  python coach.py --list-devices
""",
    )
    parser.add_argument(
        "--device", "-d",
        default="BlackHole 2ch",
        help="Audio input device name (default: BlackHole 2ch)",
    )
    parser.add_argument(
        "--mic", "-m",
        action="store_true",
        help="Use MacBook microphone instead of BlackHole",
    )
    parser.add_argument(
        "--test", "-t",
        type=Path,
        help="Test mode: transcribe a single audio file",
    )
    parser.add_argument(
        "--context", "-c",
        type=Path,
        help="Path to meeting_context.yaml for pre-meeting config",
    )
    parser.add_argument(
        "--create-context",
        action="store_true",
        help="Create a template meeting_context.yaml and exit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_devices:
        list_audio_devices()
        return

    if args.create_context:
        path = Path("meeting_context.yaml")
        create_meeting_template(path)
        print(f"Created template at {path}")
        return

    if args.test:
        if not args.test.exists():
            print(f"Error: Audio file not found: {args.test}")
            sys.exit(1)
        _test_transcription(args.test)
        return

    # Determine audio device
    if args.mic:
        audio_device = "MacBook Pro Microphone"
        print("\nMIC MODE: Speak into your microphone to test")
        print("  (Use without --mic for live meetings with BlackHole)\n")
    else:
        audio_device = args.device
        print(f"\nMEETING MODE: Capturing from {audio_device}")
        print("  (Use --mic to test with your microphone)\n")

    # Load config and start
    config = load_config(args.config)
    orchestrator = MeetingOrchestrator(
        config=config,
        audio_device=audio_device,
        meeting_context_path=args.context,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()

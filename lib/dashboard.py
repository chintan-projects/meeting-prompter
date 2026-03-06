"""Dashboard — terminal display for real-time meeting intelligence.

Supports multi-mode trigger display with priority coloring,
result history (last 5), and live transcript preview.
"""
import sys
import time
from collections import deque
from typing import Deque, List, Optional

import psutil

from lib.generation.types import GenerationResult
from lib.triggers.types import TriggerType


class Colors:
    """ANSI color codes for terminal."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


# Color per trigger type
TRIGGER_COLORS = {
    TriggerType.ALERT: Colors.RED,
    TriggerType.QUESTION: Colors.BLUE,
    TriggerType.TOPIC_MATCH: Colors.GRAY,
    TriggerType.FOLLOW_UP: Colors.MAGENTA,
}


class Dashboard:
    """Stateful CLI dashboard with result history and live transcript.

    Tracks the last N results and renders them in priority order.
    Call render() after each new result or periodically for live updates.
    """

    def __init__(self, max_history: int = 5) -> None:
        self._results: Deque[GenerationResult] = deque(maxlen=max_history)
        self._transcript_preview: str = ""
        self._meeting_title: str = ""
        self._start_time: float = time.time()
        self._seen_texts: Deque[str] = deque(maxlen=20)

    def set_meeting_title(self, title: str) -> None:
        """Set meeting title for header display."""
        self._meeting_title = title

    def add_result(self, result: GenerationResult) -> None:
        """Add a generation result and render. Deduplicates by answer text."""
        key = result.answer[:80]
        if key in self._seen_texts:
            return
        self._seen_texts.append(key)
        self._results.appendleft(result)

    def set_transcript_preview(self, text: str) -> None:
        """Update live transcript preview line."""
        self._transcript_preview = text

    def render(self) -> None:
        """Render the full dashboard to terminal."""
        lines: List[str] = []

        # Header
        elapsed = time.time() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        title = self._meeting_title or "Meeting Intelligence"
        lines.append("")
        lines.append(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        lines.append(
            f"{Colors.BOLD}  {title}{Colors.RESET}"
            f"  {Colors.DIM}{mins:02d}:{secs:02d}{Colors.RESET}"
        )
        lines.append(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

        # System stats
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        lines.append(f"{Colors.DIM}  CPU: {cpu:.0f}%  RAM: {ram:.0f}%{Colors.RESET}")
        lines.append("")

        # Results (sorted by priority — alerts first)
        if self._results:
            sorted_results = sorted(self._results, key=lambda r: r.trigger_type.priority)
            for result in sorted_results:
                lines.append(_format_result(result))
            lines.append("")
        else:
            lines.append(f"  {Colors.DIM}Listening...{Colors.RESET}")
            lines.append("")

        # Transcript preview
        if self._transcript_preview:
            preview = self._transcript_preview[-80:]
            lines.append(f"  {Colors.DIM}> {preview}{Colors.RESET}")
            lines.append("")

        lines.append(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")

        # Clear screen and print
        sys.stdout.write("\033[2J\033[H")  # clear screen, cursor to top
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()


def _format_result(result: GenerationResult) -> str:
    """Format a single result line with trigger-type styling."""
    color = TRIGGER_COLORS.get(result.trigger_type, Colors.WHITE)
    emoji = result.trigger_type.emoji
    label = result.trigger_type.label

    # Truncate answer for single-line display
    answer = result.answer.replace("\n", " ")
    if len(answer) > 70:
        answer = answer[:67] + "..."

    conf_pct = result.confidence * 100
    ms = result.latency_ms

    style = Colors.ITALIC if result.trigger_type == TriggerType.FOLLOW_UP else ""
    return (
        f"  {color}{emoji} {label:10}{Colors.RESET} "
        f"{style}{answer}{Colors.RESET}"
        f"  {Colors.DIM}({conf_pct:.0f}% {ms:.0f}ms){Colors.RESET}"
    )


# --- Legacy helpers (used by coach.py until Phase 7 wiring) ---


def clear_line() -> None:
    """Clear current terminal line."""
    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.flush()


def display_header() -> None:
    """Display dashboard header."""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Real-Time Meeting Intelligence Agent{Colors.RESET}")
    print(f"{Colors.DIM}  Powered by LFM2.5 | 100% Local Processing{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")


def display_status(message: str) -> None:
    """Display a status message."""
    print(f"{Colors.DIM}[STATUS]{Colors.RESET} {message}")


def display_update(
    transcript: str,
    vibe: str,
    vibe_emoji: str,
    confidence: float,
    context_preview: Optional[str] = None,
) -> None:
    """Display real-time update on single line (overwriting previous)."""
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    conf_pct = confidence * 100
    if conf_pct >= 50:
        conf_color = Colors.GREEN
    elif conf_pct >= 25:
        conf_color = Colors.YELLOW
    else:
        conf_color = Colors.RED

    max_transcript = 40
    if len(transcript) > max_transcript:
        transcript_display = transcript[:max_transcript] + "..."
    else:
        transcript_display = transcript

    status = (
        f"\r{Colors.DIM}[CPU:{cpu:4.0f}% RAM:{ram:4.0f}%]{Colors.RESET} "
        f"{conf_color}Conf:{conf_pct:3.0f}%{Colors.RESET} "
        f"{Colors.CYAN}|{Colors.RESET} {transcript_display}"
    )

    sys.stdout.write(status + " " * 10)
    sys.stdout.flush()


def display_transcript_line(
    timestamp: str, transcript: str, vibe: str, confidence: float,
) -> None:
    """Display a full transcript line (for logging/review mode)."""
    conf_pct = confidence * 100
    print(
        f"{Colors.DIM}[{timestamp}]{Colors.RESET} {transcript} "
        f"{Colors.DIM}| {vibe} | {conf_pct:.0f}%{Colors.RESET}"
    )


def display_summary(
    total_chunks: int, dominant_vibes: dict, avg_confidence: float,
) -> None:
    """Display meeting summary."""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Meeting Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"  Total chunks processed: {total_chunks}")
    print(f"  Average RAG confidence: {avg_confidence * 100:.1f}%")
    print(f"  Vibe breakdown:")
    for vibe, count in sorted(dominant_vibes.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = (count / total_chunks) * 100
            print(f"    {vibe}: {count} ({pct:.1f}%)")
    print()

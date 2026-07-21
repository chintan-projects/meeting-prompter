import { useCallback, useState } from "react";

export interface TranscriptSegment {
  id: string;
  text: string;
  timestamp: number;
  end_timestamp: number;
  speaker: string;
  source: string; // "mic" or "system" — audio stream origin
  is_final: boolean;
  low_confidence?: boolean; // flagged best-effort speaker label (F-606)
  edited?: boolean;
}

interface UseTranscriptReturn {
  segments: TranscriptSegment[];
  /** Create or update a segment by ID (turn-based upsert). */
  upsertSegment: (seg: TranscriptSegment) => void;
  /** Apply a user edit to a segment's text. */
  editSegment: (id: string, text: string) => void;
  /** Clear all segments (e.g., on new session). */
  clear: () => void;
}

/**
 * Manages transcript state with turn-based upsert semantics.
 *
 * The backend streams turns via WebSocket:
 * - transcript_update: partial turn (still accumulating speech)
 * - transcript_final: completed turn (pause detected)
 *
 * Both use upsertSegment — if the turn ID exists, update its text;
 * if not, create a new entry. This gives smooth live updates as
 * speech accumulates, then clean finalization on pause.
 */
export function useTranscript(): UseTranscriptReturn {
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);

  const upsertSegment = useCallback((seg: TranscriptSegment) => {
    setSegments((prev) => {
      const idx = prev.findIndex((s) => s.id === seg.id);
      if (idx >= 0) {
        // Update existing turn — preserve user edits
        const existing = prev[idx];
        if (existing.edited) return prev;
        const updated = [...prev];
        updated[idx] = { ...seg, edited: existing.edited };
        return updated;
      }
      // New turn
      return [...prev, seg];
    });
  }, []);

  const editSegment = useCallback((id: string, text: string) => {
    setSegments((prev) =>
      prev.map((s) => (s.id === id ? { ...s, text, edited: true } : s))
    );
  }, []);

  const clear = useCallback(() => setSegments([]), []);

  return { segments, upsertSegment, editSegment, clear };
}

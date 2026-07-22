import { useCallback, useEffect, useRef, useState } from "react";
import type { TranscriptSegment } from "../hooks/useTranscript";

interface TranscriptPaneProps {
  segments: TranscriptSegment[];
  collapsed: boolean;
  onToggle: () => void;
  onEdit: (id: string, text: string) => void;
  onRenameSpeaker: (oldName: string, newName: string) => void;
  width: number;
  /** Select-to-answer (D-02, spatial): answer the highlighted span on demand. */
  onAnswerSelection?: (text: string) => void;
}

// Speaker color palette for system audio speakers (diarized or renamed)
const SPEAKER_COLORS: readonly string[] = [
  "#4caf50", // green
  "#ff9800", // orange
  "#ab47bc", // purple
  "#e91e63", // pink
  "#00bcd4", // teal
  "#ff5722", // deep orange
];

/**
 * Dual-stream transcript display with chat-bubble layout.
 *
 * Mic turns (source="mic") appear right-aligned with accent styling → "You"
 * System turns (source="system") appear left-aligned with muted styling → speaker label
 * Tier 2 diarization colors individual remote speakers (Speaker A, B, ...).
 * Active turns show a live indicator. Double-click finalized turns to edit.
 */
export function TranscriptPane({
  segments,
  collapsed,
  onToggle,
  onEdit,
  onRenameSpeaker,
  width,
  onAnswerSelection,
}: TranscriptPaneProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [pinBottom, setPinBottom] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [renamingSpeaker, setRenamingSpeaker] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");

  // Stable color assignment: maps speaker names to color indices.
  // Survives renames by transferring the index from old → new name.
  const colorMapRef = useRef<Map<string, number>>(new Map());

  const getColorIndex = useCallback((speaker: string): number => {
    const map = colorMapRef.current;
    if (!map.has(speaker)) {
      map.set(speaker, map.size);
    }
    return map.get(speaker)!;
  }, []);

  const getSpeakerColor = useCallback(
    (seg: TranscriptSegment): string => {
      if (seg.source === "mic") return "var(--accent-blue, #4a9eff)";
      const speaker = seg.speaker;
      if (!speaker || speaker === "Others") return "var(--text-secondary)";
      const idx = getColorIndex(speaker);
      return SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
    },
    [getColorIndex],
  );

  const getSpeakerBubbleBg = useCallback(
    (seg: TranscriptSegment): string | undefined => {
      if (seg.source !== "system") return undefined;
      const speaker = seg.speaker;
      if (!speaker || speaker === "Others") return undefined;
      const idx = getColorIndex(speaker);
      const hex = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
      return `${hex}14`; // ~8% opacity via hex alpha
    },
    [getColorIndex],
  );

  const startRename = (speaker: string) => {
    setRenamingSpeaker(speaker);
    setRenameText(speaker);
  };

  const commitRename = () => {
    if (renamingSpeaker && renameText.trim() && renameText.trim() !== renamingSpeaker) {
      const newName = renameText.trim();
      // Transfer color index from old name to new name
      const map = colorMapRef.current;
      const oldIdx = map.get(renamingSpeaker);
      if (oldIdx !== undefined) {
        map.set(newName, oldIdx);
        map.delete(renamingSpeaker);
      }
      onRenameSpeaker(renamingSpeaker, newName);
    }
    setRenamingSpeaker(null);
  };

  // Auto-scroll when new turns arrive or active turn updates
  useEffect(() => {
    if (pinBottom) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [segments, pinBottom]);

  const startEdit = (seg: TranscriptSegment) => {
    // Only allow editing finalized turns
    if (!seg.is_final) return;
    setEditingId(seg.id);
    setEditText(seg.text);
  };

  const commitEdit = () => {
    if (editingId && editText.trim()) {
      onEdit(editingId, editText.trim());
    }
    setEditingId(null);
  };

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const isMic = (seg: TranscriptSegment): boolean => seg.source === "mic";

  // --- select-to-answer (D-02, spatial) ------------------------------------
  // Highlight any span and a single button appears at the selection. Deliberately
  // not automatic on selection: selecting text to re-read it is common, and an
  // answer that fires on every highlight is the same interruption problem in a
  // new place.
  const [selection, setSelection] = useState<{ text: string; x: number; y: number } | null>(
    null
  );

  const handleSelection = useCallback(() => {
    if (!onAnswerSelection) return;
    const sel = window.getSelection();
    const text = sel?.toString().trim() ?? "";
    // Two words is the floor — a single word rarely carries a question, and the
    // button flickering on stray clicks is worse than not offering it.
    if (!sel || sel.rangeCount === 0 || text.split(/\s+/).length < 2) {
      setSelection(null);
      return;
    }
    const rect = sel.getRangeAt(0).getBoundingClientRect();
    setSelection({ text, x: rect.left + rect.width / 2, y: rect.top });
  }, [onAnswerSelection]);

  const askSelection = useCallback(() => {
    if (selection && onAnswerSelection) onAnswerSelection(selection.text);
    setSelection(null);
    window.getSelection()?.removeAllRanges();
  }, [selection, onAnswerSelection]);

  if (collapsed) {
    return (
      <div style={styles.collapsed} onClick={onToggle}>
        <span style={styles.collapseHandle}>&#x25B6;</span>
      </div>
    );
  }

  return (
    <div style={{ ...styles.pane, width }}>
      <div style={styles.header}>
        <span style={styles.collapseHandle} onClick={onToggle}>
          &#x25C0;
        </span>
        <span style={styles.headerTitle}>TRANSCRIPT</span>
        <label style={styles.pinLabel}>
          <input
            type="checkbox"
            checked={pinBottom}
            onChange={(e) => setPinBottom(e.target.checked)}
          />
          Auto-scroll
        </label>
      </div>

      {selection && (
        <button
          style={{ ...styles.askBtn, left: selection.x, top: selection.y - 38 }}
          onMouseDown={(e) => e.preventDefault()} // keep the selection alive
          onClick={askSelection}
        >
          💡 Answer this
        </button>
      )}

      <div style={styles.body} onMouseUp={handleSelection}>
        {segments.length === 0 && (
          <div style={styles.empty}>Waiting for transcript...</div>
        )}

        {segments.map((seg) => {
          const mic = isMic(seg);
          return (
            <div
              key={seg.id}
              style={{
                ...styles.turnRow,
                justifyContent: mic ? "flex-end" : "flex-start",
              }}
            >
              <div
                style={{
                  ...styles.turn,
                  ...(mic ? styles.micTurn : styles.systemTurn),
                  ...(getSpeakerBubbleBg(seg)
                    ? { background: getSpeakerBubbleBg(seg) }
                    : {}),
                  ...(seg.is_final ? {} : styles.activeTurn),
                  ...(seg.edited ? styles.editedTurn : {}),
                }}
                onDoubleClick={() => startEdit(seg)}
              >
                <div style={styles.turnHeader}>
                  <span style={styles.timestamp}>{formatTime(seg.timestamp)}</span>
                  {renamingSpeaker === (seg.speaker || (mic ? "You" : "Others")) &&
                  !mic &&
                  seg.is_final ? (
                    <input
                      style={{
                        ...styles.renameInput,
                        color: getSpeakerColor(seg),
                      }}
                      value={renameText}
                      onChange={(e) => setRenameText(e.target.value)}
                      onBlur={commitRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          commitRename();
                        } else if (e.key === "Escape") {
                          setRenamingSpeaker(null);
                        }
                      }}
                      autoFocus
                    />
                  ) : (
                    <span
                      style={{
                        ...styles.speakerLabel,
                        color: getSpeakerColor(seg),
                        ...(!mic && seg.is_final ? styles.speakerClickable : {}),
                      }}
                      onClick={
                        !mic && seg.is_final
                          ? () => startRename(seg.speaker || "Others")
                          : undefined
                      }
                      title={!mic && seg.is_final ? "Click to rename" : undefined}
                    >
                      {seg.speaker || (mic ? "You" : "Others")}
                    </span>
                  )}
                  {seg.low_confidence && (
                    <span
                      style={styles.lowConfidenceBadge}
                      title="Best-effort speaker label — conference-room / low-confidence attribution"
                    >
                      ~ best guess
                    </span>
                  )}
                  {!seg.is_final && <span style={styles.liveIndicator} />}
                </div>

                {editingId === seg.id ? (
                  <textarea
                    style={styles.editInput}
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    onBlur={commitEdit}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        commitEdit();
                      }
                    }}
                    autoFocus
                    rows={3}
                  />
                ) : (
                  <div style={styles.turnText}>{seg.text}</div>
                )}
              </div>
            </div>
          );
        })}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  pane: {
    display: "flex",
    flexDirection: "column",
    borderRight: "1px solid var(--border)",
    width: 380,
    minWidth: 200,
    background: "var(--bg-primary)",
  },
  collapsed: {
    width: 24,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderRight: "1px solid var(--border)",
    background: "var(--bg-secondary)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 12px",
    borderBottom: "1px solid var(--border)",
    background: "var(--bg-secondary)",
  },
  headerTitle: {
    flex: 1,
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 1,
    color: "var(--text-secondary)",
  },
  pinLabel: { fontSize: 11, color: "var(--text-muted)", cursor: "pointer" },
  collapseHandle: {
    cursor: "pointer",
    color: "var(--text-muted)",
    fontSize: 12,
    userSelect: "none" as const,
  },
  body: {
    flex: 1,
    overflowY: "auto" as const,
    padding: "8px 0",
  },
  empty: { color: "var(--text-muted)", fontStyle: "italic", padding: 16 },

  /* Select-to-answer affordance — fixed so it tracks the viewport selection rect */
  askBtn: {
    position: "fixed",
    transform: "translateX(-50%)",
    zIndex: 50,
    padding: "6px 12px",
    fontSize: 12,
    fontWeight: 600,
    borderRadius: 6,
    border: "none",
    cursor: "pointer",
    background: "var(--accent-blue, #4c8bf5)",
    color: "#fff",
    boxShadow: "0 2px 8px rgba(0,0,0,0.35)",
    whiteSpace: "nowrap",
  },

  /* --- Turn blocks (chat-bubble layout) --- */
  turnRow: {
    display: "flex",
    padding: "4px 10px",
  },
  turn: {
    maxWidth: "85%",
    padding: "8px 14px",
    borderRadius: 12,
    cursor: "default",
    transition: "background 0.15s ease",
  },
  micTurn: {
    background: "rgba(74, 158, 255, 0.12)",
    borderBottomRightRadius: 4,
  },
  systemTurn: {
    background: "var(--bg-secondary)",
    borderBottomLeftRadius: 4,
  },
  activeTurn: {
    borderLeft: "2px solid var(--accent-blue, #4a9eff)",
  },
  editedTurn: {
    opacity: 0.85,
  },
  turnHeader: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 4,
  },
  timestamp: {
    fontFamily: "var(--font-mono)",
    fontSize: 11,
    color: "var(--text-muted)",
    flexShrink: 0,
  },
  speakerLabel: {
    fontSize: 11,
    fontWeight: 600,
  },
  speakerClickable: {
    cursor: "pointer",
    borderBottom: "1px dashed currentColor",
  },
  lowConfidenceBadge: {
    fontSize: 9,
    fontWeight: 600,
    color: "var(--text-secondary)",
    opacity: 0.7,
    fontStyle: "italic",
    padding: "0 4px",
    border: "1px dashed var(--text-secondary)",
    borderRadius: 3,
  },
  renameInput: {
    fontSize: 11,
    fontWeight: 600,
    background: "transparent",
    border: "none",
    borderBottom: "1px solid currentColor",
    outline: "none",
    padding: 0,
    fontFamily: "inherit",
    width: 100,
  },
  liveIndicator: {
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: "var(--accent-blue, #4a9eff)",
    animation: "pulse 1.5s ease-in-out infinite",
    flexShrink: 0,
  },
  turnText: {
    fontSize: 14,
    lineHeight: 1.6,
    color: "var(--text-primary)",
    whiteSpace: "pre-wrap" as const,
    wordBreak: "break-word" as const,
  },
  editInput: {
    width: "100%",
    background: "var(--bg-card, var(--bg-secondary))",
    border: "1px solid var(--accent-blue)",
    borderRadius: 4,
    color: "var(--text-primary)",
    padding: "6px 8px",
    fontSize: 14,
    lineHeight: "1.6",
    fontFamily: "inherit",
    outline: "none",
    resize: "vertical" as const,
  },
};

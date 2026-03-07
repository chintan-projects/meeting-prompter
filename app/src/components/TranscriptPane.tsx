import { useEffect, useRef, useState } from "react";
import type { TranscriptSegment } from "../hooks/useTranscript";

interface TranscriptPaneProps {
  segments: TranscriptSegment[];
  collapsed: boolean;
  onToggle: () => void;
  onEdit: (id: string, text: string) => void;
  width: number;
}

// Speaker color palette for Tier 2 diarized system audio speakers
const SPEAKER_COLORS: readonly string[] = [
  "#4caf50", // green — Speaker A
  "#ff9800", // orange — Speaker B
  "#ab47bc", // purple — Speaker C
  "#e91e63", // pink — Speaker D
  "#00bcd4", // teal — Speaker E
  "#ff5722", // deep orange — Speaker F
];

/** Map speaker label to a distinct color. Mic → blue, system speakers → cycling palette. */
function getSpeakerColor(seg: TranscriptSegment): string {
  if (seg.source === "mic") return "var(--accent-blue, #4a9eff)";
  const match = seg.speaker.match(/Speaker\s+([A-Z])/);
  if (match) {
    const idx = match[1].charCodeAt(0) - 65; // A=0, B=1, ...
    return SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
  }
  return "var(--text-secondary)";
}

/** Get a faint background tint for diarized system speakers. */
function getSpeakerBubbleBg(seg: TranscriptSegment): string | undefined {
  if (seg.source !== "system") return undefined;
  const match = seg.speaker.match(/Speaker\s+([A-Z])/);
  if (match) {
    const idx = match[1].charCodeAt(0) - 65;
    const hex = SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
    return `${hex}14`; // ~8% opacity via hex alpha
  }
  return undefined;
}

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
  width,
}: TranscriptPaneProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [pinBottom, setPinBottom] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");

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

      <div style={styles.body}>
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
                  <span
                    style={{
                      ...styles.speakerLabel,
                      color: getSpeakerColor(seg),
                    }}
                  >
                    {seg.speaker || (mic ? "You" : "Others")}
                  </span>
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

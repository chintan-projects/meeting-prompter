import { useEffect, useRef, useState } from "react";
import type { TranscriptSegment } from "../hooks/useTranscript";

interface TranscriptPaneProps {
  segments: TranscriptSegment[];
  collapsed: boolean;
  onToggle: () => void;
  onEdit: (id: string, text: string) => void;
}

/**
 * Turn-based transcript display.
 *
 * Each segment represents a speech turn (continuous block of speech
 * accumulated on the backend). Turns are displayed as timestamped
 * paragraphs. Active turns (still accumulating) show a live indicator.
 * Finalized turns are clean, static paragraphs. Double-click to edit.
 */
export function TranscriptPane({
  segments,
  collapsed,
  onToggle,
  onEdit,
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

  if (collapsed) {
    return (
      <div style={styles.collapsed} onClick={onToggle}>
        <span style={styles.collapseHandle}>&blacktriangleright;</span>
      </div>
    );
  }

  return (
    <div style={styles.pane}>
      <div style={styles.header}>
        <span style={styles.collapseHandle} onClick={onToggle}>
          &blacktriangleleft;
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

        {segments.map((seg) => (
          <div
            key={seg.id}
            style={{
              ...styles.turn,
              ...(seg.is_final ? {} : styles.activeTurn),
              ...(seg.edited ? styles.editedTurn : {}),
            }}
            onDoubleClick={() => startEdit(seg)}
          >
            <div style={styles.turnHeader}>
              <span style={styles.timestamp}>{formatTime(seg.timestamp)}</span>
              {seg.speaker && (
                <span style={styles.speaker}>{seg.speaker}</span>
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
        ))}

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

  /* --- Turn blocks --- */
  turn: {
    padding: "8px 14px",
    borderBottom: "1px solid var(--border)",
    cursor: "default",
    transition: "background 0.15s ease",
  },
  activeTurn: {
    background: "rgba(74, 158, 255, 0.04)",
    borderLeft: "2px solid var(--accent-blue, #4a9eff)",
  },
  editedTurn: {
    background: "rgba(74, 158, 255, 0.06)",
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
  speaker: {
    fontSize: 11,
    fontWeight: 600,
    color: "var(--text-secondary)",
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

import { useRef, useState } from "react";

interface PromptResult {
  id: number;
  trigger_type: string;
  trigger_text: string;
  answer: string;
  confidence: number;
  method: string;
  latency_ms: number;
  source: string;
  receivedAt: number;
}

interface PromptsPaneProps {
  results: PromptResult[];
}

const TRIGGER_STYLES: Record<string, { color: string; emoji: string; label: string }> = {
  alert: { color: "var(--accent-red)", emoji: "\uD83D\uDD34", label: "ALERT" },
  question: { color: "var(--accent-blue)", emoji: "\uD83D\uDCAC", label: "QUESTION" },
  topic: { color: "var(--accent-gray)", emoji: "\uD83D\uDCCB", label: "TOPIC" },
  follow_up: { color: "var(--accent-purple)", emoji: "\u2753", label: "FOLLOW-UP" },
};

const AUTO_DISMISS_MS = 60_000;

export function PromptsPane({ results }: PromptsPaneProps) {
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const topRef = useRef<HTMLDivElement>(null);

  // Filter out auto-dismissed results
  const now = Date.now();
  const visible = results.filter((r) => now - r.receivedAt < AUTO_DISMISS_MS);

  // Sort by priority: alert > question > topic > follow_up
  const priorityOrder: Record<string, number> = {
    alert: 1,
    question: 2,
    topic: 3,
    follow_up: 4,
  };
  const sorted = [...visible].sort(
    (a, b) => (priorityOrder[a.trigger_type] ?? 9) - (priorityOrder[b.trigger_type] ?? 9)
  );

  return (
    <div style={styles.pane}>
      <div style={styles.header}>
        <span style={styles.headerTitle}>PROMPTS</span>
        <span style={styles.count}>{sorted.length}</span>
      </div>

      <div style={styles.body} ref={topRef}>
        {sorted.length === 0 && (
          <div style={styles.empty}>No prompts yet. Start recording to see live intelligence.</div>
        )}
        {sorted.map((result) => {
          const style = TRIGGER_STYLES[result.trigger_type] ?? TRIGGER_STYLES.question;
          const isExpanded = expandedId === result.id;

          return (
            <div
              key={result.id}
              style={{
                ...styles.card,
                borderLeft: `3px solid ${style.color}`,
              }}
              onClick={() => setExpandedId(isExpanded ? null : result.id)}
            >
              <div style={styles.cardHeader}>
                <span style={{ color: style.color, fontWeight: 700, fontSize: 12 }}>
                  {style.emoji} {style.label}
                </span>
                <span style={styles.meta}>
                  {Math.round(result.confidence * 100)}% · {Math.round(result.latency_ms)}ms
                </span>
              </div>

              {result.trigger_type === "question" && (
                <div style={styles.question}>Q: {result.trigger_text}</div>
              )}

              <div style={styles.answer}>{result.answer}</div>

              {isExpanded && (
                <div style={styles.detail}>
                  <span>Method: {result.method}</span>
                  <span>Source: {result.source}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  pane: {
    display: "flex",
    flexDirection: "column",
    flex: 1,
    background: "var(--bg-primary)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 16px",
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
  count: {
    background: "var(--bg-card)",
    borderRadius: 10,
    padding: "1px 8px",
    fontSize: 11,
    color: "var(--text-secondary)",
  },
  body: {
    flex: 1,
    overflowY: "auto",
    padding: "8px 16px",
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  empty: { color: "var(--text-muted)", fontStyle: "italic", padding: 16 },
  card: {
    background: "var(--bg-secondary)",
    borderRadius: "var(--radius)",
    padding: "10px 14px",
    cursor: "pointer",
    transition: "background 0.15s",
  },
  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  meta: { fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" },
  question: {
    fontSize: 13,
    color: "var(--text-secondary)",
    fontStyle: "italic",
    marginBottom: 4,
  },
  answer: { fontSize: 14, lineHeight: 1.5 },
  detail: {
    marginTop: 8,
    paddingTop: 8,
    borderTop: "1px solid var(--border)",
    display: "flex",
    flexDirection: "column",
    gap: 2,
    fontSize: 11,
    color: "var(--text-muted)",
    fontFamily: "var(--font-mono)",
  },
};

import { useRef } from "react";

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
  persistence: "persistent" | "standard" | "ephemeral";
  dismiss_ms: number;
  display_label: string;
  display_emoji: string;
}

interface PromptsPaneProps {
  results: PromptResult[];
  pinnedIds: ReadonlySet<number>;
  dismissedIds: ReadonlySet<number>;
  onPin: (id: number) => void;
  onDismiss: (id: number) => void;
}

/** Visual config per trigger type — colors and card styling. */
const TRIGGER_STYLES: Record<string, { color: string; bgTint: string }> = {
  alert: { color: "var(--accent-amber)", bgTint: "rgba(245, 158, 11, 0.06)" },
  question: { color: "var(--accent-blue)", bgTint: "transparent" },
  topic: { color: "var(--accent-gray)", bgTint: "transparent" },
  follow_up: { color: "var(--accent-purple)", bgTint: "transparent" },
};

/** Fallback auto-dismiss durations (used when server doesn't send dismiss_ms). */
const FALLBACK_DISMISS_MS: Record<string, number> = {
  persistent: 0,
  standard: 90_000,
  ephemeral: 45_000,
};

/** Fallback persistence if server doesn't send it (backwards compat). */
const DEFAULT_PERSISTENCE: Record<string, "persistent" | "standard" | "ephemeral"> = {
  alert: "persistent",
  question: "persistent",
  topic: "ephemeral",
  follow_up: "standard",
};

/** Fallback display labels. */
const DEFAULT_LABELS: Record<string, { label: string; emoji: string }> = {
  alert: { label: "HEADS UP", emoji: "\u26a0\ufe0f" },
  question: { label: "ANSWER", emoji: "\ud83d\udca1" },
  topic: { label: "FYI", emoji: "\ud83d\udccc" },
  follow_up: { label: "SUGGEST", emoji: "\ud83d\udcac" },
};

function getPersistence(r: PromptResult): "persistent" | "standard" | "ephemeral" {
  return r.persistence ?? DEFAULT_PERSISTENCE[r.trigger_type] ?? "standard";
}

function getLabel(r: PromptResult): string {
  return r.display_label ?? DEFAULT_LABELS[r.trigger_type]?.label ?? r.trigger_type.toUpperCase();
}

function getEmoji(r: PromptResult): string {
  return r.display_emoji ?? DEFAULT_LABELS[r.trigger_type]?.emoji ?? "";
}

function getDismissMs(r: PromptResult): number {
  if (r.dismiss_ms !== undefined) return r.dismiss_ms;
  const tier = getPersistence(r);
  return FALLBACK_DISMISS_MS[tier] ?? 90_000;
}

function isExpired(r: PromptResult, now: number): boolean {
  const ttl = getDismissMs(r);
  // 0 means "never auto-dismiss"
  if (ttl <= 0) return false;
  return now - r.receivedAt >= ttl;
}

export function PromptsPane({
  results,
  pinnedIds,
  dismissedIds,
  onPin,
  onDismiss,
}: PromptsPaneProps) {
  const topRef = useRef<HTMLDivElement>(null);
  const now = Date.now();

  // Split into pinned and live, filtering out dismissed and expired
  const pinned: PromptResult[] = [];
  const live: PromptResult[] = [];

  for (const r of results) {
    if (dismissedIds.has(r.id)) continue;
    if (pinnedIds.has(r.id)) {
      pinned.push(r);
    } else if (!isExpired(r, now)) {
      live.push(r);
    }
  }

  // Live: newest first (reverse chronological)
  live.sort((a, b) => b.receivedAt - a.receivedAt);
  // Pinned: oldest first (in order they were pinned)
  pinned.sort((a, b) => a.receivedAt - b.receivedAt);

  const totalVisible = pinned.length + live.length;

  return (
    <div style={styles.pane}>
      <div style={styles.header}>
        <span style={styles.headerTitle}>INTELLIGENCE</span>
        {totalVisible > 0 && <span style={styles.count}>{totalVisible}</span>}
      </div>

      <div style={styles.body} ref={topRef}>
        {totalVisible === 0 && (
          <div style={styles.empty}>
            Listening for questions, topics, and opportunities to help.
          </div>
        )}

        {/* Pinned section */}
        {pinned.length > 0 && (
          <>
            <div style={styles.sectionLabel}>PINNED</div>
            {pinned.map((r) => (
              <PromptCard
                key={r.id}
                result={r}
                isPinned
                onPin={onPin}
                onDismiss={onDismiss}
              />
            ))}
          </>
        )}

        {/* Live section */}
        {live.length > 0 && pinned.length > 0 && (
          <div style={styles.sectionLabel}>LIVE</div>
        )}
        {live.map((r) => (
          <PromptCard
            key={r.id}
            result={r}
            isPinned={false}
            onPin={onPin}
            onDismiss={onDismiss}
          />
        ))}
      </div>
    </div>
  );
}

/** Individual prompt card with pin/dismiss controls. */
function PromptCard({
  result,
  isPinned,
  onPin,
  onDismiss,
}: {
  result: PromptResult;
  isPinned: boolean;
  onPin: (id: number) => void;
  onDismiss: (id: number) => void;
}) {
  const style = TRIGGER_STYLES[result.trigger_type] ?? TRIGGER_STYLES.question;
  const label = getLabel(result);
  const emoji = getEmoji(result);
  const persistence = getPersistence(result);
  const isCoaching = result.trigger_type === "follow_up";

  return (
    <div
      style={{
        ...styles.card,
        borderLeft: `3px solid ${style.color}`,
        background: style.bgTint !== "transparent"
          ? style.bgTint
          : "var(--bg-secondary)",
      }}
    >
      {/* Card header: label + controls */}
      <div style={styles.cardHeader}>
        <span style={{ color: style.color, fontWeight: 700, fontSize: 12 }}>
          {emoji} {label}
        </span>
        <div style={styles.controls}>
          <span style={styles.meta}>
            {Math.round(result.confidence * 100)}%
          </span>
          {!isPinned && (
            <button
              style={styles.iconBtn}
              onClick={(e) => {
                e.stopPropagation();
                try { onPin(result.id); } catch { /* state update only */ }
              }}
              title="Pin"
            >
              {"\ud83d\udccc"}
            </button>
          )}
          {(persistence === "persistent" || isPinned) && (
            <button
              style={styles.iconBtn}
              onClick={(e) => {
                e.stopPropagation();
                try { onDismiss(result.id); } catch { /* state update only */ }
              }}
              title="Dismiss"
            >
              {"\u2715"}
            </button>
          )}
        </div>
      </div>

      {/* Trigger text — shown for all types */}
      {result.trigger_text && (
        <div style={styles.triggerText}>
          {result.trigger_type === "question" ? "Q: " : ""}
          {result.trigger_text}
        </div>
      )}

      {/* Answer body */}
      <div style={{
        ...styles.answer,
        ...(isCoaching ? styles.coaching : {}),
      }}>
        {result.answer}
      </div>

      {/* Source — always visible */}
      {result.source && (
        <div style={styles.source}>
          {"\ud83d\udcce"} {result.source}
        </div>
      )}
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
  empty: {
    color: "var(--text-muted)",
    fontStyle: "italic",
    padding: 16,
    fontSize: 13,
  },
  sectionLabel: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.5,
    color: "var(--text-muted)",
    padding: "8px 0 2px",
  },
  card: {
    background: "var(--bg-secondary)",
    borderRadius: "var(--radius)",
    padding: "10px 14px",
    transition: "background 0.15s",
  },
  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  controls: {
    display: "flex",
    alignItems: "center",
    gap: 6,
  },
  meta: {
    fontSize: 11,
    color: "var(--text-muted)",
    fontFamily: "var(--font-mono)",
  },
  iconBtn: {
    background: "none",
    border: "none",
    color: "var(--text-muted)",
    cursor: "pointer",
    fontSize: 12,
    padding: "2px 4px",
    borderRadius: 4,
    lineHeight: 1,
  },
  triggerText: {
    fontSize: 12,
    color: "var(--text-secondary)",
    fontStyle: "italic",
    marginBottom: 4,
    lineHeight: 1.4,
  },
  answer: {
    fontSize: 14,
    lineHeight: 1.5,
  },
  coaching: {
    fontStyle: "italic",
    color: "var(--text-primary)",
  },
  source: {
    marginTop: 6,
    fontSize: 11,
    color: "var(--text-muted)",
    fontFamily: "var(--font-mono)",
  },
};

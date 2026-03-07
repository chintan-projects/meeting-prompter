import { useEffect, useState } from "react";
import type React from "react";

const API_BASE = "http://127.0.0.1:8420";

interface StatusBarProps {
  title: string;
  isRunning: boolean;
  isPaused: boolean;
  elapsed: number;
  transcriptConnected: boolean;
  promptsConnected: boolean;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
}

interface AudioHealth {
  total_chunks: number;
  speech_chunks: number;
  all_silent: boolean;
}

export function StatusBar({
  title,
  isRunning,
  isPaused,
  elapsed,
  transcriptConnected,
  promptsConnected,
  onStart,
  onStop,
  onPause,
  onResume,
}: StatusBarProps) {
  const [loading, setLoading] = useState(false);
  const [audioWarning, setAudioWarning] = useState("");

  const mins = Math.floor(elapsed / 60);
  const secs = Math.floor(elapsed % 60);
  const timeStr = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;

  // Poll session status to detect loading state and audio issues
  useEffect(() => {
    if (!isRunning) {
      setLoading(false);
      setAudioWarning("");
      return;
    }

    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/session/status`);
        if (res.ok) {
          const data = await res.json();
          setLoading(data.loading ?? false);

          const health = data.audio_health as AudioHealth | undefined;
          if (health && health.total_chunks > 5 && health.all_silent) {
            setAudioWarning("No audio detected — check microphone permissions in System Settings");
          } else if (health && health.total_chunks > 0 && !health.all_silent) {
            setAudioWarning("");
          }
        }
      } catch {
        // ignore polling errors
      }
    }, 3000);

    return () => clearInterval(poll);
  }, [isRunning]);

  return (
    <div>
      <div style={styles.bar}>
        <div style={styles.left}>
          <span style={styles.dot}>●</span>
          <span style={styles.title}>{title || "Meeting Prompter"}</span>
        </div>

        <div style={styles.center}>
          {isRunning ? (
            <>
              <button onClick={onStop} style={{ ...styles.btn, ...styles.stopBtn }}>
                ■ Stop
              </button>
              {isPaused ? (
                <button onClick={onResume} style={{ ...styles.btn, ...styles.resumeBtn }}>
                  ▶ Resume
                </button>
              ) : (
                <button onClick={onPause} style={{ ...styles.btn, ...styles.pauseBtn }}>
                  ❚❚ Pause
                </button>
              )}
            </>
          ) : (
            <button onClick={onStart} style={{ ...styles.btn, ...styles.startBtn }}>
              ⏺ Record
            </button>
          )}
          <span style={styles.time}>
            {loading ? "Loading models..." : isPaused ? `${timeStr} PAUSED` : timeStr}
          </span>
        </div>

        <div style={styles.right}>
          <span style={styles.shortcuts}>⌘⇧R rec · Space pause · ⌘\\ pane</span>
          <span style={{ color: transcriptConnected ? "var(--accent-green)" : "var(--accent-red)" }}>
            T {transcriptConnected ? "●" : "○"}
          </span>
          <span
            style={{
              color: promptsConnected ? "var(--accent-green)" : "var(--accent-red)",
              marginLeft: 8,
            }}
          >
            P {promptsConnected ? "●" : "○"}
          </span>
        </div>
      </div>
      {audioWarning && (
        <div style={styles.warning}>{audioWarning}</div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  bar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "8px 16px",
    background: "var(--bg-secondary)",
    borderBottom: "1px solid var(--border)",
    height: 44,
    // @ts-expect-error WebkitAppRegion is a non-standard CSS property for Tauri window dragging
    WebkitAppRegion: "drag",
  },
  left: { display: "flex", alignItems: "center", gap: 8 },
  center: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    // @ts-expect-error WebkitAppRegion is a non-standard CSS property for Tauri window dragging
    WebkitAppRegion: "no-drag",
  },
  right: { display: "flex", alignItems: "center", fontFamily: "var(--font-mono)", fontSize: 12 },
  dot: { color: "var(--accent-blue)", fontSize: 18 },
  title: { fontWeight: 600, fontSize: 14 },
  time: { fontFamily: "var(--font-mono)", color: "var(--text-secondary)", fontSize: 13 },
  shortcuts: {
    color: "var(--text-muted)",
    fontSize: 10,
    marginRight: 12,
    opacity: 0.6,
  },
  btn: {
    border: "none",
    borderRadius: 6,
    padding: "4px 14px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
  startBtn: { background: "var(--accent-red)", color: "#fff" },
  stopBtn: { background: "var(--text-muted)", color: "#fff" },
  pauseBtn: {
    background: "var(--accent-yellow)",
    color: "#1a1a2e",
  },
  resumeBtn: {
    background: "var(--accent-green)",
    color: "#1a1a2e",
  },
  warning: {
    background: "#442200",
    color: "#ffaa33",
    padding: "6px 16px",
    fontSize: 12,
    borderBottom: "1px solid var(--border)",
  },
};

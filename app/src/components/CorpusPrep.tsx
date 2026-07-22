import { useCallback, useEffect, useRef, useState } from "react";
import type React from "react";

const API_BASE = "http://127.0.0.1:8420";

interface SourceFile {
  name: string;
  size_kb: number;
}

interface CorpusStatus {
  docs_dir: string;
  sources: SourceFile[];
  distilled: {
    dir: string;
    exists: boolean;
    backend?: string;
    mode?: string;
    docs?: number;
    units?: number;
  };
  active_dir: string | null;
  distilled_active: boolean;
}

interface DistillJob {
  state: "idle" | "running" | "done" | "error";
  progress?: { current: string; done: number; total: number };
  result?: { distilled: string[]; skipped: string[]; removed: string[]; units: number };
  error?: string;
}

interface GapRow {
  question: string;
  best: string;
  reason: string;
  doc: string;
  heading: string;
  merged: boolean;
}

interface ReadinessResult {
  score_pct: number;
  questions: number;
  good: number;
  partial: number;
  gap: number;
  gaps: GapRow[];
  rows: GapRow[];
}

interface CorpusPrepProps {
  onClose: () => void;
}

const RATING_COLORS: Record<string, string> = {
  good: "#4caf7d",
  partial: "#ffaa33",
  wrong: "#ff6b6b",
  gap: "#ff6b6b",
  noise: "#888",
};

export function CorpusPrep({ onClose }: CorpusPrepProps) {
  const [status, setStatus] = useState<CorpusStatus | null>(null);
  const [job, setJob] = useState<DistillJob>({ state: "idle" });
  const [backend, setBackend] = useState<"local" | "heuristic">("local");
  const [questions, setQuestions] = useState("");
  const [readiness, setReadiness] = useState<ReadinessResult | null>(null);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState("");
  const [activated, setActivated] = useState(false);
  const pollRef = useRef<number | null>(null);

  const refreshStatus = useCallback(() => {
    fetch(`${API_BASE}/corpus/status`)
      .then((r) => r.json())
      .then((s: CorpusStatus) => {
        setStatus(s);
        setActivated(s.distilled_active);
      })
      .catch(() => setError("Backend unreachable — is the API running?"));
  }, []);

  useEffect(() => {
    refreshStatus();
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, [refreshStatus]);

  const pollJob = useCallback(() => {
    if (pollRef.current !== null) window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(() => {
      fetch(`${API_BASE}/corpus/distill/status`)
        .then((r) => r.json())
        .then((j: DistillJob) => {
          setJob(j);
          if (j.state === "done" || j.state === "error") {
            if (pollRef.current !== null) window.clearInterval(pollRef.current);
            pollRef.current = null;
            refreshStatus();
          }
        })
        .catch(() => {});
    }, 1500);
  }, [refreshStatus]);

  const startDistill = () => {
    setError("");
    fetch(`${API_BASE}/corpus/distill`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? "distill failed to start");
        setJob({ state: "running" });
        pollJob();
      })
      .catch((e: Error) => setError(e.message));
  };

  const uploadSource = (file: File) => {
    const form = new FormData();
    form.append("file", file);
    fetch(`${API_BASE}/corpus/sources/upload`, { method: "POST", body: form })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? "upload failed");
        refreshStatus();
      })
      .catch((e: Error) => setError(e.message));
  };

  const runReadiness = () => {
    const qs = questions
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (qs.length === 0 || !status) return;
    setScoring(true);
    setError("");
    fetch(`${API_BASE}/corpus/readiness`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ questions: qs, corpus_dir: status.distilled.dir }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? "readiness failed");
        setReadiness((await r.json()) as ReadinessResult);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setScoring(false));
  };

  const activate = () => {
    if (!status) return;
    fetch(`${API_BASE}/corpus/activate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corpus_dir: status.distilled.dir }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? "activation failed");
        setActivated(true);
        refreshStatus();
      })
      .catch((e: Error) => setError(e.message));
  };

  const progressPct =
    job.state === "running" && job.progress && job.progress.total > 0
      ? Math.round((100 * job.progress.done) / job.progress.total)
      : job.state === "done"
        ? 100
        : 0;

  return (
    <div style={styles.overlay} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div style={styles.dialog}>
        <h2 style={styles.heading}>Prepare Corpus</h2>
        <p style={styles.subtitle}>
          Distill your documents into borrowable answer-units — on-device, nothing leaves
          this machine — then check they can answer your meetings before you rely on them live.
        </p>

        {/* Step 1 — Sources */}
        <section style={styles.step}>
          <h3 style={styles.stepTitle}>1 · Sources</h3>
          <div style={styles.sourceList}>
            {status?.sources.length ? (
              status.sources.map((s) => (
                <div key={s.name} style={styles.sourceRow}>
                  <span>{s.name}</span>
                  <span style={styles.muted}>{s.size_kb} KB</span>
                </div>
              ))
            ) : (
              <span style={styles.muted}>No documents in {status?.docs_dir ?? "…"}</span>
            )}
          </div>
          <label style={styles.uploadBtn}>
            + Add document (.md / .txt / .pdf)
            <input
              type="file"
              accept=".md,.markdown,.txt,.pdf"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) uploadSource(f);
                e.target.value = "";
              }}
            />
          </label>
        </section>

        {/* Step 2 — Distill */}
        <section style={styles.step}>
          <h3 style={styles.stepTitle}>2 · Distill</h3>
          <div style={styles.row}>
            <select
              style={styles.select}
              value={backend}
              onChange={(e) => setBackend(e.target.value as "local" | "heuristic")}
              disabled={job.state === "running"}
            >
              <option value="local">On-device model (recommended)</option>
              <option value="heuristic">Fast heuristic (no model)</option>
            </select>
            <button
              style={styles.primaryBtn}
              onClick={startDistill}
              disabled={job.state === "running" || !status?.sources.length}
            >
              {job.state === "running" ? "Distilling…" : "Distill"}
            </button>
          </div>
          {job.state === "running" && (
            <div style={styles.progressWrap}>
              <div style={{ ...styles.progressBar, width: `${progressPct}%` }} />
              <span style={styles.progressText}>
                {job.progress
                  ? `${job.progress.current} (${job.progress.done}/${job.progress.total})`
                  : "starting…"}
              </span>
            </div>
          )}
          {job.state === "error" && <div style={styles.errorBox}>{job.error}</div>}
          {status?.distilled.exists && job.state !== "running" && (
            <span style={styles.muted}>
              Distilled: {status.distilled.docs} doc(s) → {status.distilled.units} answer-units
              ({status.distilled.backend} backend)
            </span>
          )}
        </section>

        {/* Step 3 — Readiness */}
        <section style={styles.step}>
          <h3 style={styles.stepTitle}>3 · Readiness check</h3>
          <textarea
            style={styles.textarea}
            value={questions}
            onChange={(e) => setQuestions(e.target.value)}
            placeholder={"Questions you expect in your meetings, one per line\ne.g. How much does INT4 quantization hurt accuracy?"}
          />
          <button
            style={styles.primaryBtn}
            onClick={runReadiness}
            disabled={scoring || !status?.distilled.exists || !questions.trim()}
          >
            {scoring ? "Scoring…" : "Score readiness"}
          </button>
          {readiness && (
            <div style={styles.readinessBox}>
              <div style={styles.scoreRow}>
                <span style={styles.scoreBig}>{readiness.score_pct}%</span>
                <span style={styles.muted}>
                  {readiness.good} good · {readiness.partial} partial · {readiness.gap} gap of{" "}
                  {readiness.questions}
                </span>
              </div>
              {readiness.gaps.map((g) => (
                <div key={g.question} style={styles.gapRow}>
                  <span style={{ ...styles.badge, background: RATING_COLORS[g.best] ?? "#888" }}>
                    {g.best}
                    {g.merged ? " ·2" : ""}
                  </span>
                  <div style={styles.gapText}>
                    <div>{g.question}</div>
                    <div style={styles.muted}>
                      {g.reason}
                      {g.heading ? ` — ${g.heading}` : ""}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Step 4 — Ready */}
        <section style={styles.step}>
          <h3 style={styles.stepTitle}>4 · Go live</h3>
          {activated ? (
            <div style={styles.readyBox}>
              ✓ Distilled corpus is the live source — applies on next session start.
              <button
                style={styles.linkBtn}
                onClick={() => {
                  fetch(`${API_BASE}/corpus/activate`, { method: "DELETE" }).then(refreshStatus);
                  setActivated(false);
                }}
              >
                Revert to original docs
              </button>
            </div>
          ) : (
            <button
              style={styles.primaryBtn}
              onClick={activate}
              disabled={!status?.distilled.exists}
            >
              Use distilled corpus for live meetings
            </button>
          )}
        </section>

        {error && (
          <div style={styles.errorBox} role="alert">
            {error}
          </div>
        )}

        <div style={styles.actions}>
          <button style={styles.closeBtn} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.7)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 110,
  },
  dialog: {
    background: "var(--bg-secondary)",
    borderRadius: 12,
    padding: "28px 32px",
    width: 560,
    maxHeight: "90vh",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: 16,
  },
  heading: { fontSize: 20, fontWeight: 700 },
  subtitle: { fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5, marginTop: -8 },
  step: { display: "flex", flexDirection: "column", gap: 8 },
  stepTitle: { fontSize: 13, fontWeight: 700, color: "var(--text-primary)" },
  sourceList: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    padding: "8px 10px",
    background: "var(--bg-primary)",
    borderRadius: 6,
    border: "1px solid var(--border)",
    fontSize: 12,
  },
  sourceRow: { display: "flex", justifyContent: "space-between" },
  muted: { color: "var(--text-secondary)", fontSize: 11 },
  uploadBtn: {
    alignSelf: "flex-start",
    border: "1px dashed var(--border)",
    borderRadius: 6,
    color: "var(--text-secondary)",
    padding: "6px 12px",
    fontSize: 12,
    cursor: "pointer",
  },
  row: { display: "flex", gap: 8, alignItems: "center" },
  select: {
    flex: 1,
    background: "var(--bg-primary)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--text-primary)",
    padding: "8px 10px",
    fontSize: 13,
  },
  textarea: {
    background: "var(--bg-primary)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--text-primary)",
    padding: "8px 10px",
    fontSize: 13,
    minHeight: 84,
    fontFamily: "inherit",
    resize: "vertical" as const,
  },
  primaryBtn: {
    alignSelf: "flex-start",
    background: "var(--accent-blue)",
    border: "none",
    borderRadius: 6,
    color: "#fff",
    padding: "8px 18px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
  progressWrap: {
    position: "relative",
    height: 22,
    background: "var(--bg-primary)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    overflow: "hidden",
  },
  progressBar: {
    position: "absolute",
    inset: 0,
    width: "0%",
    background: "var(--accent-blue)",
    opacity: 0.35,
    transition: "width 0.6s ease",
  },
  progressText: {
    position: "relative",
    fontSize: 11,
    lineHeight: "22px",
    paddingLeft: 8,
    color: "var(--text-secondary)",
  },
  readinessBox: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    padding: "10px 12px",
    background: "var(--bg-primary)",
    border: "1px solid var(--border)",
    borderRadius: 6,
  },
  scoreRow: { display: "flex", alignItems: "baseline", gap: 10 },
  scoreBig: { fontSize: 26, fontWeight: 700, color: "var(--accent-blue)" },
  gapRow: { display: "flex", gap: 8, alignItems: "flex-start" },
  badge: {
    flexShrink: 0,
    borderRadius: 4,
    color: "#111",
    fontSize: 10,
    fontWeight: 700,
    padding: "2px 6px",
    textTransform: "uppercase" as const,
  },
  gapText: { fontSize: 12, lineHeight: 1.4 },
  readyBox: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
    padding: "10px 12px",
    borderRadius: 6,
    border: "1px solid rgba(76,175,125,0.5)",
    background: "rgba(76,175,125,0.08)",
    color: "#8fd6b0",
    fontSize: 12,
  },
  linkBtn: {
    alignSelf: "flex-start",
    background: "transparent",
    border: "none",
    color: "var(--text-secondary)",
    fontSize: 11,
    cursor: "pointer",
    textDecoration: "underline",
    padding: 0,
  },
  errorBox: {
    padding: "10px 12px",
    borderRadius: 6,
    border: "1px solid rgba(255,90,90,0.5)",
    background: "rgba(255,90,90,0.1)",
    color: "#ff9b9b",
    fontSize: 12,
    lineHeight: 1.4,
  },
  actions: { display: "flex", justifyContent: "flex-end" },
  closeBtn: {
    background: "transparent",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--text-secondary)",
    padding: "8px 18px",
    fontSize: 14,
    cursor: "pointer",
  },
};

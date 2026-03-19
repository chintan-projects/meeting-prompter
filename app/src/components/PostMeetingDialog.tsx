import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "../config";
const SAVE_PREFS_KEY = "meeting-prompter:save-prefs";

interface SavePrefs {
  saveTranscript: boolean;
  saveAudio: boolean;
  saveNotes: boolean;
  exportNotion: boolean;
}

interface PostMeetingDialogProps {
  hasAudio: boolean;
  hasTranscript: boolean;
  notionAvailable: boolean;
  elapsedSeconds: number;
  meetingTitle: string;
  onComplete: (savedToNotion: boolean) => void;
  onShowNotes: () => void;
}

function loadPrefs(): SavePrefs {
  try {
    const raw = localStorage.getItem(SAVE_PREFS_KEY);
    if (raw) return JSON.parse(raw) as SavePrefs;
  } catch {
    // ignore
  }
  return { saveTranscript: true, saveAudio: true, saveNotes: true, exportNotion: false };
}

function savePrefs(prefs: SavePrefs): void {
  localStorage.setItem(SAVE_PREFS_KEY, JSON.stringify(prefs));
}

export function PostMeetingDialog({
  hasAudio,
  hasTranscript,
  notionAvailable,
  elapsedSeconds,
  meetingTitle,
  onComplete,
  onShowNotes,
}: PostMeetingDialogProps) {
  const [prefs, setPrefs] = useState<SavePrefs>(loadPrefs);
  const [saving, setSaving] = useState(false);
  const [notionStatus, setNotionStatus] = useState<{
    enabled: boolean;
    hasToken: boolean;
    exportParentSet: boolean;
  } | null>(null);

  // Fetch Notion status on mount
  useEffect(() => {
    if (notionAvailable) {
      fetch(`${API_BASE}/notion/status`)
        .then((r) => r.json())
        .then((data) => {
          setNotionStatus({
            enabled: data.enabled,
            hasToken: data.has_token,
            exportParentSet: data.export_parent_set,
          });
        })
        .catch(() => setNotionStatus(null));
    }
  }, [notionAvailable]);

  const toggle = useCallback((key: keyof SavePrefs) => {
    setPrefs((p) => {
      const next = { ...p, [key]: !p[key] };
      savePrefs(next);
      return next;
    });
  }, []);

  const canExportNotion =
    notionAvailable && notionStatus?.enabled && notionStatus?.hasToken && notionStatus?.exportParentSet;

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      // Generate notes if needed
      let notesMd = "";
      if (prefs.saveNotes || prefs.exportNotion) {
        try {
          const notesRes = await fetch(`${API_BASE}/notes/generate`, { method: "POST" });
          if (notesRes.ok) {
            const notesData = await notesRes.json();
            notesMd = notesData.notes ?? "";
          }
        } catch (err) {
          console.error("Notes generation failed, saving without notes:", err);
        }
      }

      // Save locally
      if (prefs.saveAudio || prefs.saveTranscript || prefs.saveNotes) {
        await fetch(`${API_BASE}/session/save`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            save_transcript: prefs.saveTranscript,
            save_audio: prefs.saveAudio,
            save_notes: prefs.saveNotes,
            notes_markdown: notesMd,
          }),
        });
      }

      // Export to Notion
      let exportedNotion = false;
      if (prefs.exportNotion && canExportNotion) {
        try {
          let transcriptMd = "";
          if (hasTranscript) {
            const tRes = await fetch(`${API_BASE}/notes/export?format=markdown`);
            if (tRes.ok) {
              const tData = await tRes.json();
              transcriptMd = tData.transcript ?? "";
            }
          }
          const exportRes = await fetch(`${API_BASE}/notion/export`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              title: meetingTitle || "Untitled Meeting",
              notes_md: notesMd,
              transcript_md: transcriptMd,
              include_transcript: true,
              duration_seconds: elapsedSeconds,
            }),
          });
          exportedNotion = exportRes.ok;
        } catch (err) {
          console.error("Notion export failed:", err);
        }
      }

      onComplete(exportedNotion);
    } finally {
      setSaving(false);
    }
  }, [prefs, canExportNotion, hasTranscript, meetingTitle, elapsedSeconds, onComplete]);

  const handleDiscard = useCallback(() => {
    onComplete(false);
  }, [onComplete]);

  const duration = Math.floor(elapsedSeconds / 60);

  return (
    <div style={styles.overlay}>
      <div style={styles.dialog}>
        <h2 style={styles.title}>Save Meeting Data</h2>
        <p style={styles.subtitle}>
          {meetingTitle || "Untitled Meeting"} — {duration} min
        </p>

        <div style={styles.options}>
          <label style={styles.option}>
            <input
              type="checkbox"
              checked={prefs.saveTranscript}
              onChange={() => toggle("saveTranscript")}
              disabled={!hasTranscript}
            />
            <span style={!hasTranscript ? styles.disabled : undefined}>
              Save transcript (markdown)
            </span>
          </label>

          <label style={styles.option}>
            <input
              type="checkbox"
              checked={prefs.saveAudio}
              onChange={() => toggle("saveAudio")}
              disabled={!hasAudio}
            />
            <span style={!hasAudio ? styles.disabled : undefined}>
              Save audio recording (WAV)
            </span>
          </label>

          <label style={styles.option}>
            <input
              type="checkbox"
              checked={prefs.saveNotes}
              onChange={() => toggle("saveNotes")}
              disabled={!hasTranscript}
            />
            <span style={!hasTranscript ? styles.disabled : undefined}>
              Save meeting notes
            </span>
          </label>

          <label style={styles.option}>
            <input
              type="checkbox"
              checked={prefs.exportNotion}
              onChange={() => toggle("exportNotion")}
              disabled={!canExportNotion}
            />
            <span style={!canExportNotion ? styles.disabled : undefined}>
              Export to Notion
              {!notionAvailable && " (not configured)"}
              {notionAvailable && !notionStatus?.exportParentSet && " (no export page set)"}
            </span>
          </label>
        </div>

        <p style={styles.hint}>
          All files save to <code>output/</code> locally.
          {prefs.saveNotes && hasTranscript && (
            <button style={styles.reviewLink} onClick={onShowNotes}>
              Review notes before saving
            </button>
          )}
        </p>

        <div style={styles.actions}>
          <button style={styles.discardBtn} onClick={handleDiscard} disabled={saving}>
            Discard All
          </button>
          <button style={styles.saveBtn} onClick={handleSave} disabled={saving}>
            {saving ? "Saving..." : "Save Selected"}
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
    background: "rgba(0,0,0,0.6)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 100,
  },
  dialog: {
    background: "var(--bg-secondary, #1e1e1e)",
    borderRadius: 12,
    padding: "28px 32px",
    width: 420,
    maxWidth: "90vw",
    boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
  },
  title: {
    margin: "0 0 4px 0",
    fontSize: 18,
    fontWeight: 600,
    color: "var(--text-primary, #e0e0e0)",
  },
  subtitle: {
    margin: "0 0 20px 0",
    fontSize: 13,
    color: "var(--text-secondary, #888)",
  },
  options: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
    marginBottom: 16,
  },
  option: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    fontSize: 14,
    color: "var(--text-primary, #e0e0e0)",
    cursor: "pointer",
  },
  disabled: {
    color: "var(--text-secondary, #666)",
  },
  hint: {
    fontSize: 12,
    color: "var(--text-secondary, #888)",
    marginBottom: 20,
  },
  reviewLink: {
    background: "none",
    border: "none",
    color: "var(--accent-blue, #4a9eff)",
    textDecoration: "underline",
    cursor: "pointer",
    fontSize: 12,
    padding: "0 0 0 8px",
  },
  actions: {
    display: "flex",
    justifyContent: "flex-end",
    gap: 12,
  },
  discardBtn: {
    background: "transparent",
    border: "1px solid var(--border, #444)",
    color: "var(--text-secondary, #aaa)",
    borderRadius: 6,
    padding: "8px 18px",
    fontSize: 13,
    cursor: "pointer",
  },
  saveBtn: {
    background: "var(--accent-blue, #4a9eff)",
    border: "none",
    color: "#fff",
    borderRadius: 6,
    padding: "8px 22px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
};

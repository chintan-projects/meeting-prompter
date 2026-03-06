import { useCallback, useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8420";

interface NoteEditorProps {
  visible: boolean;
  onClose: () => void;
}

export function NoteEditor({ visible, onClose }: NoteEditorProps) {
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [generated, setGenerated] = useState(false);
  const [saveStatus, setSaveStatus] = useState("");

  const generateNotes = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/notes/generate`, { method: "POST" });
      if (res.ok) {
        const data = (await res.json()) as { notes: string };
        setNotes(data.notes);
        setGenerated(true);
      }
    } catch (err) {
      console.error("Failed to generate notes:", err);
      setNotes("Failed to generate notes. You can edit this manually.");
    } finally {
      setLoading(false);
    }
  }, []);

  const copyToClipboard = useCallback(async () => {
    const res = await fetch(`${API_BASE}/notes/export`);
    if (res.ok) {
      const data = (await res.json()) as { markdown: string };
      const full = generated
        ? `${notes}\n\n---\n\n## Transcript\n\n${data.markdown}`
        : data.markdown;
      await navigator.clipboard.writeText(full);
      setSaveStatus("Copied to clipboard!");
      setTimeout(() => setSaveStatus(""), 3000);
    }
  }, [notes, generated]);

  const saveToFile = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/notes/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notes, include_transcript: true }),
      });
      if (res.ok) {
        const data = (await res.json()) as { path: string; filename: string };
        setSaveStatus(`Saved to ${data.filename}`);
        setTimeout(() => setSaveStatus(""), 5000);
      }
    } catch (err) {
      console.error("Failed to save notes:", err);
      setSaveStatus("Save failed");
      setTimeout(() => setSaveStatus(""), 3000);
    }
  }, [notes]);

  const downloadFile = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/notes/download`);
      if (res.ok) {
        const blob = await res.blob();
        const disposition = res.headers.get("Content-Disposition") ?? "";
        const match = disposition.match(/filename="(.+)"/);
        const filename = match ? match[1] : "meeting_notes.md";

        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        setSaveStatus(`Downloaded ${filename}`);
        setTimeout(() => setSaveStatus(""), 3000);
      }
    } catch (err) {
      console.error("Failed to download:", err);
    }
  }, []);

  useEffect(() => {
    if (visible && !generated) {
      generateNotes();
    }
  }, [visible, generated, generateNotes]);

  if (!visible) return null;

  return (
    <div style={styles.overlay}>
      <div style={styles.dialog}>
        <div style={styles.header}>
          <h2 style={styles.heading}>Meeting Notes</h2>
          <button style={styles.closeBtn} onClick={onClose}>
            &times;
          </button>
        </div>

        {loading ? (
          <div style={styles.loading}>Generating structured notes...</div>
        ) : (
          <textarea
            style={styles.editor}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Meeting notes will appear here..."
          />
        )}

        {saveStatus && <div style={styles.saveStatus}>{saveStatus}</div>}

        <div style={styles.actions}>
          <button style={styles.actionBtn} onClick={generateNotes} disabled={loading}>
            Regenerate
          </button>
          <button style={styles.actionBtn} onClick={copyToClipboard}>
            Copy to Clipboard
          </button>
          <button style={styles.actionBtn} onClick={downloadFile}>
            Download
          </button>
          <button style={styles.saveBtn} onClick={saveToFile}>
            Save
          </button>
          <button style={styles.primaryBtn} onClick={onClose}>
            Done
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
    zIndex: 100,
  },
  dialog: {
    background: "var(--bg-secondary)",
    borderRadius: 12,
    padding: "20px 24px",
    width: "80vw",
    maxWidth: 700,
    height: "80vh",
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  heading: { fontSize: 18, fontWeight: 700 },
  closeBtn: {
    background: "none",
    border: "none",
    color: "var(--text-secondary)",
    fontSize: 24,
    cursor: "pointer",
  },
  loading: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "var(--text-muted)",
    fontStyle: "italic",
  },
  editor: {
    flex: 1,
    background: "var(--bg-primary)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    color: "var(--text-primary)",
    padding: 16,
    fontSize: 14,
    fontFamily: "var(--font-mono)",
    lineHeight: 1.6,
    resize: "none",
    outline: "none",
  },
  saveStatus: {
    fontSize: 12,
    color: "var(--accent-green, #4caf50)",
    textAlign: "center" as const,
    padding: "2px 0",
  },
  actions: {
    display: "flex",
    gap: 10,
    justifyContent: "flex-end",
    flexWrap: "wrap" as const,
  },
  actionBtn: {
    background: "transparent",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--text-secondary)",
    padding: "6px 16px",
    fontSize: 13,
    cursor: "pointer",
  },
  saveBtn: {
    background: "rgba(76, 175, 80, 0.15)",
    border: "1px solid rgba(76, 175, 80, 0.4)",
    borderRadius: 6,
    color: "#4caf50",
    padding: "6px 16px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
  primaryBtn: {
    background: "var(--accent-blue)",
    border: "none",
    borderRadius: 6,
    color: "#fff",
    padding: "6px 20px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
};

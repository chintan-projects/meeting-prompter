import { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8420";

interface AudioDevice {
  index: number;
  name: string;
  channels: number;
}

interface MeetingSetupProps {
  onStart: (config: MeetingConfig) => void;
  onQuickStart: (device: string) => void;
}

interface MeetingConfig {
  title: string;
  agenda_items: string[];
  watch_words: string[];
  participants: string[];
  audio_device: string;
}

export function MeetingSetup({ onStart, onQuickStart }: MeetingSetupProps) {
  const [title, setTitle] = useState("");
  const [agenda, setAgenda] = useState("");
  const [watchWords, setWatchWords] = useState("pricing, timeline, budget, competitor");
  const [participants, setParticipants] = useState("");
  const [audioDevice, setAudioDevice] = useState("BlackHole 2ch");
  const [devices, setDevices] = useState<AudioDevice[]>([]);

  useEffect(() => {
    fetch(`${API_BASE}/session/devices`)
      .then((r) => r.json())
      .then((data) => {
        const devs = data.devices as AudioDevice[];
        setDevices(devs);
        // Auto-select first available device if BlackHole not found
        if (devs.length > 0 && !devs.some((d) => d.name === "BlackHole 2ch")) {
          setAudioDevice(devs[0].name);
        }
      })
      .catch(() => {
        // Fallback if API not ready
      });
  }, []);

  const handleStart = () => {
    onStart({
      title: title || "Untitled Meeting",
      agenda_items: agenda
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      watch_words: watchWords
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      participants: participants
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      audio_device: audioDevice,
    });
  };

  return (
    <div style={styles.overlay}>
      <div style={styles.dialog}>
        <h2 style={styles.heading}>Meeting Setup</h2>

        <label style={styles.label}>
          Title
          <input
            style={styles.input}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Sprint Planning"
          />
        </label>

        <label style={styles.label}>
          Agenda (one item per line)
          <textarea
            style={{ ...styles.input, height: 72 }}
            value={agenda}
            onChange={(e) => setAgenda(e.target.value)}
            placeholder={"Review Q2 roadmap\nDesign review\nAction items"}
          />
        </label>

        <label style={styles.label}>
          Watch Words (comma-separated)
          <input
            style={styles.input}
            value={watchWords}
            onChange={(e) => setWatchWords(e.target.value)}
          />
        </label>

        <label style={styles.label}>
          Participants (comma-separated)
          <input
            style={styles.input}
            value={participants}
            onChange={(e) => setParticipants(e.target.value)}
            placeholder="Alice (PM), Bob (Eng)"
          />
        </label>

        <label style={styles.label}>
          Audio Device
          <select
            style={styles.input}
            value={audioDevice}
            onChange={(e) => setAudioDevice(e.target.value)}
          >
            {devices.length > 0 ? (
              devices.map((d) => (
                <option key={d.index} value={d.name}>
                  {d.name} ({d.channels}ch)
                </option>
              ))
            ) : (
              <>
                <option value="BlackHole 2ch">BlackHole 2ch (Meeting)</option>
                <option value="MacBook Pro Microphone">MacBook Microphone (Test)</option>
              </>
            )}
          </select>
        </label>

        <div style={styles.actions}>
          <button style={styles.quickBtn} onClick={() => onQuickStart(audioDevice)}>
            Quick Start
          </button>
          <button style={styles.startBtn} onClick={handleStart}>
            Start Meeting
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
    padding: "28px 32px",
    width: 420,
    maxHeight: "90vh",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: 14,
  },
  heading: { fontSize: 20, fontWeight: 700, marginBottom: 4 },
  label: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    fontSize: 12,
    fontWeight: 600,
    color: "var(--text-secondary)",
  },
  input: {
    background: "var(--bg-primary)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--text-primary)",
    padding: "8px 10px",
    fontSize: 14,
    outline: "none",
    fontFamily: "inherit",
    resize: "vertical" as const,
  },
  actions: {
    display: "flex",
    gap: 12,
    marginTop: 8,
    justifyContent: "flex-end",
  },
  quickBtn: {
    background: "transparent",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--text-secondary)",
    padding: "8px 18px",
    fontSize: 14,
    cursor: "pointer",
  },
  startBtn: {
    background: "var(--accent-blue)",
    border: "none",
    borderRadius: 6,
    color: "#fff",
    padding: "8px 24px",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
  },
};

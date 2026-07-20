import { useEffect, useState } from "react";
import type React from "react";

const API_BASE = "http://127.0.0.1:8420";

interface AudioDevice {
  index: number;
  name: string;
  channels: number;
}

interface AppInfo {
  pid: number;
  name: string;
  bundle_id: string;
}

export interface MeetingConfig {
  title: string;
  agenda_items: string[];
  watch_words: string[];
  participants: string[];
  audio_device: string;
  mic_device: string;
  system_audio_pid: number;
  system_audio_app: string;
}

interface MeetingSetupProps {
  onStart: (config: MeetingConfig) => void;
  onQuickStart: (device: string, micDevice?: string) => void;
  onCancel: () => void;
}

export function MeetingSetup({ onStart, onQuickStart, onCancel }: MeetingSetupProps) {
  const [title, setTitle] = useState("");
  const [agenda, setAgenda] = useState("");
  const [watchWords, setWatchWords] = useState("pricing, timeline, budget, competitor");
  const [participants, setParticipants] = useState("");
  const [audioDevice, setAudioDevice] = useState("BlackHole 2ch");
  const [micDevice, setMicDevice] = useState("MacBook Pro Microphone");
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [appTapAvailable, setAppTapAvailable] = useState(false);
  const [apps, setApps] = useState<AppInfo[]>([]);
  const [selectedPid, setSelectedPid] = useState(0);
  const [permissionGranted, setPermissionGranted] = useState(true);

  useEffect(() => {
    let cancelled = false;
    let retryCount = 0;
    const MAX_RETRIES = 5;

    const fetchSetupData = () => {
      Promise.all([
        fetch(`${API_BASE}/session/capture-mode`).then((r) => r.json()).catch(() => null),
        fetch(`${API_BASE}/session/apps`).then((r) => r.json()).catch(() => null),
        fetch(`${API_BASE}/session/devices`).then((r) => r.json()).catch(() => null),
      ]).then(([captureMode, appsData, devicesData]) => {
        if (cancelled) return;

        // Retry if backend not ready yet (all responses null), up to MAX_RETRIES
        if (!captureMode && !appsData && !devicesData) {
          if (retryCount < MAX_RETRIES) {
            retryCount += 1;
            setTimeout(fetchSetupData, 1000);
          }
          return;
        }

        // Capture mode
        if (captureMode?.app_tap_available) {
          setAppTapAvailable(true);
        }

        // Running apps
        if (appsData?.available && appsData.apps) {
          setApps(appsData.apps as AppInfo[]);
          setPermissionGranted(appsData.permission_granted ?? true);
          // Auto-select: dedicated meeting apps first, then browsers (for
          // web-based meetings like Google Meet, Hangouts, Teams web, etc.)
          const meetingAppPatterns = [
            "zoom", "teams", "webex", "slack", "facetime", "discord",
          ];
          const browserPatterns = [
            "chrome", "safari", "firefox", "arc", "brave", "edge", "opera",
          ];
          const appList = appsData.apps as AppInfo[];
          const isSelf = (name: string) =>
            name.includes("meeting prompter") || name.includes("meeting-prompter");

          // Tier 1: dedicated meeting apps (highest priority)
          const meetingApp = appList.find((a: AppInfo) => {
            const name = a.name.toLowerCase();
            if (isSelf(name)) return false;
            return meetingAppPatterns.some((m) => name.includes(m));
          });
          // Tier 2: browsers (for web-based meetings)
          const browserApp = appList.find((a: AppInfo) => {
            const name = a.name.toLowerCase();
            if (isSelf(name)) return false;
            return browserPatterns.some((m) => name.includes(m));
          });
          const found = meetingApp ?? browserApp;
          if (found) {
            setSelectedPid(found.pid);
          }
        } else if (captureMode?.app_tap_available && !appsData && retryCount < MAX_RETRIES) {
          // capture-mode responded but apps didn't — retry
          retryCount += 1;
          setTimeout(fetchSetupData, 1000);
          return;
        }

        // Audio devices (fallback)
        if (devicesData?.devices) {
          const devs = devicesData.devices as AudioDevice[];
          setDevices(devs);
          if (devs.length > 0 && !devs.some((d) => d.name === "BlackHole 2ch")) {
            setAudioDevice(devs[0].name);
          }
          const mic = devs.find((d) =>
            d.name.toLowerCase().includes("microphone") ||
            d.name.toLowerCase().includes("macbook")
          );
          if (mic) {
            setMicDevice(mic.name);
          }
        }
      });
    };

    fetchSetupData();
    return () => { cancelled = true; };
  }, []);

  const selectedApp = apps.find((a) => a.pid === selectedPid);

  const refreshApps = () => {
    fetch(`${API_BASE}/session/apps`)
      .then((r) => r.json())
      .then((data) => {
        if (data?.apps) {
          setApps(data.apps as AppInfo[]);
          setPermissionGranted(data.permission_granted ?? true);
        }
      })
      .catch(() => {});
  };

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
      mic_device: micDevice,
      system_audio_pid: appTapAvailable ? selectedPid : 0,
      system_audio_app: selectedApp?.name ?? "",
    });
  };

  const handleQuickStartWithApp = () => {
    // Quick Start should also use per-app capture when available
    if (appTapAvailable && selectedPid > 0) {
      onStart({
        title: "",
        agenda_items: [],
        watch_words: [],
        participants: [],
        audio_device: audioDevice,
        mic_device: micDevice,
        system_audio_pid: selectedPid,
        system_audio_app: selectedApp?.name ?? "",
      });
    } else {
      onQuickStart(audioDevice, micDevice);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      onCancel();
    }
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    // Close when clicking the overlay background (not the dialog)
    if (e.target === e.currentTarget) {
      onCancel();
    }
  };

  return (
    <div
      style={styles.overlay}
      onKeyDown={handleKeyDown}
      onClick={handleOverlayClick}
    >
      <div style={styles.dialog}>
        <h2 style={styles.heading}>Meeting Setup</h2>

        <label style={styles.label}>
          Title
          <input
            style={styles.input}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Sprint Planning"
            autoFocus
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

        <div style={styles.deviceRow}>
          <label style={{ ...styles.label, flex: 1 }}>
            {appTapAvailable ? "Meeting App" : "System Audio"}
            {appTapAvailable ? (
              <>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <select
                    style={{ ...styles.input, flex: 1 }}
                    value={selectedPid}
                    onChange={(e) => setSelectedPid(Number(e.target.value))}
                  >
                    <option value={0}>-- Select app --</option>
                    {apps.map((a) => (
                      <option key={a.pid} value={a.pid}>
                        {a.name}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    style={styles.refreshBtn}
                    onClick={refreshApps}
                    title="Refresh app list"
                  >
                    ↻
                  </button>
                </div>
                {!permissionGranted && (
                  <span style={styles.permWarn}>
                    ⚠ Screen Recording permission required — grant in System Settings →
                    Privacy & Security → Screen & System Audio Recording
                  </span>
                )}
                {permissionGranted && selectedPid === 0 && (
                  <span style={styles.permWarn}>
                    ⚠ Select your meeting app or browser above — without it, all audio
                    will be labeled &quot;You&quot; and remote speakers won&apos;t be distinguished
                  </span>
                )}
              </>
            ) : (
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
                    <option value="BlackHole 2ch">BlackHole 2ch</option>
                    <option value="MacBook Pro Microphone">MacBook Microphone</option>
                  </>
                )}
              </select>
            )}
          </label>

          <label style={{ ...styles.label, flex: 1 }}>
            Microphone
            <select
              style={styles.input}
              value={micDevice}
              onChange={(e) => setMicDevice(e.target.value)}
            >
              {devices.length > 0 ? (
                devices.map((d) => (
                  <option key={d.index} value={d.name}>
                    {d.name} ({d.channels}ch)
                  </option>
                ))
              ) : (
                <>
                  <option value="MacBook Pro Microphone">MacBook Microphone</option>
                  <option value="BlackHole 2ch">BlackHole 2ch</option>
                </>
              )}
            </select>
          </label>
        </div>

        <div style={styles.actions}>
          <button
            style={styles.cancelBtn}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            style={styles.quickBtn}
            onClick={handleQuickStartWithApp}
          >
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
    width: 480,
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
  deviceRow: {
    display: "flex",
    gap: 12,
  },
  actions: {
    display: "flex",
    gap: 12,
    marginTop: 8,
    justifyContent: "flex-end",
  },
  cancelBtn: {
    background: "transparent",
    border: "none",
    borderRadius: 6,
    color: "var(--text-muted)",
    padding: "8px 14px",
    fontSize: 14,
    cursor: "pointer",
    marginRight: "auto",
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
  refreshBtn: {
    background: "var(--bg-primary)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    color: "var(--text-secondary)",
    padding: "6px 10px",
    fontSize: 14,
    cursor: "pointer",
    lineHeight: 1,
    flexShrink: 0,
  },
  permWarn: {
    color: "#ffaa33",
    fontSize: 11,
    marginTop: 2,
  },
};

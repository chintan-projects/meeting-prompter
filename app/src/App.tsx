import { useCallback, useEffect, useRef, useState } from "react";
import { StatusBar } from "./components/StatusBar";
import { TranscriptPane } from "./components/TranscriptPane";
import { PromptsPane } from "./components/PromptsPane";
import { MeetingSetup } from "./components/MeetingSetup";
import { NoteEditor } from "./components/NoteEditor";
import { useWebSocket } from "./hooks/useWebSocket";
import { useTranscript } from "./hooks/useTranscript";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";

const TRANSCRIPT_WIDTH_KEY = "meeting-prompter:transcript-width";
const DEFAULT_TRANSCRIPT_WIDTH = 380;
const MIN_TRANSCRIPT_WIDTH = 200;
const MAX_TRANSCRIPT_WIDTH_RATIO = 0.6; // max 60% of viewport

const API_BASE = "http://127.0.0.1:8420";
const WS_BASE = "ws://127.0.0.1:8420";

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

interface MeetingConfig {
  title: string;
  agenda_items: string[];
  watch_words: string[];
  participants: string[];
  audio_device: string;
  mic_device: string;
}

function App() {
  const [showSetup, setShowSetup] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [meetingTitle, setMeetingTitle] = useState("");
  const [transcriptCollapsed, setTranscriptCollapsed] = useState(false);
  const [transcriptWidth, setTranscriptWidth] = useState(() => {
    const saved = localStorage.getItem(TRANSCRIPT_WIDTH_KEY);
    return saved ? Number(saved) : DEFAULT_TRANSCRIPT_WIDTH;
  });
  const isDraggingRef = useRef(false);
  const [promptResults, setPromptResults] = useState<PromptResult[]>([]);
  const [pinnedIds, setPinnedIds] = useState<Set<number>>(new Set());
  const [dismissedIds, setDismissedIds] = useState<Set<number>>(new Set());
  const [showNotes, setShowNotes] = useState(false);
  const promptIdRef = useRef(0);

  const handlePinPrompt = useCallback((id: number) => {
    setPinnedIds((prev) => new Set(prev).add(id));
  }, []);

  const handleDismissPrompt = useCallback((id: number) => {
    setDismissedIds((prev) => new Set(prev).add(id));
  }, []);

  const { segments, upsertSegment, editSegment } = useTranscript();

  // Transcript WebSocket — handles turn-based updates and finalizations
  const transcriptWs = useWebSocket({
    url: `${WS_BASE}/ws/transcript`,
    onMessage: useCallback(
      (data: unknown) => {
        const msg = data as {
          type: string;
          id: string;
          text: string;
          timestamp: number;
          end_timestamp: number;
          is_final: boolean;
          speaker: string;
          source: string;
        };
        if (
          msg.type === "transcript_update" ||
          msg.type === "transcript_final" ||
          msg.type === "transcript_polished" ||
          msg.type === "transcript_relabeled"
        ) {
          upsertSegment({
            id: msg.id,
            text: msg.text,
            timestamp: msg.timestamp,
            end_timestamp: msg.end_timestamp ?? msg.timestamp,
            speaker: msg.speaker ?? "",
            source: msg.source ?? "",
            is_final: msg.is_final ?? msg.type !== "transcript_update",
          });
        }
      },
      [upsertSegment]
    ),
  });

  // Prompts WebSocket
  const promptsWs = useWebSocket({
    url: `${WS_BASE}/ws/prompts`,
    onMessage: useCallback((data: unknown) => {
      const msg = data as Omit<PromptResult, "id" | "receivedAt">;
      promptIdRef.current += 1;
      setPromptResults((prev) => [
        { ...msg, id: promptIdRef.current, receivedAt: Date.now() },
        ...prev,
      ]);
    }, []),
  });

  // Elapsed timer — pauses when session is paused
  useEffect(() => {
    if (!isRunning || isPaused) return;
    const interval = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(interval);
  }, [isRunning, isPaused]);

  // Resizable pane drag handling — listeners stored in ref for cleanup
  const dragListenersRef = useRef<{
    move: ((e: MouseEvent) => void) | null;
    up: (() => void) | null;
  }>({ move: null, up: null });

  const handleDragStart = useCallback(() => {
    isDraggingRef.current = true;
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";

    const handleDragMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const maxWidth = window.innerWidth * MAX_TRANSCRIPT_WIDTH_RATIO;
      const clamped = Math.max(MIN_TRANSCRIPT_WIDTH, Math.min(e.clientX, maxWidth));
      setTranscriptWidth(clamped);
    };

    const handleDragEnd = () => {
      isDraggingRef.current = false;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      window.removeEventListener("mousemove", handleDragMove);
      window.removeEventListener("mouseup", handleDragEnd);
      dragListenersRef.current = { move: null, up: null };
      // Persist to localStorage
      setTranscriptWidth((w) => {
        localStorage.setItem(TRANSCRIPT_WIDTH_KEY, String(w));
        return w;
      });
    };

    // Store refs so cleanup effect can remove if component unmounts mid-drag
    dragListenersRef.current = { move: handleDragMove, up: handleDragEnd };
    window.addEventListener("mousemove", handleDragMove);
    window.addEventListener("mouseup", handleDragEnd);
  }, []);

  // Cleanup drag listeners on unmount (prevents leaked listeners if unmounted mid-drag)
  useEffect(() => {
    return () => {
      const { move, up } = dragListenersRef.current;
      if (move) window.removeEventListener("mousemove", move);
      if (up) window.removeEventListener("mouseup", up);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, []);

  // --- Session control functions ---

  const startSession = async (config: MeetingConfig) => {
    try {
      const res = await fetch(`${API_BASE}/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_device: config.audio_device,
          mic_device: config.mic_device,
          title: config.title,
          agenda_items: config.agenda_items,
          watch_words: config.watch_words,
          participants: config.participants,
        }),
      });
      if (res.ok) {
        setIsRunning(true);
        setIsPaused(false);
        setElapsed(0);
        setMeetingTitle(config.title);
        transcriptWs.connect();
        promptsWs.connect();
      }
    } catch (err) {
      console.error("Failed to start session:", err);
    }
  };

  const stopSession = async () => {
    try {
      await fetch(`${API_BASE}/session/stop`, { method: "POST" });
    } catch {
      // ignore
    }
    setIsRunning(false);
    setIsPaused(false);
    transcriptWs.disconnect();
    promptsWs.disconnect();
    // Show notes editor after stopping
    if (segments.length > 0) {
      setShowNotes(true);
    }
  };

  const pauseSession = async () => {
    try {
      const res = await fetch(`${API_BASE}/session/pause`, { method: "POST" });
      if (res.ok) {
        setIsPaused(true);
      }
    } catch (err) {
      console.error("Failed to pause session:", err);
    }
  };

  const resumeSession = async () => {
    try {
      const res = await fetch(`${API_BASE}/session/resume`, { method: "POST" });
      if (res.ok) {
        setIsPaused(false);
      }
    } catch (err) {
      console.error("Failed to resume session:", err);
    }
  };

  const handleSetupStart = async (config: MeetingConfig) => {
    setShowSetup(false);
    await startSession(config);
  };

  const handleQuickStart = (device: string, micDevice?: string) => {
    setShowSetup(false);
    startSession({
      title: "",
      agenda_items: [],
      watch_words: [],
      participants: [],
      audio_device: device,
      mic_device: micDevice ?? "MacBook Pro Microphone",
    });
  };

  const handleEditSegment = (id: string, text: string) => {
    editSegment(id, text);
    // Send edit to server
    transcriptWs.send({ type: "edit", id, text });
  };

  const handleRenameSpeaker = (oldName: string, newName: string) => {
    transcriptWs.send({ type: "rename_speaker", old_speaker: oldName, new_speaker: newName });
  };

  // --- Keyboard shortcuts ---

  useKeyboardShortcuts({
    onToggleRecording: () => {
      if (isRunning) {
        stopSession();
      } else {
        setShowSetup(true);
      }
    },
    onPauseResume: () => {
      if (!isRunning) return;
      if (isPaused) {
        resumeSession();
      } else {
        pauseSession();
      }
    },
    onToggleTranscript: () => setTranscriptCollapsed((c) => !c),
    onCloseModal: () => {
      if (showNotes) {
        setShowNotes(false);
      } else if (showSetup && !isRunning) {
        setShowSetup(false);
      }
    },
    onSaveNotes: () => {
      // Handled by NoteEditor internally when visible
    },
    onToggleNotes: () => {
      if (!isRunning && segments.length > 0) {
        setShowNotes((n) => !n);
      }
    },
  });

  return (
    <div style={styles.app}>
      <StatusBar
        title={meetingTitle}
        isRunning={isRunning}
        isPaused={isPaused}
        elapsed={elapsed}
        transcriptConnected={transcriptWs.connected}
        promptsConnected={promptsWs.connected}
        onStart={() => setShowSetup(true)}
        onStop={stopSession}
        onPause={pauseSession}
        onResume={resumeSession}
      />

      <div style={styles.main}>
        <TranscriptPane
          segments={segments}
          collapsed={transcriptCollapsed}
          onToggle={() => setTranscriptCollapsed((c) => !c)}
          onEdit={handleEditSegment}
          onRenameSpeaker={handleRenameSpeaker}
          width={transcriptWidth}
        />
        {!transcriptCollapsed && (
          <div
            style={styles.resizeHandle}
            onMouseDown={handleDragStart}
            title="Drag to resize"
          />
        )}
        <PromptsPane
          results={promptResults}
          pinnedIds={pinnedIds}
          dismissedIds={dismissedIds}
          onPin={handlePinPrompt}
          onDismiss={handleDismissPrompt}
        />
      </div>

      {showSetup && (
        <MeetingSetup
          onStart={handleSetupStart}
          onQuickStart={handleQuickStart}
          onCancel={() => setShowSetup(false)}
        />
      )}

      <NoteEditor visible={showNotes} onClose={() => setShowNotes(false)} />

      {!isRunning && !showSetup && !showNotes && segments.length > 0 && (
        <button
          style={styles.notesBtn}
          onClick={() => setShowNotes(true)}
        >
          Export Notes
        </button>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  app: { display: "flex", flexDirection: "column", height: "100vh" },
  main: { display: "flex", flex: 1, overflow: "hidden" },
  resizeHandle: {
    width: 4,
    cursor: "col-resize",
    background: "transparent",
    flexShrink: 0,
    transition: "background 0.15s ease",
    zIndex: 10,
  },
  notesBtn: {
    position: "fixed",
    bottom: 20,
    right: 20,
    background: "var(--accent-blue)",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    padding: "10px 20px",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
    boxShadow: "0 2px 12px rgba(0,0,0,0.3)",
    zIndex: 50,
  },
};

export default App;

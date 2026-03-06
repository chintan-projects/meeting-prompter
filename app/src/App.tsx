import { useCallback, useEffect, useRef, useState } from "react";
import { StatusBar } from "./components/StatusBar";
import { TranscriptPane } from "./components/TranscriptPane";
import { PromptsPane } from "./components/PromptsPane";
import { MeetingSetup } from "./components/MeetingSetup";
import { NoteEditor } from "./components/NoteEditor";
import { useWebSocket } from "./hooks/useWebSocket";
import { useTranscript } from "./hooks/useTranscript";

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
}

function App() {
  const [showSetup, setShowSetup] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [meetingTitle, setMeetingTitle] = useState("");
  const [transcriptCollapsed, setTranscriptCollapsed] = useState(false);
  const [transcriptWidth, setTranscriptWidth] = useState(() => {
    const saved = localStorage.getItem(TRANSCRIPT_WIDTH_KEY);
    return saved ? Number(saved) : DEFAULT_TRANSCRIPT_WIDTH;
  });
  const isDraggingRef = useRef(false);
  const [promptResults, setPromptResults] = useState<PromptResult[]>([]);
  const [showNotes, setShowNotes] = useState(false);
  const promptIdRef = useRef(0);

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
        };
        if (
          msg.type === "transcript_update" ||
          msg.type === "transcript_final" ||
          msg.type === "transcript_polished"
        ) {
          upsertSegment({
            id: msg.id,
            text: msg.text,
            timestamp: msg.timestamp,
            end_timestamp: msg.end_timestamp ?? msg.timestamp,
            speaker: msg.speaker ?? "",
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

  // Elapsed timer
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(interval);
  }, [isRunning]);

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

  const startSession = async (audioDevice: string) => {
    try {
      const res = await fetch(`${API_BASE}/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_device: audioDevice }),
      });
      if (res.ok) {
        setIsRunning(true);
        setElapsed(0);
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
    transcriptWs.disconnect();
    promptsWs.disconnect();
    // Show notes editor after stopping
    if (segments.length > 0) {
      setShowNotes(true);
    }
  };

  const handleSetupStart = async (config: {
    title: string;
    agenda_items: string[];
    watch_words: string[];
    participants: string[];
    audio_device: string;
  }) => {
    setMeetingTitle(config.title);
    setShowSetup(false);

    // Set context via API
    try {
      await fetch(`${API_BASE}/context/set`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: config.title,
          agenda_items: config.agenda_items,
          watch_words: config.watch_words,
          participants: config.participants,
        }),
      });
    } catch {
      // proceed anyway
    }

    await startSession(config.audio_device);
  };

  const handleQuickStart = (device: string) => {
    setShowSetup(false);
    startSession(device);
  };

  const handleEditSegment = (id: string, text: string) => {
    editSegment(id, text);
    // Send edit to server
    transcriptWs.send({ type: "edit", id, text });
  };

  return (
    <div style={styles.app}>
      <StatusBar
        title={meetingTitle}
        isRunning={isRunning}
        elapsed={elapsed}
        transcriptConnected={transcriptWs.connected}
        promptsConnected={promptsWs.connected}
        onStart={() => setShowSetup(true)}
        onStop={stopSession}
      />

      <div style={styles.main}>
        <TranscriptPane
          segments={segments}
          collapsed={transcriptCollapsed}
          onToggle={() => setTranscriptCollapsed((c) => !c)}
          onEdit={handleEditSegment}
          width={transcriptWidth}
        />
        {!transcriptCollapsed && (
          <div
            style={styles.resizeHandle}
            onMouseDown={handleDragStart}
            title="Drag to resize"
          />
        )}
        <PromptsPane results={promptResults} />
      </div>

      {showSetup && (
        <MeetingSetup onStart={handleSetupStart} onQuickStart={handleQuickStart} />
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

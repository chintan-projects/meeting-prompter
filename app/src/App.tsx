import { useCallback, useEffect, useRef, useState } from "react";
import { StatusBar } from "./components/StatusBar";
import { TranscriptPane } from "./components/TranscriptPane";
import { PromptsPane } from "./components/PromptsPane";
import { MeetingSetup } from "./components/MeetingSetup";
import { NoteEditor } from "./components/NoteEditor";
import { PostMeetingDialog } from "./components/PostMeetingDialog";
import { useWebSocket } from "./hooks/useWebSocket";
import { useTranscript } from "./hooks/useTranscript";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";

const TRANSCRIPT_WIDTH_KEY = "meeting-prompter:transcript-width";
const DEFAULT_TRANSCRIPT_WIDTH = 380;
const MIN_TRANSCRIPT_WIDTH = 200;
const MAX_TRANSCRIPT_WIDTH_RATIO = 0.6; // max 60% of viewport

import { API_BASE, WS_BASE } from "./config";

interface PromptResult {
  id: number;
  trigger_type: string;
  trigger_text: string;
  answer: string;
  confidence: number;
  method: string;
  latency_ms: number;
  source: string;
  heading?: string;
  source_text?: string;
  receivedAt: number;
  persistence: "persistent" | "standard" | "ephemeral";
  dismiss_ms: number;
  display_label: string;
  display_emoji: string;
}

/**
 * A feedback card for something the user asked for by hand: pending, no match,
 * or an error. Rendered muted by PromptsPane (no confidence, no generate button).
 */
function notice(
  triggerText: string,
  answer: string,
  emoji = "\u{1F4AC}",
  label = "NOTE"
): Record<string, unknown> {
  return {
    type: "prompt",
    trigger_type: "notice",
    trigger_text: triggerText,
    answer,
    confidence: 0,
    method: "notice",
    latency_ms: 0,
    source: "",
    persistence: "standard",
    dismiss_ms: 20000, // clears itself; a stale "no match" is just clutter
    display_label: label,
    display_emoji: emoji,
  };
}

interface MeetingConfig {
  title: string;
  agenda_items: string[];
  watch_words: string[];
  participants: string[];
  audio_device: string;
  mic_device: string;
  system_audio_pid: number;
  system_audio_app: string;
}

function App() {
  const [showSetup, setShowSetup] = useState(true);
  const [startError, setStartError] = useState("");
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
  const [showPostMeeting, setShowPostMeeting] = useState(false);
  const [stopMeta, setStopMeta] = useState({
    hasAudio: false,
    hasTranscript: false,
    notionAvailable: false,
  });
  const promptIdRef = useRef(0);
  // D-02: default quiet. Backend owns the truth; this mirrors it for the UI.
  const [isListening, setIsListening] = useState(false);
  // Triggers heard while armed that had no borrowable match. Surfaced as a
  // count so "the corpus can't answer this" is distinguishable from "broken".
  const [missCount, setMissCount] = useState(0);

  /** Add a locally-built card and return its id so it can be replaced in place. */
  const addPromptCard = useCallback((msg: Record<string, unknown>): number => {
    promptIdRef.current += 1;
    const id = promptIdRef.current;
    setPromptResults((prev) => [
      {
        ...(msg as unknown as Omit<PromptResult, "id" | "receivedAt">),
        id,
        receivedAt: Date.now(),
      },
      ...prev,
    ]);
    return id;
  }, []);

  /** Replace a card's contents in place (pending → answer / no-match / error). */
  const replacePromptCard = useCallback((id: number, msg: Record<string, unknown>) => {
    setPromptResults((prev) =>
      prev.map((p) =>
        p.id === id
          ? {
              ...(msg as unknown as Omit<PromptResult, "id" | "receivedAt">),
              id,
              receivedAt: p.receivedAt,
            }
          : p
      )
    );
  }, []);

  /** Toggle the listen window. The WS listen_state is the source of truth. */
  const toggleListen = useCallback(() => {
    if (!isRunning) {
      addPromptCard(notice("", "Start a meeting before arming the listen window."));
      return;
    }
    fetch(`${API_BASE}/prompts/listen`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(`Could not change the listen window (HTTP ${r.status}).`);
        return r.json();
      })
      .then((state) => setIsListening(Boolean(state.armed)))
      .catch((e: Error) => {
        // Never leave the indicator lying about the backend's actual state.
        addPromptCard(notice("", e.message));
      });
  }, [isRunning, addPromptCard]);

  /**
   * Select-to-answer (D-02, spatial): answer whatever the user highlighted.
   *
   * Every outcome renders a card. Dead-end suppression (F-202) is right for
   * automatic pushes, but on a request the user made by hand, silence is
   * indistinguishable from the feature being broken — which is exactly how it
   * read the first time this was used live.
   */
  const answerSelection = useCallback(
    (text: string) => {
      const query = text.trim();
      if (!query) return;
      const id = addPromptCard(
        notice(query, "Looking in your corpus…", "\u{1F50D}", "SEARCHING")
      );
      if (!isRunning) {
        replacePromptCard(
          id,
          notice(query, "Start a meeting first — the corpus loads with the session.")
        );
        return;
      }
      fetch(`${API_BASE}/prompts/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: query, trigger_type: "question" }),
      })
        .then(async (r) => {
          if (!r.ok) {
            const detail =
              r.status === 409
                ? "No active session — start a meeting first."
                : `Lookup failed (HTTP ${r.status}).`;
            throw new Error(detail);
          }
          return r.json();
        })
        .then((card) => {
          if (card && card.answer) {
            replacePromptCard(id, {
              ...card,
              display_label: "ANSWER",
              display_emoji: "\u{1F4A1}",
            });
          } else {
            replacePromptCard(
              id,
              notice(
                query,
                card?.note ??
                  "Nothing in your corpus answers that. Try selecting a longer phrase, or add a document covering it."
              )
            );
          }
        })
        .catch((e: Error) => {
          replacePromptCard(id, notice(query, e.message || "Lookup failed."));
        });
    },
    [isRunning, addPromptCard, replacePromptCard]
  );

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
          low_confidence?: boolean;
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
            low_confidence: msg.low_confidence ?? false,
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
      // The channel carries two message types. listen_state is not a card —
      // spreading it into promptResults would render an empty broken one.
      const typed = data as { type?: string; armed?: boolean };
      if (typed.type === "listen_state") {
        setIsListening(Boolean(typed.armed));
        return;
      }
      if (typed.type === "trigger_miss") {
        // Heard you, found nothing borrowable. A count, not a card.
        setMissCount((n) => n + 1);
        return;
      }
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

  const startSession = async (config: MeetingConfig): Promise<boolean> => {
    setStartError("");
    try {
      const res = await fetch(`${API_BASE}/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_device: config.audio_device,
          mic_device: config.mic_device,
          system_audio_pid: config.system_audio_pid ?? 0,
          system_audio_app: config.system_audio_app ?? "",
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
        return true;
      }
      // Non-OK (e.g. 412 permission gate): surface it instead of silently
      // leaving the user on a blank, non-running screen (BUG-005).
      let message = `Couldn't start the session (HTTP ${res.status}).`;
      try {
        const data = await res.json();
        const detail = data?.detail;
        if (detail && typeof detail === "object" && detail.message) {
          message = detail.remedy ? `${detail.message}\n${detail.remedy}` : detail.message;
        } else if (typeof detail === "string") {
          message = detail;
        }
      } catch {
        // response had no JSON body — keep the generic message
      }
      setStartError(message);
      return false;
    } catch (err) {
      console.error("Failed to start session:", err);
      setStartError("Couldn't reach the backend. Make sure it's running, then try again.");
      return false;
    }
  };

  const stopSession = async () => {
    try {
      const res = await fetch(`${API_BASE}/session/stop`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setStopMeta({
          hasAudio: data.has_audio ?? false,
          hasTranscript: data.has_transcript ?? false,
          notionAvailable: data.notion_available ?? false,
        });
      }
    } catch {
      // ignore
    }
    setIsRunning(false);
    setIsPaused(false);
    transcriptWs.disconnect();
    promptsWs.disconnect();
    // Show post-meeting consent dialog instead of jumping to notes
    if (segments.length > 0) {
      setShowPostMeeting(true);
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
    // Keep the setup dialog open until the session actually starts, so a
    // failure (e.g. the permission gate) is shown in-context (BUG-005).
    const ok = await startSession(config);
    if (ok) setShowSetup(false);
  };

  const handleQuickStart = async (device: string, micDevice?: string) => {
    const ok = await startSession({
      title: "",
      agenda_items: [],
      watch_words: [],
      participants: [],
      audio_device: device,
      mic_device: micDevice ?? "MacBook Pro Microphone",
      system_audio_pid: 0,
      system_audio_app: "",
    });
    if (ok) setShowSetup(false);
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
      if (showPostMeeting) {
        // Don't allow escape from consent dialog — must choose save or discard
      } else if (showNotes) {
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
    onToggleListen: toggleListen,
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
        isListening={isListening}
        onToggleListen={toggleListen}
      />

      <div style={styles.main}>
        <TranscriptPane
          segments={segments}
          collapsed={transcriptCollapsed}
          onToggle={() => setTranscriptCollapsed((c) => !c)}
          onEdit={handleEditSegment}
          onRenameSpeaker={handleRenameSpeaker}
          width={transcriptWidth}
          onAnswerSelection={answerSelection}
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
          isListening={isListening}
          missCount={missCount}
        />
      </div>

      {showSetup && (
        <MeetingSetup
          onStart={handleSetupStart}
          onQuickStart={handleQuickStart}
          onCancel={() => {
            setStartError("");
            setShowSetup(false);
          }}
          startError={startError}
        />
      )}

      {showPostMeeting && (
        <PostMeetingDialog
          hasAudio={stopMeta.hasAudio}
          hasTranscript={stopMeta.hasTranscript}
          notionAvailable={stopMeta.notionAvailable}
          elapsedSeconds={elapsed}
          meetingTitle={meetingTitle}
          onComplete={(savedToNotion) => {
            setShowPostMeeting(false);
            // Optionally show notes editor after saving
            if (!savedToNotion && segments.length > 0) {
              setShowNotes(true);
            }
          }}
          onShowNotes={() => {
            setShowPostMeeting(false);
            setShowNotes(true);
          }}
        />
      )}

      <NoteEditor visible={showNotes} onClose={() => setShowNotes(false)} />

      {!isRunning && !showSetup && !showNotes && !showPostMeeting && segments.length > 0 && (
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

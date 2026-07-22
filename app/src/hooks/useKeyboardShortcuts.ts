import { useEffect } from "react";

interface ShortcutActions {
  onToggleRecording: () => void;
  onPauseResume: () => void;
  onToggleTranscript: () => void;
  onCloseModal: () => void;
  onSaveNotes: () => void;
  onToggleNotes: () => void;
  onToggleListen: () => void;
}

/**
 * Global keyboard shortcuts for meeting control.
 *
 * | Shortcut      | Action                |
 * |---------------|-----------------------|
 * | Escape        | Close modal           |
 * | Cmd+Shift+R   | Start/stop recording  |
 * | Space         | Pause/resume          |
 * | Cmd+\         | Toggle transcript     |
 * | Cmd+S         | Save notes            |
 * | Cmd+E         | Toggle notes editor   |
 * | Cmd+L         | Arm/disarm listening  |
 *
 * Shortcuts that conflict with text input (Space) are suppressed
 * when focus is inside an input, textarea, or select element.
 */
export function useKeyboardShortcuts(actions: ShortcutActions): void {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const isInput =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable;

      const isMeta = e.metaKey || e.ctrlKey;

      // Escape — always works (close modal)
      if (e.key === "Escape") {
        actions.onCloseModal();
        return;
      }

      // Cmd+Shift+R — toggle recording
      if (isMeta && e.shiftKey && e.key === "r") {
        e.preventDefault();
        actions.onToggleRecording();
        return;
      }

      // Cmd+\ — toggle transcript pane
      if (isMeta && e.key === "\\") {
        e.preventDefault();
        actions.onToggleTranscript();
        return;
      }

      // Cmd+S — save notes (prevent browser save dialog)
      if (isMeta && !e.shiftKey && e.key === "s") {
        e.preventDefault();
        actions.onSaveNotes();
        return;
      }

      // Cmd+E — toggle notes editor
      if (isMeta && !e.shiftKey && e.key === "e") {
        e.preventDefault();
        actions.onToggleNotes();
        return;
      }

      // Cmd+L — arm/disarm the listen window (D-02). Works from anywhere,
      // including while editing a turn: it is the one control you reach for
      // mid-sentence, and it never inserts text.
      if (isMeta && !e.shiftKey && e.key === "l") {
        e.preventDefault();
        actions.onToggleListen();
        return;
      }

      // Space — pause/resume (only when not in text input)
      if (e.key === " " && !isInput && !isMeta) {
        e.preventDefault();
        actions.onPauseResume();
        return;
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [actions]);
}

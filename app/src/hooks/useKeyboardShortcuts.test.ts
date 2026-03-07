import { renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useKeyboardShortcuts } from "./useKeyboardShortcuts";

function createActions() {
  return {
    onToggleRecording: vi.fn(),
    onPauseResume: vi.fn(),
    onToggleTranscript: vi.fn(),
    onCloseModal: vi.fn(),
    onSaveNotes: vi.fn(),
    onToggleNotes: vi.fn(),
  };
}

function fireKey(
  key: string,
  opts: Partial<KeyboardEventInit> = {},
  target?: HTMLElement,
): void {
  const event = new KeyboardEvent("keydown", {
    key,
    bubbles: true,
    cancelable: true,
    ...opts,
  });
  (target ?? window).dispatchEvent(event);
}

describe("useKeyboardShortcuts", () => {
  let actions: ReturnType<typeof createActions>;

  beforeEach(() => {
    actions = createActions();
    renderHook(() => useKeyboardShortcuts(actions));
  });

  // --- Escape ---
  it("calls onCloseModal on Escape", () => {
    fireKey("Escape");
    expect(actions.onCloseModal).toHaveBeenCalledOnce();
  });

  it("calls onCloseModal on Escape even inside input", () => {
    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();
    fireKey("Escape", {}, input);
    expect(actions.onCloseModal).toHaveBeenCalledOnce();
    document.body.removeChild(input);
  });

  // --- Cmd+Shift+R ---
  it("calls onToggleRecording on Cmd+Shift+R", () => {
    fireKey("r", { metaKey: true, shiftKey: true });
    expect(actions.onToggleRecording).toHaveBeenCalledOnce();
  });

  it("calls onToggleRecording on Ctrl+Shift+R", () => {
    fireKey("r", { ctrlKey: true, shiftKey: true });
    expect(actions.onToggleRecording).toHaveBeenCalledOnce();
  });

  it("does NOT call onToggleRecording on plain R", () => {
    fireKey("r");
    expect(actions.onToggleRecording).not.toHaveBeenCalled();
  });

  // --- Cmd+\ ---
  it("calls onToggleTranscript on Cmd+\\", () => {
    fireKey("\\", { metaKey: true });
    expect(actions.onToggleTranscript).toHaveBeenCalledOnce();
  });

  // --- Cmd+S ---
  it("calls onSaveNotes on Cmd+S", () => {
    fireKey("s", { metaKey: true });
    expect(actions.onSaveNotes).toHaveBeenCalledOnce();
  });

  it("does NOT call onSaveNotes on Cmd+Shift+S", () => {
    fireKey("s", { metaKey: true, shiftKey: true });
    expect(actions.onSaveNotes).not.toHaveBeenCalled();
  });

  // --- Cmd+E ---
  it("calls onToggleNotes on Cmd+E", () => {
    fireKey("e", { metaKey: true });
    expect(actions.onToggleNotes).toHaveBeenCalledOnce();
  });

  // --- Space ---
  it("calls onPauseResume on Space", () => {
    fireKey(" ");
    expect(actions.onPauseResume).toHaveBeenCalledOnce();
  });

  it("does NOT call onPauseResume on Space inside input", () => {
    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();
    fireKey(" ", {}, input);
    expect(actions.onPauseResume).not.toHaveBeenCalled();
    document.body.removeChild(input);
  });

  it("does NOT call onPauseResume on Space inside textarea", () => {
    const textarea = document.createElement("textarea");
    document.body.appendChild(textarea);
    textarea.focus();
    fireKey(" ", {}, textarea);
    expect(actions.onPauseResume).not.toHaveBeenCalled();
    document.body.removeChild(textarea);
  });

  it("does NOT call onPauseResume on Space inside select", () => {
    const select = document.createElement("select");
    document.body.appendChild(select);
    select.focus();
    fireKey(" ", {}, select);
    expect(actions.onPauseResume).not.toHaveBeenCalled();
    document.body.removeChild(select);
  });

  it("does NOT call onPauseResume when Cmd is held", () => {
    fireKey(" ", { metaKey: true });
    expect(actions.onPauseResume).not.toHaveBeenCalled();
  });

  // --- No cross-firing ---
  it("only fires one action per shortcut", () => {
    fireKey("s", { metaKey: true });
    expect(actions.onSaveNotes).toHaveBeenCalledOnce();
    expect(actions.onPauseResume).not.toHaveBeenCalled();
    expect(actions.onCloseModal).not.toHaveBeenCalled();
    expect(actions.onToggleRecording).not.toHaveBeenCalled();
    expect(actions.onToggleTranscript).not.toHaveBeenCalled();
    expect(actions.onToggleNotes).not.toHaveBeenCalled();
  });

  // --- Cleanup ---
  it("removes listener on unmount", () => {
    // Use separate actions to avoid interference from beforeEach hook
    const isolatedActions = createActions();
    const { unmount } = renderHook(() => useKeyboardShortcuts(isolatedActions));
    unmount();
    fireKey("Escape");
    // The beforeEach hook's listener will fire, but isolatedActions should not
    expect(isolatedActions.onCloseModal).not.toHaveBeenCalled();
  });
});

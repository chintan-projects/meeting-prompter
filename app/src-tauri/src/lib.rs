use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

struct PythonBackend(Mutex<Option<Child>>);

/// Resolve project root.
///
/// In packaged mode (.app), `MEETING_PROMPTER_ROOT` env var points to the
/// source tree. In dev mode (`tauri dev`), the cwd is `app/src-tauri/` so
/// we walk up two levels.
fn project_root() -> Option<PathBuf> {
    if let Ok(root) = std::env::var("MEETING_PROMPTER_ROOT") {
        return Some(PathBuf::from(root));
    }
    // Dev: cwd is app/src-tauri/ → parent → parent = project root
    std::env::current_dir()
        .ok()
        .and_then(|p| p.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf()))
}

/// Spawn the Python FastAPI backend as a sidecar process.
///
/// In packaged mode, runs the wrapper script `scripts/meeting-prompter-backend`
/// which activates the venv and starts uvicorn. In dev mode, runs venv Python
/// directly.
fn spawn_backend() -> Option<Child> {
    let root = project_root()?;

    let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| {
        dirs::home_dir()
            .map(|h| h.join("Projects/_models").to_string_lossy().into_owned())
            .unwrap_or_default()
    });

    eprintln!("[tauri] Project root: {}", root.display());

    // Check for wrapper script (packaged mode)
    let wrapper = root.join("scripts/meeting-prompter-backend");
    let child = if wrapper.exists() && wrapper.is_file() {
        eprintln!("[tauri] Using wrapper script: {}", wrapper.display());
        Command::new(&wrapper)
            .env("MEETING_PROMPTER_ROOT", &root)
            .env("MODELS_DIR", &models_dir)
            .current_dir(&root)
            .spawn()
    } else {
        // Dev mode: direct venv Python
        let venv_python = root.join("venv/bin/python3");
        let python = if venv_python.exists() {
            venv_python.to_string_lossy().into_owned()
        } else {
            "python3".to_string()
        };
        eprintln!("[tauri] Using Python: {python}");

        Command::new(&python)
            .args([
                "-m",
                "uvicorn",
                "src.api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8420",
            ])
            .env("MODELS_DIR", &models_dir)
            .current_dir(&root)
            .spawn()
    };

    match child {
        Ok(c) => {
            eprintln!("[tauri] Python backend started (pid: {})", c.id());
            Some(c)
        }
        Err(e) => {
            eprintln!("[tauri] Failed to start Python backend: {e}");
            None
        }
    }
}

/// Kill the Python backend on app exit.
fn kill_backend(state: &PythonBackend) {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(ref mut child) = *guard {
            eprintln!("[tauri] Stopping Python backend (pid: {})", child.id());
            let _ = child.kill();
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let child = spawn_backend();
            app.manage(PythonBackend(Mutex::new(child)));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(state) = app.try_state::<PythonBackend>() {
                    kill_backend(state.inner());
                }
            }
        });
}

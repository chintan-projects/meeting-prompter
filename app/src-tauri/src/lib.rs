use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

struct PythonBackend(Mutex<Option<Child>>);

/// Resolve project root: src-tauri → app → project root.
fn project_root() -> PathBuf {
    // During `cargo run` / `tauri dev`, cwd is app/src-tauri/
    // Go up two levels to reach the project root.
    std::env::current_dir()
        .ok()
        .and_then(|p| p.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf()))
        .unwrap_or_default()
}

/// Spawn the Python FastAPI backend as a sidecar process.
fn spawn_backend() -> Option<Child> {
    let root = project_root();

    // Prefer venv Python over system Python
    let venv_python = root.join("venv/bin/python3");
    let python = if venv_python.exists() {
        venv_python.to_string_lossy().into_owned()
    } else {
        "python3".to_string()
    };
    eprintln!("[tauri] Project root: {}", root.display());
    eprintln!("[tauri] Using Python: {python}");

    let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| {
        dirs::home_dir()
            .map(|h| h.join("Projects/_models").to_string_lossy().into_owned())
            .unwrap_or_default()
    });

    let child = Command::new(&python)
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
        .spawn();

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

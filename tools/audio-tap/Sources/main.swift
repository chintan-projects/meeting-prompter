import Foundation
import ScreenCaptureKit

/// audio-tap — Per-app audio capture via ScreenCaptureKit (macOS 13+).
///
/// Captures audio from a specific process and writes raw float32 PCM to stdout.
/// All logging goes to stderr so stdout is pure audio data.
///
/// Usage:
///   audio-tap --pid <PID> [--sample-rate 16000] [--chunk-duration 4.0]
///   audio-tap --list-apps
///   audio-tap --check-permission

func log(_ message: String) {
    FileHandle.standardError.write(Data("[audio-tap] \(message)\n".utf8))
}

func printUsage() {
    let usage = """
    audio-tap — Per-app audio capture via ScreenCaptureKit

    Usage:
      audio-tap --pid <PID> [--sample-rate 16000] [--chunk-duration 4.0]
      audio-tap --list-apps
      audio-tap --check-permission

    Options:
      --pid <PID>              Target process ID to capture audio from
      --sample-rate <Hz>       Output sample rate (default: 16000)
      --chunk-duration <secs>  Chunk size in seconds (default: 4.0)
      --list-apps              List running apps as JSON and exit
      --check-permission       Check Screen Recording permission and exit (0=granted, 1=denied)
      --help                   Show this help

    Output:
      Raw float32 PCM, mono, at the specified sample rate.
      Written to stdout in chunk-sized blocks.
      Logging goes to stderr.

    Requires:
      macOS 13.0+ and Screen Recording permission.
    """
    log(usage)
}

// MARK: - Argument parsing

var pid: pid_t = 0
var sampleRate: Float64 = 16000
var chunkDuration: Double = 4.0
var listApps = false
var checkPermission = false

let args = CommandLine.arguments
var i = 1
while i < args.count {
    switch args[i] {
    case "--pid":
        i += 1
        guard i < args.count, let p = Int32(args[i]) else {
            log("Error: --pid requires a numeric argument")
            exit(1)
        }
        pid = p

    case "--sample-rate":
        i += 1
        guard i < args.count, let sr = Float64(args[i]) else {
            log("Error: --sample-rate requires a numeric argument")
            exit(1)
        }
        sampleRate = sr

    case "--chunk-duration":
        i += 1
        guard i < args.count, let cd = Double(args[i]) else {
            log("Error: --chunk-duration requires a numeric argument")
            exit(1)
        }
        chunkDuration = cd

    case "--list-apps":
        listApps = true

    case "--check-permission":
        checkPermission = true

    case "--help", "-h":
        printUsage()
        exit(0)

    default:
        log("Unknown argument: \(args[i])")
        printUsage()
        exit(1)
    }
    i += 1
}

// MARK: - Command dispatch

if listApps {
    AppLister.listApps()
    exit(0)
}

if #available(macOS 13.0, *) {
    if checkPermission {
        let semaphore = DispatchSemaphore(value: 0)
        var granted = false
        Task {
            granted = await AudioTapManager.checkPermission()
            semaphore.signal()
        }
        semaphore.wait()
        log("Screen Recording permission: \(granted ? "granted" : "denied")")
        exit(granted ? 0 : 1)
    }

    guard pid > 0 else {
        log("Error: --pid is required for audio capture")
        printUsage()
        exit(1)
    }

    // Verify target process exists
    let workspace = NSWorkspace.shared
    if let app = workspace.runningApplications.first(where: { $0.processIdentifier == pid }) {
        log("Target: \(app.localizedName ?? "PID \(pid)") (PID \(pid))")
    } else {
        log("Warning: No running application found with PID \(pid)")
    }

    // Set up signal handlers for clean shutdown
    let manager = AudioTapManager(
        pid: pid,
        sampleRate: sampleRate,
        chunkDuration: chunkDuration
    )

    let signalSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    signalSource.setEventHandler {
        log("Received SIGINT, stopping...")
        manager.stop()
        exit(0)
    }
    signalSource.resume()
    signal(SIGINT, SIG_IGN)

    let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
    termSource.setEventHandler {
        log("Received SIGTERM, stopping...")
        manager.stop()
        exit(0)
    }
    termSource.resume()
    signal(SIGTERM, SIG_IGN)

    // Start capture in an async context
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            try await manager.start()
        } catch {
            log("Fatal: \(error)")
            exit(1)
        }
        semaphore.signal()
    }

    // Keep the main thread alive (RunLoop for signal handling)
    dispatchMain()
} else {
    log("Error: macOS 13.0+ required for ScreenCaptureKit")
    exit(1)
}

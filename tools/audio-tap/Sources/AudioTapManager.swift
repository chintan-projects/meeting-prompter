import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

/// Captures audio from a specific application via ScreenCaptureKit (macOS 13+).
///
/// Creates an SCStream targeting the given PID's application, captures audio
/// sample buffers, resamples to the target rate, and writes raw float32 PCM
/// to stdout in fixed-size chunks.
@available(macOS 13.0, *)
final class AudioTapManager: NSObject, SCStreamOutput, SCStreamDelegate {
    private let pid: pid_t
    private let targetSampleRate: Float64
    private let chunkDuration: Double

    private var stream: SCStream?
    private var running = false

    // Resampling
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?

    // Accumulation buffer for chunk-based output
    private let chunkSamples: Int
    private var accumulationBuffer: [Float32] = []
    private let bufferLock = NSLock()

    init(pid: pid_t, sampleRate: Float64 = 16000, chunkDuration: Double = 4.0) {
        self.pid = pid
        self.targetSampleRate = sampleRate
        self.chunkDuration = chunkDuration
        self.chunkSamples = Int(sampleRate * chunkDuration)
        super.init()
    }

    /// Start capturing. Blocks until stop() is called or an error occurs.
    func start() async throws {
        // Get shareable content to find the target app
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false
        )

        guard let targetApp = content.applications.first(
            where: { $0.processID == pid }
        ) else {
            throw CaptureError.appNotFound(pid)
        }

        guard let display = content.displays.first else {
            throw CaptureError.noDisplay
        }

        log("Target: \(targetApp.applicationName) (PID \(pid))")

        // Create a content filter for just this application
        let filter = SCContentFilter(
            display: display,
            including: [targetApp],
            exceptingWindows: []
        )

        // Configure for audio capture with minimal video overhead
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = Int(targetSampleRate <= 48000 ? 48000 : targetSampleRate)
        config.channelCount = 1
        config.excludesCurrentProcessAudio = true
        // We only need audio, but macOS 13 SCStream requires a video config.
        // Keep it minimal (small frame, 1 fps) — but stay clear of sub-minimum
        // dimensions that some macOS versions reject (which, with a nil delegate,
        // used to fail silently). 16x16 is negligible overhead and safely valid.
        config.width = 16
        config.height = 16
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1) // 1 fps

        // Set up resampler if capture rate differs from target
        let captureRate = Double(config.sampleRate)
        if let targetFmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) {
            targetFormat = targetFmt
            if captureRate != targetSampleRate {
                if let captureFmt = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32,
                    sampleRate: captureRate,
                    channels: 1,
                    interleaved: false
                ) {
                    converter = AVAudioConverter(from: captureFmt, to: targetFmt)
                    converter?.sampleRateConverterQuality = AVAudioQuality.medium.rawValue
                    log("Resampler: \(Int(captureRate))Hz -> \(Int(targetSampleRate))Hz")
                }
            }
        }

        // Create and start the stream
        let audioQueue = DispatchQueue(label: "audio-tap.audio", qos: .userInteractive)
        // Pass a delegate so stream-level failures (target app quit/crashed,
        // permission revoked mid-session, OS teardown) are observed instead of
        // leaving the process running blind. Previously delegate was nil.
        stream = SCStream(filter: filter, configuration: config, delegate: self)

        try stream?.addStreamOutput(self, type: .audio, sampleHandlerQueue: audioQueue)
        try await stream?.startCapture()

        running = true
        log("Capturing \(targetApp.applicationName) at \(Int(targetSampleRate))Hz mono...")

        // Block until stopped
        while running {
            try await Task.sleep(nanoseconds: 100_000_000) // 100ms
        }
    }

    /// Stop capturing and clean up.
    func stop() {
        running = false
        flushBuffer()

        Task {
            try? await stream?.stopCapture()
            stream = nil
        }

        log("Capture stopped")
    }

    // MARK: - SCStreamDelegate

    /// Called when the OS tears down the stream (target app quit/crashed,
    /// permission revoked, or an internal capture error). Without this, the
    /// process kept running blind. Log and exit non-zero so the Python parent
    /// detects the failed subprocess and surfaces it as capture_error rather
    /// than silently producing no "Others" audio.
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("Stream stopped with error: \(error.localizedDescription)")
        running = false
        exit(3)
    }

    // MARK: - SCStreamOutput delegate

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio else { return }
        guard let formatDesc = sampleBuffer.formatDescription else { return }

        let audioFormat = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc)
        guard let asbd = audioFormat?.pointee else { return }

        // Extract float32 samples from the CMSampleBuffer
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(
            blockBuffer, atOffset: 0, lengthAtOffsetOut: nil,
            totalLengthOut: &length, dataPointerOut: &dataPointer
        )
        guard status == noErr, let ptr = dataPointer, length > 0 else { return }

        let sampleCount = length / MemoryLayout<Float32>.size
        let floatPtr = UnsafeRawPointer(ptr).assumingMemoryBound(to: Float32.self)
        var samples = Array(UnsafeBufferPointer(start: floatPtr, count: sampleCount))

        // Handle multi-channel: mixdown to mono
        let channels = Int(asbd.mChannelsPerFrame)
        if channels > 1 {
            let frameCount = sampleCount / channels
            var mono = [Float32](repeating: 0, count: frameCount)
            for frame in 0 ..< frameCount {
                var sum: Float32 = 0
                for ch in 0 ..< channels {
                    sum += samples[frame * channels + ch]
                }
                mono[frame] = sum / Float32(channels)
            }
            samples = mono
        }

        // Resample if needed
        let outputSamples: [Float32]
        if let converter, let targetFormat {
            let sourceRate = asbd.mSampleRate
            if let sourceFmt = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sourceRate,
                channels: 1,
                interleaved: false
            ) {
                // Recreate converter if source format changed
                if converter.inputFormat.sampleRate != sourceRate {
                    self.converter = AVAudioConverter(from: sourceFmt, to: targetFormat)
                    self.converter?.sampleRateConverterQuality = AVAudioQuality.medium.rawValue
                }
            }

            let frameCount = samples.count
            guard frameCount > 0 else { return }

            guard let inputBuffer = AVAudioPCMBuffer(
                pcmFormat: converter.inputFormat,
                frameCapacity: AVAudioFrameCount(frameCount)
            ) else { return }
            inputBuffer.frameLength = AVAudioFrameCount(frameCount)
            if let channelData = inputBuffer.floatChannelData?[0] {
                for i in 0 ..< frameCount {
                    channelData[i] = samples[i]
                }
            }

            let ratio = targetSampleRate / converter.inputFormat.sampleRate
            let outFrameCount = AVAudioFrameCount(Double(frameCount) * ratio + 1)
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: outFrameCount
            ) else { return }

            var error: NSError?
            converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return inputBuffer
            }

            if error != nil { return }
            let outCount = Int(outputBuffer.frameLength)
            guard outCount > 0, let outData = outputBuffer.floatChannelData?[0] else { return }
            outputSamples = Array(UnsafeBufferPointer(start: outData, count: outCount))
        } else {
            outputSamples = samples
        }

        // Accumulate and emit full chunks
        bufferLock.lock()
        accumulationBuffer.append(contentsOf: outputSamples)

        while accumulationBuffer.count >= chunkSamples {
            let chunk = Array(accumulationBuffer.prefix(chunkSamples))
            accumulationBuffer.removeFirst(chunkSamples)
            bufferLock.unlock()
            writeChunkToStdout(chunk)
            bufferLock.lock()
        }
        bufferLock.unlock()
    }

    // MARK: - Output

    private func writeChunkToStdout(_ chunk: [Float32]) {
        chunk.withUnsafeBufferPointer { bufferPointer in
            let rawPointer = UnsafeRawBufferPointer(bufferPointer)
            let data = Data(rawPointer)
            FileHandle.standardOutput.write(data)
        }
    }

    private func flushBuffer() {
        bufferLock.lock()
        let remaining = accumulationBuffer
        accumulationBuffer.removeAll()
        bufferLock.unlock()

        if !remaining.isEmpty {
            writeChunkToStdout(remaining)
        }
    }

    private func log(_ message: String) {
        FileHandle.standardError.write(Data("[audio-tap] \(message)\n".utf8))
    }

    // MARK: - Permission check

    /// Check if screen recording permission is available by querying shareable content.
    static func checkPermission() async -> Bool {
        do {
            _ = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: false
            )
            return true
        } catch {
            return false
        }
    }

    enum CaptureError: Error, CustomStringConvertible {
        case appNotFound(pid_t)
        case noDisplay

        var description: String {
            switch self {
            case .appNotFound(let p):
                return "No running application found with PID \(p)"
            case .noDisplay:
                return "No display available for capture"
            }
        }
    }
}

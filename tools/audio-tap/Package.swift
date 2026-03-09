// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "audio-tap",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "audio-tap",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("AppKit"),
            ]
        ),
    ]
)

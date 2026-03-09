import AppKit
import Foundation

/// Lists running GUI applications suitable for audio capture.
struct AppLister {
    struct AppInfo: Encodable {
        let pid: Int32
        let name: String
        let bundleId: String

        enum CodingKeys: String, CodingKey {
            case pid
            case name
            case bundleId = "bundle_id"
        }
    }

    /// Returns JSON array of running GUI apps to stdout.
    static func listApps() {
        let workspace = NSWorkspace.shared
        var apps: [AppInfo] = []

        for app in workspace.runningApplications {
            // Only include regular (GUI) applications
            guard app.activationPolicy == .regular else { continue }
            guard let name = app.localizedName, !name.isEmpty else { continue }

            apps.append(AppInfo(
                pid: app.processIdentifier,
                name: name,
                bundleId: app.bundleIdentifier ?? ""
            ))
        }

        // Sort by name for stable output
        apps.sort { $0.name.lowercased() < $1.name.lowercased() }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(apps),
           let json = String(data: data, encoding: .utf8)
        {
            print(json)
        } else {
            print("[]")
        }
    }
}

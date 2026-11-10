// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CellularGrowth",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "CellularGrowth",
            path: "Sources/CellularGrowth"
        )
    ]
)

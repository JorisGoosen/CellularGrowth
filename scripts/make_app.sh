#!/bin/bash
# Assemble a double-clickable CellularGrowth.app from the release build.
set -euo pipefail
cd "$(dirname "$0")/.."

swift build -c release

APP="CellularGrowth.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp .build/release/CellularGrowth "$APP/Contents/MacOS/"

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>CellularGrowth</string>
    <key>CFBundleDisplayName</key><string>Cellular Growth</string>
    <key>CFBundleIdentifier</key><string>com.virtuoos.cellulargrowth</string>
    <key>CFBundleExecutable</key><string>CellularGrowth</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
EOF

codesign --force --sign - "$APP"
echo "Built $APP — start with: open $APP"

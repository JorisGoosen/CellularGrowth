//  AppDelegate.swift
//  Manual NSApplication lifecycle (keeps main.swift top-level code simple and
//  allows the headless runner to share the executable).

import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {

    var window: NSWindow?
    private var viewModel: SimViewModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let vm = SimViewModel()
        viewModel = vm

        let content = NSHostingView(rootView: ContentView(vm: vm))
        content.autoresizingMask = [.width, .height]

        let window = NSWindow(
            contentRect: NSRect(x: 80, y: 80, width: 1400, height: 880),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Cellular Growth — M1 (forces)"
        window.contentView = content
        window.center()
        window.makeKeyAndOrderFront(nil)
        self.window = window

        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

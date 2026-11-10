//  main.swift
//  Entry point: headless validation mode or the GUI app.

import AppKit

if HeadlessRunner.shouldRun() {
    HeadlessRunner.run()
} else {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate

    let menu = NSMenu()
    let appItem = NSMenuItem()
    menu.addItem(appItem)
    let appMenu = NSMenu()
    appMenu.addItem(NSMenuItem(title: "Quit CellularGrowth",
                               action: #selector(NSApplication.terminate(_:)),
                               keyEquivalent: "q"))
    appItem.submenu = appMenu
    app.mainMenu = menu

    app.setActivationPolicy(.regular)
    app.run()
}

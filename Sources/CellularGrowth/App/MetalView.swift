//  MetalView.swift
//  MTKView subclass carrying orbit/zoom input for the simulation view.

import Cocoa
import MetalKit

protocol ViewInput: AnyObject {
    func orbit(dx: Float, dy: Float)
    func zoom(factor: Float)
}

final class MetalView: MTKView {

    weak var controller: ViewInput?
    private var lastDragPoint: NSPoint?

    override func mouseDown(with event: NSEvent) {
        lastDragPoint = convert(event.locationInWindow, from: nil)
    }

    override func mouseDragged(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        if let last = lastDragPoint {
            let dx = Float(p.x - last.x)
            let dy = Float(p.y - last.y)
            controller?.orbit(dx: dx, dy: dy)
        }
        lastDragPoint = p
    }

    override func mouseUp(with event: NSEvent) {
        lastDragPoint = nil
    }

    override func scrollWheel(with event: NSEvent) {
        let dy = Float(event.scrollingDeltaY)
        controller?.zoom(factor: exp(-dy * 0.0022))
    }

    override func magnify(with event: NSEvent) {
        controller?.zoom(factor: 1 + Float(event.magnification))
    }
}

//  SimViewModel.swift
//  Owns the Metal device/queue, the simulation and the renderer; exposes
//  parameters and stats to SwiftUI. UI mutations happen on the main thread;
//  the display-link thread only reads plain vars and hops back to main for
//  stats publishing.

import Combine
import Foundation
import Metal
import MetalKit
import SwiftUI

final class SimViewModel: ObservableObject, ViewInput {

    let device: MTLDevice
    let queue: MTLCommandQueue
    let sim: CellSim
    let renderer: Renderer
    let metalView: MetalView

    // Simulation parameters (forwarded to the GPU params buffer on change).
    @Published var playing = true {
        didSet { renderer.isPlaying = playing }
    }
    @Published var stepsPerFrame: Double = 4 {
        didSet { renderer.stepsPerFrame = Int(stepsPerFrame) }
    }
    @Published var springFactor: Double = 0.33 {
        didSet { sim.params.springFactor = Float(springFactor); sim.syncParams() }
    }
    @Published var planarFactor: Double = 0.33 {
        didSet { sim.params.planarFactor = Float(planarFactor); sim.syncParams() }
    }
    @Published var bulgeFactor: Double = 0.33 {
        didSet { sim.params.bulgeFactor = Float(bulgeFactor); sim.syncParams() }
    }
    @Published var radiusScale: Double = 0.5 {
        didSet { sim.params.radiusScale = Float(radiusScale); sim.syncParams() }
    }
    @Published var foodIncrement: Double = 0.01 {
        didSet { sim.params.foodIncrement = Float(foodIncrement); sim.syncParams() }
    }
    @Published var splitThreshold: Double = 1.0 {
        didSet { sim.params.splitThreshold = Float(splitThreshold); sim.syncParams() }
    }
    @Published var divisionEnabled = true {
        didSet { sim.params.divisionEnabled = divisionEnabled ? 1 : 0; sim.syncParams() }
    }
    // --- M3 repulsion ---
    @Published var roi: Double = 1.6 {
        didSet { sim.params.roi = Float(roi); sim.syncParams() }
    }
    @Published var repulsionStrength: Double = 0.1 {
        didSet { sim.params.repulsionStrength = Float(repulsionStrength); sim.syncParams() }
    }
    // --- M5 growth modes ---
    enum GrowthMode: Int, CaseIterable, Identifiable {
        case uniform = 0
        case light = 1
        case rde = 2
        var id: Int { rawValue }
        var label: String { ["Uniform", "Light", "RDE"][rawValue] }
    }
    /// Switching modes takes effect immediately; entering RDE reseeds the
    /// chemicals (v = 0 everywhere is a fixed point — no pattern would form).
    @Published var growthMode: GrowthMode = .uniform {
        didSet {
            sim.params.growthMode = UInt32(growthMode.rawValue)
            sim.syncParams()
            if growthMode == .rde { sim.reseedChemicals() }
        }
    }
    @Published var lightDiffusionRate: Double = 0.02 {
        didSet { sim.params.lightDiffusionRate = Float(lightDiffusionRate); sim.syncParams() }
    }
    @Published var lightNutrient: Double = 0.05 {
        didSet { sim.params.lightNutrient = Float(lightNutrient); sim.syncParams() }
    }
    @Published var lightDecay: Double = 0.02 {
        didSet { sim.params.lightDecay = Float(lightDecay); sim.syncParams() }
    }
    @Published var lightBase: Double = 0.15 {
        didSet { sim.params.lightBase = Float(lightBase); sim.syncParams() }
    }
    @Published var lightAzimuth: Double = 54 { didSet { updateLightDir() } }    // degrees; cosmetic defaults matching the (0.55, 0.75, 0.4) shared light
    @Published var lightElevation: Double = 47.8 { didSet { updateLightDir() } } // degrees (didSet does not fire at init, so the SimParams default stays until touched)
    @Published var rdeF: Double = 0.037 {
        didSet { sim.params.rdeF = Float(rdeF); sim.syncParams() }
    }
    @Published var rdeK: Double = 0.06 {
        didSet { sim.params.rdeK = Float(rdeK); sim.syncParams() }
    }
    @Published var rdeDu: Double = 0.35 {
        didSet { sim.params.rdeDu = Float(rdeDu); sim.syncParams() }
    }
    @Published var rdeDv: Double = 0.15 {
        didSet { sim.params.rdeDv = Float(rdeDv); sim.syncParams() }
    }
    @Published var rdeBase: Double = 0.25 {
        didSet { sim.params.rdeBase = Float(rdeBase); sim.syncParams() }
    }
    @Published var rdeGain: Double = 2.5 {
        didSet { sim.params.rdeGain = Float(rdeGain); sim.syncParams() }
    }
    @Published var jitter: Double = 0.2          // applied on reset
    @Published var subdivision: Double = 3       // applied on reset

    // Live readouts.
    @Published var cellCount = 0
    @Published var meanLinkLength: Float = 0
    @Published var stepsPerSecond: Double = 0

    init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("CellularGrowth: no Metal device available")
        }
        self.device = device
        self.queue = queue
        sim = CellSim(device: device, queue: queue)
        renderer = Renderer(device: device, queue: queue, sim: sim)

        let view = MetalView(frame: NSRect(x: 0, y: 0, width: 1000, height: 700))
        metalView = view
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.065, green: 0.075, blue: 0.095, alpha: 1)
        view.clearDepth = 1.0
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60
        view.autoResizeDrawable = true
        view.delegate = renderer
        view.controller = self

        renderer.onStats = { [weak self] cells, mean, sps in
            DispatchQueue.main.async {
                self?.cellCount = cells
                self?.meanLinkLength = mean
                self?.stepsPerSecond = sps
            }
        }

        reset()
    }

    func reset() {
        var (positions, links) = Icosphere.generate(subdivisions: Int(subdivision),
                                                    spacing: sim.params.linkRestLength)
        let j = Float(jitter)
        if j > 0 {
            positions = positions.map { $0 + Self.randomInBall(radius: j) }
        }
        sim.params.growthMode = UInt32(growthMode.rawValue)
        sim.syncParams()
        sim.reset(positions: positions, links: links)
        cellCount = positions.count

        let ballRadius = positions.map { simd_length($0) }.max() ?? 1
        renderer.camera.target = .zero
        renderer.camera.distance = ballRadius * 2.6
    }

    /// Azimuth/elevation (degrees) -> unit light direction. Elevation 90 = straight up.
    private func updateLightDir() {
        let az = Float(lightAzimuth) * .pi / 180
        let el = Float(lightElevation) * .pi / 180
        let d = SIMD3<Float>(cos(el) * sin(az), sin(el), cos(el) * cos(az))
        sim.params.lightDirX = d.x
        sim.params.lightDirY = d.y
        sim.params.lightDirZ = d.z
        sim.syncParams()
    }

    // MARK: - ViewInput (orbit / zoom)

    func orbit(dx: Float, dy: Float) {
        renderer.camera.yaw -= dx * 0.006
        renderer.camera.pitch = min(max(renderer.camera.pitch + dy * 0.006, -1.5), 1.5)
    }

    func zoom(factor: Float) {
        renderer.camera.distance = min(max(renderer.camera.distance * factor, 0.5), 500)
    }

    // MARK: - Helpers

    static func randomInBall(radius: Float) -> SIMD3<Float> {
        while true {
            let v = SIMD3<Float>(Float.random(in: -1...1),
                                 Float.random(in: -1...1),
                                 Float.random(in: -1...1))
            if simd_length_squared(v) <= 1 { return v * radius }
        }
    }
}

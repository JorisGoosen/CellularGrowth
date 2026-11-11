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
    /// During a state load the reseed is suppressed (v3 files carry their
    /// chemicals; v1/v2 files are reseeded by the loader itself).
    @Published var growthMode: GrowthMode = .uniform {
        didSet {
            sim.params.growthMode = UInt32(growthMode.rawValue)
            sim.syncParams()
            if growthMode == .rde && !loadingState { sim.reseedChemicals() }
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
    @Published var lightInterval: Double = 8 {
        didSet { sim.params.lightInterval = UInt32(max(Int(lightInterval), 1)); sim.syncParams() }
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
    // --- V2.2 tint modes (cellTint kernel; feeds BOTH render paths and thus
    // the coloured X-ray) ---
    enum TintMode: Int, CaseIterable, Identifiable {
        case auto = 0       // chemical iff rde mode (the pre-V2.2 behaviour)
        case none = 1       // neutral albedo
        case chemical = 2   // rde u/v mapping in every growth mode
        case food = 3       // food/nutrient heatmap
        case pulse = 4      // magenta division-pulse flashes over the base tint
        var id: Int { rawValue }
        var label: String { ["Default", "None", "Chemical", "Food heat", "Mitosis pulse"][rawValue] }
    }
    @Published var tintMode: TintMode = .auto {
        didSet { sim.params.tintMode = UInt32(tintMode.rawValue); sim.syncParams() }
    }
    /// Food-heatmap normalization scale (0 = the shader-side auto default:
    /// light mode splitThreshold + lightNutrient/lightDecay, else
    /// splitThreshold).
    @Published var heatScale: Double = 0 {
        didSet { sim.params.heatScale = Float(max(heatScale, 0)); sim.syncParams() }
    }
    /// Mitosis pulse per-step decay multiplier (division writes 1.0; 0.95 ->
    /// half-life ~13.5 steps).
    @Published var pulseDecay: Double = 0.95 {
        didSet { sim.params.pulseDecay = Float(min(max(pulseDecay, 0), 0.9999)); sim.syncParams() }
    }
    @Published var jitter: Double = 0.2          // applied on reset
    @Published var subdivision: Double = 3       // applied on reset

    // --- M6 render style (Solid = historical path; X-ray = additive impostor
    // density accumulation — the interactive approximation of the offline
    // X-ray; see cellFragmentXray) ---
    @Published var renderStyle: Renderer.RenderStyle = .solid {
        didSet { renderer.renderStyle = renderStyle }
    }
    /// Interactive X-ray exposure gain (per-sprite premultiplier; the 8-bit
    /// additive clamp is the tone map).
    @Published var xrayGain: Double = 0.3 {
        didSet { renderer.xrayGain = Float(xrayGain) }
    }
    /// V2.1 edge-layer strength (solid impostor path; 0 = off). Same value
    /// the ray tracer's --edges uses — the wall blend is a render parameter,
    /// not a style.
    @Published var edges: Double = 0 {
        didSet { renderer.edgeStrength = Float(edges) }
    }
    // --- M6 presets (Core/Presets.swift) ---
    @Published var presetName: String? {
        didSet {
            guard !applyingPreset, let name = presetName,
                  let preset = SimPreset.named(name) else { return }
            applyPreset(preset)
        }
    }
    private var applyingPreset = false
    /// True while a state file loads: the published vars are being synced
    /// from the loaded SimParams, and side effects tied to `didSet` (RDE
    /// chemical reseeding) must NOT run — a v3 file carries its chemicals.
    private var loadingState = false
    // --- M6 state save/load ---
    @Published var statusMessage: String?

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

    // MARK: - M6 presets

    /// Apply a named preset: full param bundle + reset-time shape knobs,
    /// then reset. The panel is synced first (each didSet pushes the value
    /// into SimParams), so sliders always show what runs.
    func applyPreset(_ preset: SimPreset) {
        applyingPreset = true
        defer { applyingPreset = false }
        let p = preset.params
        springFactor = Double(p.springFactor)
        planarFactor = Double(p.planarFactor)
        bulgeFactor = Double(p.bulgeFactor)
        radiusScale = Double(p.radiusScale)
        foodIncrement = Double(p.foodIncrement)
        splitThreshold = Double(p.splitThreshold)
        divisionEnabled = p.divisionEnabled != 0
        roi = Double(p.roi)
        repulsionStrength = Double(p.repulsionStrength)
        growthMode = GrowthMode(rawValue: Int(p.growthMode)) ?? .uniform
        lightDiffusionRate = Double(p.lightDiffusionRate)
        lightNutrient = Double(p.lightNutrient)
        lightDecay = Double(p.lightDecay)
        lightBase = Double(p.lightBase)
        lightInterval = Double(p.lightInterval)
        setLightDirFromParams(x: p.lightDirX, y: p.lightDirY, z: p.lightDirZ)
        rdeF = Double(p.rdeF)
        rdeK = Double(p.rdeK)
        rdeDu = Double(p.rdeDu)
        rdeDv = Double(p.rdeDv)
        rdeBase = Double(p.rdeBase)
        rdeGain = Double(p.rdeGain)
        subdivision = Double(preset.subdivision)
        jitter = Double(preset.jitter)
        reset()
        statusMessage = "Preset '\(preset.name)' applied"
    }

    /// Convert a stored light direction into the azimuth/elevation slider
    /// pair (inverse of updateLightDir).
    private func setLightDirFromParams(x: Float, y: Float, z: Float) {
        let len = max(simd_length(SIMD3<Float>(x, y, z)), 1e-6)
        lightElevation = Double(asin(y / len)) * 180 / .pi
        lightAzimuth = Double(atan2(x, z)) * 180 / .pi
    }

    // MARK: - M6 state save/load (CGST v3)

    /// Save the full sim state via NSSavePanel (default directory:
    /// ~/Documents — the standard user-visible location; the CLI
    /// --state-save takes explicit paths instead).
    func saveStatePanel() {
        let panel = NSSavePanel()
        panel.title = "Save simulation state (CGST v3)"
        panel.nameFieldStringValue = "cellular-state.cgst"
        panel.directoryURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        panel.allowedContentTypes = [.data]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try sim.saveState(to: url.path)
            statusMessage = "State saved (\(sim.cellCount) cells): \(url.lastPathComponent)"
        } catch {
            statusMessage = "State save FAILED: \(error.localizedDescription)"
        }
    }

    /// Load a state snapshot (v1/v2/v3) via NSOpenPanel and sync the panel
    /// to the loaded params. v3 restores chemicals/food/RNG exactly; v1/v2
    /// reseed them (historical semantics).
    func loadStatePanel() {
        let panel = NSOpenPanel()
        panel.title = "Load simulation state (CGST)"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        panel.allowedContentTypes = [.data]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let snap = try sim.loadState(from: url.path)
            refreshPublishedFromParams()
            let modes = ["uniform", "light", "rde"]
            statusMessage = "State loaded (v\(snap.version), \(sim.cellCount) cells, mode \(modes[Int(min(snap.growthMode, 2))]))"
        } catch {
            statusMessage = "State load FAILED: \(error.localizedDescription)"
        }
    }

    /// Sync every published parameter from sim.params after a load (the
    /// loaded values must SHOW on the panel). Side effects suppressed while
    /// `loadingState` is up.
    private func refreshPublishedFromParams() {
        loadingState = true
        defer { loadingState = false }
        let p = sim.params
        growthMode = GrowthMode(rawValue: Int(p.growthMode)) ?? .uniform
        springFactor = Double(p.springFactor)
        planarFactor = Double(p.planarFactor)
        bulgeFactor = Double(p.bulgeFactor)
        radiusScale = Double(p.radiusScale)
        foodIncrement = Double(p.foodIncrement)
        splitThreshold = Double(p.splitThreshold)
        divisionEnabled = p.divisionEnabled != 0
        roi = Double(p.roi)
        repulsionStrength = Double(p.repulsionStrength)
        lightDiffusionRate = Double(p.lightDiffusionRate)
        lightNutrient = Double(p.lightNutrient)
        lightDecay = Double(p.lightDecay)
        lightBase = Double(p.lightBase)
        lightInterval = Double(p.lightInterval)
        setLightDirFromParams(x: p.lightDirX, y: p.lightDirY, z: p.lightDirZ)
        rdeF = Double(p.rdeF)
        rdeK = Double(p.rdeK)
        rdeDu = Double(p.rdeDu)
        rdeDv = Double(p.rdeDv)
        rdeBase = Double(p.rdeBase)
        rdeGain = Double(p.rdeGain)
        // V2.2 tint knobs (a loaded state carries them in its raw params).
        tintMode = TintMode(rawValue: Int(p.tintMode)) ?? .auto
        heatScale = Double(p.heatScale)
        pulseDecay = Double(p.pulseDecay)
        cellCount = sim.cellCount
        // Re-frame the camera on the loaded structure.
        let ballRadius = sim.readPositions().map { simd_length($0) }.max() ?? 1
        renderer.camera.target = .zero
        renderer.camera.distance = ballRadius * 2.6
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

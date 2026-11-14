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
    /// V2.7: weak back-reference set by AppDelegate — the export start
    /// resizes the window to fit the export aspect ratio (4:3 + panel).
    weak var owningWindow: NSWindow?

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
    // --- M5 growth modes / V2.8 composable drives ---
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
    /// V2.8: the mode picker is the PRESET-SETTER for the three additive
    /// drive sliders below (uniform -> U=1, light -> L=1, rde -> R=1) —
    /// setGrowthMode applies the preset and the slider bindings are synced
    /// to show it. Dragging a drive slider afterwards turns the effective
    /// configuration into a free combination (the mode label stays, the
    /// drives rule — the panel caption spells this out).
    @Published var growthMode: GrowthMode = .uniform {
        didSet {
            sim.setGrowthMode(UInt32(growthMode.rawValue))
            sim.syncParams()
            // Show the preset on the drive sliders (their didSets re-push the
            // same values — idempotent).
            growthUniform = Double(sim.params.growthUniform)
            growthLight = Double(sim.params.growthLight)
            growthRde = Double(sim.params.growthRde)
            if growthMode == .rde && !loadingState { sim.reseedChemicals() }
        }
    }
    // --- V2.8 additive growth drives (each 0 = off; 1.0 = the legacy
    // single-drive level). Shown next to their parameter groups in the panel
    // (the uniform drive with the food knobs, the light drive at the top of
    // the light group, the RDE drive at the top of the RDE group) — a drive
    // at 0 visually demotes (disables) its group's parameters.
    @Published var growthUniform: Double = 1 {
        didSet { sim.params.growthUniform = Float(max(growthUniform, 0)); sim.syncParams() }
    }
    @Published var growthLight: Double = 0 {
        didSet { sim.params.growthLight = Float(max(growthLight, 0)); sim.syncParams() }
    }
    @Published var growthRde: Double = 0 {
        didSet { sim.params.growthRde = Float(max(growthRde, 0)); sim.syncParams() }
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
    // --- V2.2/V2.3 tint modes (cellTint kernel; feeds BOTH render paths and
    // thus the coloured X-ray) ---
    enum TintMode: Int, CaseIterable, Identifiable {
        case auto = 0       // chemical iff rde mode (the pre-V2.2 behaviour)
        case none = 1       // neutral albedo
        case chemical = 2   // rde u/v mapping in every growth mode
        case food = 3       // food/nutrient heatmap
        case pulse = 4      // magenta division-pulse flashes over the base tint
        case type = 5       // V2.3 type colours (stem/lamel/dal/cap)
        case typePulse = 6  // V2.3 type colours + the mitosis-pulse overlay
        case starve = 7     // V2.5 the starvation-death front (red = starving)
        var id: Int { rawValue }
        var label: String {
            ["Default", "None", "Chemical", "Food heat", "Mitosis pulse",
             "Type", "Type + pulse", "Starving"][rawValue]
        }
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

    // --- V2.3 type layout (phase-0 polar typing + the two-phase switch) ---
    /// Type axis components (normalized at apply). Dragging any typing knob
    /// re-types the LIVE cells immediately (the GUI "re-type command");
    /// reset() re-types the fresh ball when typing is on.
    @Published var typingEnabled = false
    @Published var typeAxisX: Double = 0 { didSet { applyTypingLive() } }
    @Published var typeAxisY: Double = 1 { didSet { applyTypingLive() } }
    @Published var typeAxisZ: Double = 0 { didSet { applyTypingLive() } }
    /// s < typeT1 -> stem.
    @Published var typeT1: Double = -0.35 { didSet { applyTypingLive() } }
    /// s > typeT2 -> cap; the lamel/dal band sits between.
    @Published var typeT2: Double = 0.30 { didSet { applyTypingLive() } }
    /// Phase-2 switch (executed divisions; 0 = never).
    @Published var phaseX: Double = 0 {
        didSet { sim.params.phaseSwitchX = UInt32(max(Int(phaseX), 0)); sim.syncParams() }
    }
    /// Phase-2 band growth rates (see foodStep in Shaders.swift).
    @Published var lamelRate: Double = 1.0 {
        didSet { sim.params.lamelRate = Float(max(lamelRate, 0)); sim.syncParams() }
    }
    @Published var dalRate: Double = 0.5 {
        didSet { sim.params.dalRate = Float(max(dalRate, 0)); sim.syncParams() }
    }

    // --- V2.4 differentiation phase (RDE-driven typing with hysteresis) ---
    /// The differentiation phase: division frozen, the RDE keeps stepping,
    /// typeTransition converts type-0 cells from the chemical field every
    /// step (v >= typeCapV -> cap, v <= typeStemV -> stem; typed cells then
    /// ignore the RDE — type-memory hysteresis). Toggled by the
    /// "Differentiate now" / "Resume growth" button; takes effect from the
    /// next rendered frame (the phase flag is read per encoded step).
    @Published var differentiationPhase = false {
        didSet { sim.params.differentiationPhase = differentiationPhase ? 1 : 0; sim.syncParams() }
    }
    /// typeTransition thresholds (stem must stay below cap).
    @Published var typeCapV: Double = 0.30 {
        didSet { sim.params.typeCapV = Float(typeCapV); syncDiffThresholds() }
    }
    @Published var typeStemV: Double = 0.10 {
        didSet { sim.params.typeStemV = Float(typeStemV); syncDiffThresholds() }
    }
    /// Keep stem < cap (the kernel's dead band must be non-empty).
    private func syncDiffThresholds() {
        let cap = max(sim.params.typeCapV, sim.params.typeStemV + 0.01)
        sim.params.typeCapV = cap
        sim.syncParams()
    }
    /// The button: entering the phase picks the rule-matrix mode — polar
    /// typing on -> band (the V2.3 matrix governs the polar types), otherwise
    /// region (RDE typing; every region divides uniformly). Leaving the phase
    /// just resumes growth.
    func toggleDifferentiation() {
        if !differentiationPhase {
            sim.params.typeMode = typingEnabled ? 0 : 1
            sim.syncParams()
        }
        differentiationPhase.toggle()
        statusMessage = differentiationPhase
            ? (sim.params.growthMode == 2
               ? "Differentiation phase: division frozen, RDE typing active (mode \(typingEnabled ? "band" : "region"))"
               : "Differentiation phase: division frozen (non-RDE mode: growth pause only)")
            : "Growth resumed"
    }

    // --- V2.7 death knobs (the V2.5 open issue "GUI death sliders": death
    // parameters were CLI/preset-side only) — LIVE SimParams, like every
    // other growth knob. The window is the pruning aggressiveness (a
    // growth-rate detector: see starvationStep in Shaders.swift — fed cells
    // re-cross the threshold in ~5-10 steps at the default gate values,
    // shaded cells need hundreds).
    @Published var deathEnabled = false {
        didSet { sim.params.deathEnabled = deathEnabled ? 1 : 0; sim.syncParams() }
    }
    /// Starvation threshold on the accumulated food ledger. Default 1.0 =
    /// splitThreshold (a cell starves while it cannot reach its own division
    /// threshold fast enough) — but the two sliders are INDEPENDENT knobs:
    /// raising the split threshold does NOT move the death threshold.
    @Published var deathThreshold: Double = 1.0 {
        didSet { sim.params.deathThreshold = Float(deathThreshold); sim.syncParams() }
    }
    /// Consecutive starving steps before death; 0 = never (death disabled
    /// in effect even with the toggle on — the toggle is the explicit
    /// master switch, the window 0 is the tuning limit).
    @Published var deathSteps: Double = 250 {
        didSet { sim.params.deathSteps = UInt32(max(Int(deathSteps), 0)); sim.syncParams() }
    }

    /// V2.3: apply the current type layout to the live cells (the Re-type
    /// button; also fired by the typing sliders). Re-bands the CURRENT
    /// geometry — after growth the centroid/shape have moved, so this is a
    /// re-typing of the grown form, not the original ball.
    func retypeCells() {
        let axis = SIMD3<Float>(Float(typeAxisX), Float(typeAxisY), Float(typeAxisZ))
        guard simd_length(axis) > 1e-6 else {
            statusMessage = "Type axis must be non-zero"
            return
        }
        sim.applyPolarTypes(axis: axis, t1: Float(typeT1), t2: Float(typeT2))
        statusMessage = String(format: "Types applied (t1 %.2f, t2 %.2f)", typeT1, typeT2)
    }

    /// Live re-type on slider drags — suppressed while a preset is being
    /// applied (its slider writes would re-type the PREVIOUS geometry) and
    /// while a state loads (types come from the file, not the sliders).
    private func applyTypingLive() {
        guard typingEnabled, !applyingPreset, !loadingState else { return }
        retypeCells()
    }

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
    /// V2.1b foam composite opacity (the foam impostor path). 0.5 makes the
    /// two-layer blend order-insensitive; the back-to-front instance sort
    /// handles deeper stacks.
    @Published var foamOpacity: Double = 0.5 {
        didSet { renderer.foamOpacity = Float(foamOpacity) }
    }
    // --- V2.6 camera choreography (host-side viewer state) ---
    // GUI defaults: follow-centroid ON, circle OFF. The HEADLESS paths keep
    // the legacy fixed-target camera (the goldens cover those paths; the GUI
    // is not golden-covered) — the asymmetry is documented, deliberate.
    @Published var followCentroid = true {
        didSet { renderer.followCentroid = followCentroid }
    }
    /// Follow-target exponential smoothing per rendered frame (0 = snap).
    @Published var followSmoothing: Double = 0.15 {
        didSet { renderer.followSmoothing = Float(followSmoothing) }
    }
    /// Circle mode: yaw advances `orbitRate` rad EVERY rendered frame
    /// (deterministic per-frame rate — export frames are equally spaced in
    /// steps; at the GUI's ~60 fps 0.005 ≈ one full turn per 21 s).
    @Published var orbitEnabled = false {
        didSet { renderer.orbiting = orbitEnabled }
    }
    @Published var orbitRate: Double = 0.005 {
        didSet { renderer.orbitRate = Float(orbitRate) }
    }
    // --- V2.6 video export (shares HeadlessRunner.exportAnimation with the
    // CLI --anim path — ONE deterministic offscreen implementation) ---
    // V2.7: frames slider goes to 10000 (stoppability + the live preview
    // make long exports workable — no other artificial limits), the
    // steps/frame cap is raised 600 -> 2000 (sized for the blocking-export
    // era before V2.7; the practical limit now is patience, not safety).
    @Published var exportFrames: Double = 60
    @Published var exportStepsPerFrame: Double = 100
    // --- V2.7 non-blocking export state (the export runs on a CLONED sim
    // with its own command queue — the GUI keeps stepping/rendering; GPU
    // contention may lower the GUI fps, accepted and documented) ---
    @Published var exportRunning = false
    /// Most recent completed export frame, downscaled for the panel preview.
    @Published var exportPreviewImage: NSImage?
    /// "frame 12/300 — elapsed 0:45" (frame-cadence updates + a 0.5 s timer).
    @Published var exportProgressText: String?
    /// V2.7: true between the Export click and the clone's completion — the
    /// one-off window in which the GUI sim is frozen (playing = false) and
    /// its buffers are being memcpy'd to the clone. Play/Reset/Export are
    /// disabled during it: a step or reset enqueued mid-clone would race
    /// GPU writes against the clone's reads. The window is sub-second
    /// (~0.5 s drain + the one-off clone).
    @Published var exportSetupActive = false
    /// Finished movie path (nil while running / on failure) — "Show in
    /// Finder" reveals it, or the frames directory when the encode failed.
    @Published var exportResultPath: String?
    /// Frames directory (set while running; the "Show in Finder" fallback).
    @Published var exportFramesDir: String?
    /// Stop flag: written on the main thread (Stop button), read on the
    /// export thread between frames — the codebase's plain-var cross-thread
    /// pattern; worst case one extra rendered frame.
    private var exportStopRequested = false
    private var exportDoneFrames = 0
    private var exportTotalFrames = 0
    private var exportStartDate = Date()
    private var exportProgressTimer: Timer?
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

        // V2.6: the GUI follows the growing form by default (the @Published
        // defaults above; didSet does not fire at init, so the renderer is
        // synced explicitly once here).
        renderer.followCentroid = followCentroid
        renderer.followSmoothing = Float(followSmoothing)
        renderer.orbiting = orbitEnabled
        renderer.orbitRate = Float(orbitRate)

        // V2.8: apply the mode's drive preset once at startup — the @Published
        // drive-slider defaults above (U=1) must match what the sim runs
        // (the SimParams struct defaults are all-zero = no drive), and didSet
        // does not fire at init (same pattern as the choreography sync).
        sim.setGrowthMode(UInt32(growthMode.rawValue))

        reset()
    }

    func reset() {
        // V2.4: a fresh ball must not stay frozen — clear the differentiation
        // phase (the flag is a SimParams field, not reset()-cleared state).
        differentiationPhase = false
        var (positions, links) = Icosphere.generate(subdivisions: Int(subdivision),
                                                    spacing: sim.params.linkRestLength)
        let j = Float(jitter)
        if j > 0 {
            positions = positions.map { $0 + Self.randomInBall(radius: j) }
        }
        sim.params.growthMode = UInt32(growthMode.rawValue)
        sim.syncParams()
        sim.reset(positions: positions, links: links)
        // V2.3: re-type the fresh ball when typing is on (reset zeroed the
        // types — typing is reset-time state).
        if typingEnabled { retypeCells() }
        cellCount = positions.count

        let ballRadius = positions.map { simd_length($0) }.max() ?? 1
        renderer.camera.target = .zero
        renderer.camera.distance = ballRadius * 2.6
        // V2.6: clear the follow smoothing — the next frame snaps to the
        // fresh ball's centroid (which is ~the origin here anyway).
        renderer.resetChoreography()
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
        // V2.3: two-phase growth rates (SimParams) + the polar typing knobs
        // (reset-time). Non-typed presets switch typing OFF (a previously
        // selected typed preset must not re-type the next reset).
        phaseX = Double(p.phaseSwitchX)
        lamelRate = Double(p.lamelRate)
        dalRate = Double(p.dalRate)
        // V2.7: presets can carry the death knobs (prune-demo does). Synced
        // like every other param — the panel shows what runs; explicit user
        // input wins AFTER the load (the sliders push straight into
        // SimParams until the next preset/state load).
        deathEnabled = p.deathEnabled != 0
        deathThreshold = Double(p.deathThreshold)
        deathSteps = Double(p.deathSteps)
        // V2.4: a preset switch leaves the differentiation phase (presets do
        // not carry it).
        differentiationPhase = false
        typingEnabled = (preset.typeAxis != nil)
        if let ax = preset.typeAxis {
            typeAxisX = Double(ax.x)
            typeAxisY = Double(ax.y)
            typeAxisZ = Double(ax.z)
            typeT1 = Double(preset.typeT1)
            typeT2 = Double(preset.typeT2)
        }
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
        // V2.8: the loaded drives (the growthMode didSet above applied the
        // preset first; a full v3 params block carries its own drives, a
        // pre-V2.8 block was mapped by the loader — both end up here).
        growthUniform = Double(p.growthUniform)
        growthLight = Double(p.growthLight)
        growthRde = Double(p.growthRde)
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
        // V2.3 two-phase growth params (the loaded state's own values).
        phaseX = Double(p.phaseSwitchX)
        lamelRate = Double(p.lamelRate)
        dalRate = Double(p.dalRate)
        // V2.4 differentiation state (stem first — the cap didSet clamps
        // against the CURRENT stem; a saved mid-differentiation state resumes
        // in the phase).
        typeStemV = Double(p.typeStemV)
        typeCapV = Double(p.typeCapV)
        differentiationPhase = (p.differentiationPhase != 0)
        // V2.7: the death knobs sync from the loaded raw params too (v3
        // files carry them from V2.5 on; legacy files load with the struct
        // defaults — death off).
        deathEnabled = p.deathEnabled != 0
        deathThreshold = Double(p.deathThreshold)
        deathSteps = Double(p.deathSteps)
        cellCount = sim.cellCount
        // Re-frame the camera on the loaded structure.
        let ballRadius = sim.readPositions().map { simd_length($0) }.max() ?? 1
        renderer.camera.target = .zero
        renderer.camera.distance = ballRadius * 2.6
        // V2.6: clear the follow smoothing — the next frame snaps to the
        // loaded structure's live centroid.
        renderer.resetChoreography()
    }

    // MARK: - V2.7 video export (GUI, NON-BLOCKING on a cloned sim)

    /// GUI "Export video…": pick a directory, then run the SAME
    /// deterministic offscreen export the CLI --anim uses
    /// (HeadlessRunner.exportAnimation — factored, not duplicated) — but on
    /// a CLONE of the live state (its own CellSim + its own MTLCommandQueue,
    /// never the GUI's queue), while the GUI keeps stepping and rendering.
    ///
    /// V2.7 protocol (replaces the V2.6 blocking export, which paused the
    /// display link and owned the live sim for its whole duration):
    ///   1. main: playing = false — no further sim steps are encoded (the
    ///      view keeps drawing; draw-only frames write NO sim buffers).
    ///   2. main: +0.5 s drain — any frame already in flight when playing
    ///      went false commits its command buffer within ~1–2 frames (a
    ///      full GUI frame costs ≤ ~45 ms GPU at 2M cells, benchlog V2.5
    ///      extrapolation); 0.5 s is ~10× that margin. After it the sim
    ///      buffers are quiescent.
    ///   3. background: build the export sim (CellSim init compiles the
    ///      Metal library — one-time cost) + cloneState(from: live sim) —
    ///      CPU memcpy of the live prefixes; the only concurrent traffic is
    ///      draw-only frames READING the same buffers (read/read = safe).
    ///   4. main: playing restored, exportRunning = true, window resized to
    ///      the export aspect ratio; the frame loop runs on the background
    ///      thread on the clone.
    ///   5. Stop sets exportStopRequested; it is checked between frames and
    ///      whatever rendered is still encoded (partial export — the user
    ///      always gets a video).
    /// GPU contention between the GUI sim and the export's ray renderer
    /// lowers the GUI fps for the export's duration — accepted (documented
    /// milestone choice). The preview shows the export's OFFSCREEN frames,
    /// not the GUI render.
    func exportVideoPanel() {
        guard !exportRunning, !exportSetupActive else { return }
        let panel = NSOpenPanel()
        panel.title = "Export video (PNG frames + H.264 .mp4)"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.directoryURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        guard panel.runModal() == .OK, let dir = panel.url?.path else { return }

        let frames = max(1, Int(exportFrames))
        let steps = max(0, Int(exportStepsPerFrame))
        let wasPlaying = playing
        playing = false
        exportSetupActive = true
        statusMessage = "Cloning the live sim state for the export…"
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self else { return }
            // The export's OWN command queue — never the GUI sim's queue
            // (two queues alongside each other are fine in Metal).
            guard let exportQueue = self.device.makeCommandQueue() else {
                self.failExport(NSError(domain: "export", code: 1,
                                        userInfo: [NSLocalizedDescriptionKey: "could not create the export command queue"]))
                return
            }
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else { return }
                do {
                    // One-time clone cost: CellSim init (Metal library
                    // compile) + the O(live) buffer-copy pass; timed for
                    // the benchlog.
                    let tClone = CFAbsoluteTimeGetCurrent()
                    let exportSim = CellSim(device: self.device, queue: exportQueue,
                                            maxCells: self.exportCapacityEstimate(frames: frames, stepsPerFrame: steps))
                    exportSim.cloneState(from: self.sim)
                    let cloneMs = (CFAbsoluteTimeGetCurrent() - tClone) * 1000
                    print("video export: clone ready (\(exportSim.liveCount) live cells, maxCells \(exportSim.maxCells)) in \(String(format: "%.0f", cloneMs)) ms")
                    // The clone holds the state — the GUI resumes NOW; the
                    // export never touches the live sim again.
                    DispatchQueue.main.async {
                        self.playing = wasPlaying
                        self.exportSetupActive = false
                        self.exportRunning = true
                        self.exportStopRequested = false
                        self.exportFramesDir = dir
                        self.exportResultPath = nil
                        self.exportDoneFrames = 0
                        self.exportTotalFrames = frames
                        self.exportStartDate = Date()
                        self.refreshExportProgress()
                        self.startExportProgressTimer()
                        self.resizeWindowForExport()
                        self.statusMessage = "Exporting \(frames) frames × \(steps) steps on a clone — the sim keeps running; Stop finishes a partial video"
                    }

                    var cfg = HeadlessRunner.VideoExportConfig(
                        dir: dir, frames: frames, stepsPerFrame: steps,
                        settings: self.offlineExportSettings())
                    cfg.followCentroid = self.renderer.followCentroid
                    cfg.followSmoothing = self.renderer.followSmoothing
                    cfg.orbitRate = self.renderer.orbiting ? self.renderer.orbitRate : 0
                    // Progress signal (plain-var read, the codebase's relaxed
                    // cross-thread pattern, same as the V2.6 onFrame hook).
                    cfg.onFrame = { done, total in
                        DispatchQueue.main.async {
                            self.exportDoneFrames = done
                            self.exportTotalFrames = total
                            self.refreshExportProgress()
                        }
                    }
                    // Live preview: decode + downscale OFF-main (pure
                    // CoreGraphics), the main thread only swaps the image.
                    cfg.onPreview = { path, done, total in
                        if let img = Self.previewImage(fromPNG: path) {
                            DispatchQueue.main.async {
                                self.exportPreviewImage = img
                                self.exportDoneFrames = done
                                self.exportTotalFrames = total
                                self.refreshExportProgress()
                            }
                        }
                    }
                    cfg.shouldStop = { self.exportStopRequested }
                    let result = try HeadlessRunner.exportAnimation(queue: exportQueue, sim: exportSim, config: cfg)
                    DispatchQueue.main.async { self.finishExport(result: result, requested: frames) }
                } catch {
                    DispatchQueue.main.async { self.failExport(error) }
                }
            }
        }
    }

    /// Stop button: the flag is polled between frames (a frame in flight
    /// always completes); the movie then assembles from everything rendered
    /// so far — a stoppable export ALWAYS yields a video, and the status
    /// line marks it as a partial export.
    func stopExport() {
        guard exportRunning else { return }
        exportStopRequested = true
        statusMessage = "Stop requested — finishing the current frame, then encoding the partial video…"
    }

    private func finishExport(result: HeadlessRunner.VideoExportResult, requested: Int) {
        exportRunning = false
        exportSetupActive = false
        stopExportProgressTimer()
        exportResultPath = result.moviePath
        if result.frames == 0 {
            statusMessage = "Export stopped before the first frame completed — nothing rendered, no video."
        } else if let mp = result.moviePath {
            statusMessage = (result.stopped
                ? "PARTIAL export (stopped at \(result.frames)/\(requested) frames): \(mp)"
                : "Video exported: \(mp) (\(result.frames) frames)")
                + (result.capacityCapped
                   ? " — export sim hit its cell capacity (growth stalled from there on)" : "")
        } else {
            statusMessage = "Export: \(result.frames) PNG frames in \(result.dir) — MP4 assembly FAILED (see console)"
        }
    }

    private func failExport(_ error: Error) {
        exportRunning = false
        exportSetupActive = false
        stopExportProgressTimer()
        statusMessage = "Video export FAILED: \(error.localizedDescription)"
    }

    /// "Show in Finder": the finished movie, or the frames directory when
    /// the encode failed / the export was stopped pre-movie.
    func revealExportResult() {
        let urls: [URL] = [exportResultPath, exportFramesDir]
            .compactMap { $0 }.map { URL(fileURLWithPath: $0) }
        guard !urls.isEmpty else { return }
        NSWorkspace.shared.activateFileViewerSelecting(urls)
    }

    // MARK: V2.7 export progress + preview plumbing

    private func startExportProgressTimer() {
        exportProgressTimer?.invalidate()
        let t = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.refreshExportProgress()
        }
        RunLoop.main.add(t, forMode: .common)   // .common: keeps ticking during modal panels
        exportProgressTimer = t
    }

    private func stopExportProgressTimer() {
        exportProgressTimer?.invalidate()
        exportProgressTimer = nil
    }

    private func refreshExportProgress() {
        guard exportRunning else { return }
        let e = Date().timeIntervalSince(exportStartDate)
        exportProgressText = String(format: "frame %d/%d — elapsed %d:%02d",
                                    exportDoneFrames, exportTotalFrames,
                                    Int(e) / 60, Int(e) % 60)
    }

    /// Decode + downscale a finished frame PNG for the panel preview (pure
    /// CoreGraphics — thread-safe OFF-main; called on the export thread so
    /// the 1600×1200 decode never touches the GUI run loop). Long side
    /// bounded at 480 px (the panel shows it at ~266 pt — retina-sharp).
    private static func previewImage(fromPNG path: String, maxPixels: CGFloat = 480) -> NSImage? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let w = CGFloat(img.width), h = CGFloat(img.height)
        let scale = min(maxPixels / w, maxPixels / h, 1)
        let pw = max(1, Int(w * scale)), ph = max(1, Int(h * scale))
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: pw, height: ph, bitsPerComponent: 8,
                                  bytesPerRow: pw * 4, space: space,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue) else { return nil }
        ctx.interpolationQuality = .medium
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: pw, height: ph))
        guard let out = ctx.makeImage() else { return nil }
        return NSImage(cgImage: out, size: NSSize(width: pw, height: ph))
    }

    /// V2.7: the export clone's buffer capacity. Allocating the GUI sim's
    /// full 2M would cost a second ~400 MB buffer set per export; size it
    /// from the EXPECTED growth instead. Per-step per-capita division rate
    /// ≈ drive / (2 × splitThreshold): a cell's food ledger climbs by
    /// drive·U[0,1) per step (mean drive/2 — uniform mode), so it re-crosses
    /// the threshold every 2T/drive steps. Light/RDE modes fold their drive
    /// into the same form (lightNutrient is the per-step MEAN — the
    /// lightInterval march pulses it ×interval; RDE spots sit at v ≈ 0.4 —
    /// the SimParams comment). Compounded over the export's total steps,
    /// ×2 safety margin, clamped to [live + slack, the GUI sim's own
    /// maxCells] — at the clamp the export behaves exactly like the GUI
    /// would. If growth still overshoots, the export sim stalls at its
    /// capacity (division is a device-side hard cap) and the status line
    /// reports it (VideoExportResult.capacityCapped) — never a crash.
    /// Footprint note: at the 2M clamp the clone costs ~400 MB of
    /// storageModeShared buffers on top of the GUI sim's — fine on the
    /// 16 GB target.
    private func exportCapacityEstimate(frames: Int, stepsPerFrame: Int) -> Int {
        let live = max(sim.liveCount, 1)
        let totalSteps = max(frames * max(stepsPerFrame, 0), 1)
        // No division (or frozen mid-differentiation): the clone only needs
        // the live state (deaths never grow the allocation high-water mark).
        guard sim.params.divisionEnabled != 0, sim.params.differentiationPhase == 0 else {
            return live + 64
        }
        let drive: Float
        switch GrowthMode(rawValue: Int(sim.params.growthMode)) ?? .uniform {
        case .light:   drive = sim.params.lightBase + sim.params.lightNutrient
        case .rde:     drive = sim.params.rdeBase + sim.params.rdeGain * 0.4
        case .uniform: drive = sim.params.foodIncrement
        }
        let rate = Double(drive) / (2.0 * Double(max(sim.params.splitThreshold, 1e-6)))
        let est = Double(live) * pow(1.0 + rate, Double(totalSteps)) * 2.0
        let capped = est.isFinite ? Int(min(est, Double(sim.maxCells))) : sim.maxCells
        return min(max(capped, live + 64), sim.maxCells)
    }

    /// V2.7: grow the window to fit the export aspect ratio (1600×1200 =
    /// 4:3 plus the 290 pt control-panel column) at ~80% of the screen
    /// height. Grow-only — never shrinks the user's window — and anchored
    /// at the top-left corner (AppKit frames are bottom-left origin).
    func resizeWindowForExport() {
        guard let w = owningWindow else { return }
        guard let visible = (w.screen ?? NSScreen.main)?.visibleFrame else { return }
        let panelWidth: CGFloat = 314   // the control-panel column + padding
        let targetH = visible.height * 0.8
        let targetW = min(visible.width, targetH * 4 / 3 + panelWidth)
        let newW = max(w.frame.width, targetW)
        let newH = max(w.frame.height, targetH)
        guard newW != w.frame.width || newH != w.frame.height else { return }
        let f = w.frame
        w.setFrame(NSRect(x: f.minX, y: f.maxY - newH, width: newW, height: newH),
                   display: true, animate: false)
    }

    /// Map the current panel knobs onto the offline render settings. Style
    /// maps 1:1 (Solid/X-ray/Foam); tint needs NO mapping — the ray tracer
    /// reads the same cellTint buffer the impostors use. The X-ray EXPOSURE
    /// keeps its offline tuned default: the GUI xrayGain is the interactive
    /// impostor premultiplier (a different knob on a different scale).
    /// Light follows the CLI --render-light auto convention (directional iff
    /// light-growth mode). Resolution: the CLI anim default 1600×1200 —
    /// deterministic and independent of the window size (documented choice).
    private func offlineExportSettings() -> RayRenderSettings {
        var s = RayRenderSettings()
        switch renderStyle {
        case .solid: s.style = .solid
        case .xray: s.style = .xray
        case .foam: s.style = .foam
        }
        s.width = 1600
        s.height = 1200
        s.foamOpacity = Float(foamOpacity)
        if growthMode == .light {
            s.directionalLight = SIMD3<Float>(sim.params.lightDirX,
                                              sim.params.lightDirY,
                                              sim.params.lightDirZ)
        }
        return s
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

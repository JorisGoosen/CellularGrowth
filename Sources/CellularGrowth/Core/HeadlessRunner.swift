//  HeadlessRunner.swift
//  CLI validation modes: headless stats / screenshot / bench / selftest /
//  lockstep (M3 CPU golden reference) / golden-image capture + check (M3).
//  This is the automated check that the simulation behaves: the ball relaxes,
//  links sit at rest length, (M2) a growing structure keeps a valid
//  closed-triangulation topology while cells divide, and (M3) the GPU
//  spatial-hash repulsion matches a brute-force CPU reference and the
//  renderer reproduces stored golden images.
//
//  M2: food + division are ON by default (--no-div restores M1 behaviour).
//  M3: repulsion is ON by default (roi 1.6, strength 0.1; --repulsion 0
//  restores the M2 regime). All headless jitter is seeded (deterministic
//  runs); the GUI keeps its non-seeded reset.

import AppKit
import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import Metal
import MetalKit
import UniformTypeIdentifiers

enum HeadlessRunner {

    static func shouldRun() -> Bool {
        let args = CommandLine.arguments
        return args.contains("--headless") || args.contains("--screenshot")
            || args.contains("--bench") || args.contains("--bench-growth")
            || args.contains("--selftest")
            || args.contains("--lockstep")
            || args.contains("--golden-capture") || args.contains("--golden-check")
            || args.contains("--render")
            // --- M5 (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--growth") || args.contains("--light-dir")
            || args.contains("--light-diffuse") || args.contains("--light-nutrient")
            || args.contains("--light-decay") || args.contains("--light-base")
            || args.contains("--light-interval")
            || args.contains("--rde-F") || args.contains("--rde-k")
            || args.contains("--rde-du") || args.contains("--rde-dv")
            || args.contains("--rde-base") || args.contains("--rde-gain")
            || args.contains("--rde-probe") || args.contains("--rde-dump")
            || args.contains("--render-light")
            // --- M6 (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--preset")
            || args.contains("--anim") || args.contains("--anim-steps")
            || args.contains("--anim-frames")
            || args.contains("--stereo") || args.contains("--interocular")
            || args.contains("--screenshot-style")
            // --- V2.1 (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--shadows") || args.contains("--xray-color")
            || args.contains("--light-soft")
            // --- V2.2 (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--tint") || args.contains("--heat-scale")
            || args.contains("--pulse-decay")
            // --- V2.1b (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--radius-scale")
            // --- V2.3 (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--type-poles") || args.contains("--type-t1")
            || args.contains("--type-t2") || args.contains("--phase-x")
            || args.contains("--lamel-rate") || args.contains("--dal-rate")
            || args.contains("--interior-divider")
            // --- V2.4 (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--differentiate-at") || args.contains("--differentiate-until")
            || args.contains("--type-mode") || args.contains("--type-cap-v")
            || args.contains("--type-stem-v") || args.contains("--diff-probe")
            // --- V2.5 (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--death") || args.contains("--death-threshold")
            || args.contains("--death-steps")
            // --- V2.6 (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--follow-centroid") || args.contains("--follow-smooth")
            || args.contains("--orbit-rate")
            // --- V2.8 (pitfall 10: every new flag routes to the headless path) ---
            || args.contains("--growth-uniform") || args.contains("--growth-light")
            || args.contains("--growth-rde") || args.contains("--drive-probe")
            // V2.5: --state-load was only ever routed headless via its
            // --render/--screenshot companions (a bare --state-load silently
            // launched the GUI event loop and hung — pitfall 10 verbatim).
            || args.contains("--state-load")
    }

    struct Options {
        var steps = 1000
        var subdiv = 3
        var jitter: Float = 0
        var spring: Float = 0.33
        var planar: Float = 0.33
        var bulge: Float = 0.33
        var food: Float = 0.01
        var threshold: Float = 1.0
        var division = true
        var seed: UInt32 = 12345
        var maxCells = 40_000_000
        var screenshotPath: String?
        var bench = false
        var benchGrowth = false
        var benchSteps = 50
        var selftest = false
        var divStop = Int.max   // diagnostic: disable division after this step
        // --- M3 ---
        var roi: Float = 1.6
        var repulsion: Float = 0.1
        var lockstep = false
        var goldenCapture: String?
        var goldenCheck: String?
        // --- M4 ---
        var renderPath: String?
        var style = "solid"        // solid | xray
        var styleGiven = false     // explicit --style switches goldens to the ray renderer
        var width = 1600
        var height = 1200
        var tiled = false
        var samples = 1
        var stateSave: String?     // save the grown state (CGST snapshot) after the run
        var stateLoad: String?     // load a state snapshot instead of growing
        // --- M5 ---
        var growth = "uniform"     // uniform | light | rde
        var lightDir = SIMD3<Float>(0, 1, 0)   // toward the light (normalized at apply)
        var lightDiffuse: Float = 0.02
        var lightNutrient: Float = 0.05
        var lightDecay: Float = 0.02
        var lightBase: Float = 0.15
        var lightInterval = 8
        var rdeF: Float = 0.037
        var rdeK: Float = 0.06
        var rdeDu: Float = 0.35
        var rdeDv: Float = 0.15
        var rdeBase: Float = 0.25
        var rdeGain: Float = 2.5
        var rdeProbe = false       // static-ball RDE pattern validation mode
        var rdeDump: String?       // chemical scatter dump directory (probe + runs)
        var renderLight = "auto"   // auto | growth | unbiased (M5 unified light)
        // --- M6 ---
        var preset: String?        // named parameter preset (Core/Presets.swift)
        var animDir: String?       // animation export: numbered PNG frames + .mp4 (V2.6; the M6 .mov was replaced)
        var animSteps = 150        // steps between frames
        var animFrames = 20        // total frames (frame 0 = the initial state)
        var stereo = false         // ray tracer: side-by-side stereo pair
        var interocular: Float = 0.03  // eye separation as a fraction of the scene scale
        var screenshotStyle = "solid"  // raster screenshot style: solid | xray (impostor)
        // --- V2.8 ---
        /// Explicit drive-strength overrides (nil = the --growth enum preset
        /// stands; an explicit flag wins over the enum-set component).
        var growthUniform: Float?
        var growthLight: Float?
        var growthRde: Float?
        /// Per-hemisphere per-capita growth probe (the V2.8 gating proof +
        /// the all-zero sanity check).
        var driveProbe = false
        // --- V2.1 ---
        var shadows: Float = 1.0       // per-pixel shadow strength (directional solid)
        var xrayColor = false          // coloured X-ray (per-channel tinted density)
        var lightSoft: Float = 0.35    // light angular radius (rad); 0 = binary verdict
        // --- V2.2 ---
        var tint = "auto"              // auto | none | chemical | food | pulse
        var heatScale: Float = 0       // food-heatmap normalization; 0 = auto default
        var pulseDecay: Float = 0.95   // per-step multiplier on the division pulse
        var heatScaleGiven = false     // --heat-scale present (golden meta provenance)
        var pulseDecayGiven = false    // --pulse-decay present (golden meta provenance)
        // --- V2.1b ---
        /// Render radius scale (computeRadii: mean link distance x radiusScale).
        /// The foam style reads best at 0.7 (neighbours overlap, walls have
        /// real area); 0.5 = spheres merely touch, walls degenerate to points.
        /// Only applied when the flag is given (the growth/loaded state keeps
        /// its own value otherwise). Radii are RENDER-side data — but light
        /// mode's illumination march uses them, so in light mode the flag
        /// changes the growth too (explicitly requested = fine).
        var radiusScale: Float = 0.5
        var radiusScaleGiven = false
        // --- V2.3 (cell types + two-phase growth) ---
        /// Polar typing axis (normalized at apply); nil = NO typing (all
        /// cells type 0 — the historical behaviour). Presence of ANY of
        /// --type-poles/--type-t1/--type-t2 enables typing (defaults: the
        /// vertical axis (0,1,0), t1 -0.35, t2 0.30).
        var typePoles: SIMD3<Float>?
        /// Stem threshold: s < t1 -> stem (type 1).
        var typeT1: Float = -0.35
        /// Cap threshold: s > t2 -> cap (type 4); t1 <= s <= t2 -> the
        /// lamel/dal band.
        var typeT2: Float = 0.30
        /// Phase-2 switch: divisions executed before the rule matrix flips
        /// (0 = never — pure phase 1).
        var phaseX = 0
        /// Phase-2 growth rates (SimParams mirrors; see foodStep).
        var lamelRate: Float = 1.0
        var dalRate: Float = 0.5
        var interiorDivider: Float = 10.0
        // --- V2.4 (differentiation phase: RDE-driven typing with hysteresis) ---
        /// Step at which the differentiation phase starts (0 = never): from
        /// this step the division block is skipped (division counter frozen,
        /// food frozen) and typeTransition converts type-0 cells from the RDE
        /// field every step.
        var differentiateAt = 0
        /// Step at which growth resumes (0 = stay differentiated until the
        /// host/GUI ends the phase).
        var differentiateUntil = 0
        /// Rule-matrix mode: band (polar typing — the V2.3 matrix) | region
        /// (RDE typing — every type divides uniformly within its region).
        /// Auto-resolution when not given explicitly: differentiation
        /// requested -> region, else band.
        var typeMode = "band"
        var typeModeGiven = false
        /// typeTransition thresholds (v >= -> cap, v <= -> stem; must satisfy
        /// stem < cap).
        var typeCapV: Float = 0.30
        var typeStemV: Float = 0.10
        /// Static-ball differentiation validation mode (like --rde-probe).
        var diffProbe = false
        // --- V2.5 (starvation death — the inverse of division) ---
        /// Death on (--death). Default off: every historical behaviour and
        /// every golden is death-free.
        var death = false
        /// Starvation threshold on the accumulated food ledger (default =
        /// splitThreshold: a cell starves while it cannot climb to its own
        /// division threshold fast enough).
        var deathThreshold: Float = 1.0
        /// Consecutive starving steps before death (default 100 — safely
        /// above the ~20-step food climb of fed cells at the default light
        /// gate values, well below the hundreds of steps a shaded cell needs;
        /// see starvationStep in Shaders.swift).
        var deathSteps = 250
        // --- V2.6 (camera choreography + video export) ---
        /// GUI-default-ON camera features are OPT-IN headless: follow-centroid
        /// and circle mode apply ONLY to --anim exports (the video paths).
        /// --screenshot/--render/goldens keep the legacy fixed-target framing
        /// unconditionally — without the flags every byte of those outputs is
        /// unchanged by V2.6. Documented asymmetry: the goldens cover the
        /// headless paths, not the GUI.
        var followCentroid = false
        /// Follow-target exponential smoothing per rendered frame
        /// (0 = snap, 1 = frozen; see CameraChoreography).
        var followSmoothing: Float = 0.15
        /// Circle mode: yaw advance in radians PER RENDERED FRAME (deterministic
        /// — export frames are equally spaced in steps). 0 = off.
        var orbitRate: Float = 0
    }

    private static func parseOptions() -> Options {
        var o = Options()
        let args = CommandLine.arguments
        func value(_ flag: String) -> String? {
            guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
            return args[idx + 1]
        }
        func given(_ flag: String) -> Bool { args.contains(flag) }

        // M6 presets FIRST: a preset is a full param bundle that pre-fills
        // the options; explicit CLI flags later in the loop still win.
        if let name = value("--preset") {
            guard let preset = SimPreset.named(name) else {
                let names = SimPreset.all.map { $0.name }.joined(separator: " | ")
                print("--preset '\(name)' unknown (\(names))")
                exit(1)
            }
            o.subdiv = preset.subdivision
            o.jitter = preset.jitter
            o.spring = preset.params.springFactor
            o.planar = preset.params.planarFactor
            o.bulge = preset.params.bulgeFactor
            o.food = preset.params.foodIncrement
            o.threshold = preset.params.splitThreshold
            o.division = preset.params.divisionEnabled != 0
            o.roi = preset.params.roi
            o.repulsion = preset.params.repulsionStrength
            o.growth = ["uniform", "light", "rde"][Int(min(preset.params.growthMode, 2))]
            let ld = SIMD3<Float>(preset.params.lightDirX, preset.params.lightDirY, preset.params.lightDirZ)
            if simd_length(ld) > 1e-6 { o.lightDir = ld }
            o.lightDiffuse = preset.params.lightDiffusionRate
            o.lightNutrient = preset.params.lightNutrient
            o.lightDecay = preset.params.lightDecay
            o.lightBase = preset.params.lightBase
            o.lightInterval = Int(preset.params.lightInterval)
            o.rdeF = preset.params.rdeF
            o.rdeK = preset.params.rdeK
            o.rdeDu = preset.params.rdeDu
            o.rdeDv = preset.params.rdeDv
            o.rdeBase = preset.params.rdeBase
            o.rdeGain = preset.params.rdeGain
            // V2.5: presets can carry the death knobs (explicit flags win).
            o.death = preset.params.deathEnabled != 0
            o.deathThreshold = preset.params.deathThreshold
            o.deathSteps = Int(preset.params.deathSteps)
            // V2.3: presets can carry the two-phase growth rates + the
            // polar typing knobs (explicit flags below still win).
            o.lamelRate = preset.params.lamelRate
            o.dalRate = preset.params.dalRate
            o.interiorDivider = preset.params.interiorDivider
            o.phaseX = Int(preset.params.phaseSwitchX)
            if let ax = preset.typeAxis {
                o.typePoles = ax
                o.typeT1 = preset.typeT1
                o.typeT2 = preset.typeT2
            }
            print("preset '\(preset.name)': \(preset.summary)")
        }

        for a in args {
            switch a {
            case "--steps": o.steps = Int(value("--steps") ?? "") ?? o.steps
            case "--subdiv": o.subdiv = Int(value("--subdiv") ?? "") ?? o.subdiv
            case "--jitter": o.jitter = Float(value("--jitter") ?? "") ?? o.jitter
            case "--spring": o.spring = Float(value("--spring") ?? "") ?? o.spring
            case "--planar": o.planar = Float(value("--planar") ?? "") ?? o.planar
            case "--bulge": o.bulge = Float(value("--bulge") ?? "") ?? o.bulge
            case "--food": o.food = Float(value("--food") ?? "") ?? o.food
            case "--threshold": o.threshold = Float(value("--threshold") ?? "") ?? o.threshold
            case "--no-div": o.division = false
            case "--seed": o.seed = UInt32(value("--seed") ?? "") ?? o.seed
            case "--max-cells": o.maxCells = Int(value("--max-cells") ?? "") ?? o.maxCells
            case "--screenshot": o.screenshotPath = value("--screenshot")
            case "--bench": o.bench = true
            case "--bench-steps": o.benchSteps = Int(value("--bench-steps") ?? "") ?? o.benchSteps
            case "--bench-growth": o.benchGrowth = true
            case "--selftest": o.selftest = true
            case "--div-stop": o.divStop = Int(value("--div-stop") ?? "") ?? o.divStop
            // --- M3 ---
            case "--roi": o.roi = Float(value("--roi") ?? "") ?? o.roi
            case "--repulsion": o.repulsion = Float(value("--repulsion") ?? "") ?? o.repulsion
            case "--lockstep": o.lockstep = true
            case "--golden-capture": o.goldenCapture = value("--golden-capture")
            case "--golden-check": o.goldenCheck = value("--golden-check")
            // --- M4 ---
            case "--render": o.renderPath = value("--render")
            case "--style":
                o.style = (value("--style") ?? "solid").lowercased()
                o.styleGiven = true
            case "--width": o.width = Int(value("--width") ?? "") ?? o.width
            case "--height": o.height = Int(value("--height") ?? "") ?? o.height
            case "--tiled": o.tiled = true
            case "--samples": o.samples = Int(value("--samples") ?? "") ?? o.samples
            case "--state-save": o.stateSave = value("--state-save")
            case "--state-load": o.stateLoad = value("--state-load")
            // --- M5 ---
            case "--growth": o.growth = (value("--growth") ?? "uniform").lowercased()
            case "--light-dir":
                if let v = value("--light-dir") {
                    let parts = v.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
                    if parts.count == 3 {
                        o.lightDir = SIMD3<Float>(parts[0], parts[1], parts[2])
                    } else {
                        print("--light-dir expects \"x,y,z\"; got '\(v)'")
                        exit(1)
                    }
                }
            case "--light-diffuse": o.lightDiffuse = Float(value("--light-diffuse") ?? "") ?? o.lightDiffuse
            case "--light-nutrient": o.lightNutrient = Float(value("--light-nutrient") ?? "") ?? o.lightNutrient
            case "--light-decay": o.lightDecay = Float(value("--light-decay") ?? "") ?? o.lightDecay
            case "--light-base": o.lightBase = Float(value("--light-base") ?? "") ?? o.lightBase
            case "--light-interval": o.lightInterval = Int(value("--light-interval") ?? "") ?? o.lightInterval
            case "--rde-F": o.rdeF = Float(value("--rde-F") ?? "") ?? o.rdeF
            case "--rde-k": o.rdeK = Float(value("--rde-k") ?? "") ?? o.rdeK
            case "--rde-du": o.rdeDu = Float(value("--rde-du") ?? "") ?? o.rdeDu
            case "--rde-dv": o.rdeDv = Float(value("--rde-dv") ?? "") ?? o.rdeDv
            case "--rde-base": o.rdeBase = Float(value("--rde-base") ?? "") ?? o.rdeBase
            case "--rde-gain": o.rdeGain = Float(value("--rde-gain") ?? "") ?? o.rdeGain
            case "--rde-probe": o.rdeProbe = true
            case "--rde-dump": o.rdeDump = value("--rde-dump")
            case "--render-light": o.renderLight = (value("--render-light") ?? "auto").lowercased()
            // --- M6 ---
            case "--preset": break   // consumed before the loop (preset-first)
            case "--anim": o.animDir = value("--anim")
            case "--anim-steps": o.animSteps = Int(value("--anim-steps") ?? "") ?? o.animSteps
            case "--anim-frames": o.animFrames = Int(value("--anim-frames") ?? "") ?? o.animFrames
            case "--stereo": o.stereo = true
            case "--interocular": o.interocular = Float(value("--interocular") ?? "") ?? o.interocular
            case "--screenshot-style":
                o.screenshotStyle = (value("--screenshot-style") ?? "solid").lowercased()
            // --- V2.1 ---
            case "--shadows": o.shadows = Float(value("--shadows") ?? "") ?? o.shadows
            case "--xray-color": o.xrayColor = true
            case "--light-soft": o.lightSoft = Float(value("--light-soft") ?? "") ?? o.lightSoft
            // --- V2.2 ---
            case "--tint": o.tint = (value("--tint") ?? "auto").lowercased()
            case "--heat-scale":
                o.heatScale = Float(value("--heat-scale") ?? "") ?? o.heatScale
                o.heatScaleGiven = true
            case "--pulse-decay":
                o.pulseDecay = Float(value("--pulse-decay") ?? "") ?? o.pulseDecay
                o.pulseDecayGiven = true
            // --- V2.1b ---
            case "--radius-scale":
                o.radiusScale = Float(value("--radius-scale") ?? "") ?? o.radiusScale
                o.radiusScaleGiven = true
            // --- V2.3 (cell types + two-phase growth) ---
            case "--type-poles":
                if let v = value("--type-poles") {
                    let parts = v.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
                    if parts.count == 3 {
                        o.typePoles = SIMD3<Float>(parts[0], parts[1], parts[2])
                    } else {
                        print("--type-poles expects \"x,y,z\"; got '\(v)'")
                        exit(1)
                    }
                }
            case "--type-t1": o.typeT1 = Float(value("--type-t1") ?? "") ?? o.typeT1
            case "--type-t2": o.typeT2 = Float(value("--type-t2") ?? "") ?? o.typeT2
            case "--phase-x": o.phaseX = Int(value("--phase-x") ?? "") ?? o.phaseX
            case "--lamel-rate": o.lamelRate = Float(value("--lamel-rate") ?? "") ?? o.lamelRate
            case "--dal-rate": o.dalRate = Float(value("--dal-rate") ?? "") ?? o.dalRate
            case "--interior-divider": o.interiorDivider = Float(value("--interior-divider") ?? "") ?? o.interiorDivider
            // --- V2.4 (differentiation phase) ---
            case "--differentiate-at": o.differentiateAt = Int(value("--differentiate-at") ?? "") ?? o.differentiateAt
            case "--differentiate-until": o.differentiateUntil = Int(value("--differentiate-until") ?? "") ?? o.differentiateUntil
            case "--type-mode":
                o.typeMode = (value("--type-mode") ?? "band").lowercased()
                o.typeModeGiven = true
            case "--type-cap-v": o.typeCapV = Float(value("--type-cap-v") ?? "") ?? o.typeCapV
            case "--type-stem-v": o.typeStemV = Float(value("--type-stem-v") ?? "") ?? o.typeStemV
            case "--diff-probe": o.diffProbe = true
            // --- V2.5 (starvation death) ---
            case "--death": o.death = true
            case "--death-threshold": o.deathThreshold = Float(value("--death-threshold") ?? "") ?? o.deathThreshold
            case "--death-steps": o.deathSteps = Int(value("--death-steps") ?? "") ?? o.deathSteps
            // --- V2.6 (camera choreography + video export) ---
            case "--follow-centroid": o.followCentroid = true
            case "--follow-smooth": o.followSmoothing = Float(value("--follow-smooth") ?? "") ?? o.followSmoothing
            case "--orbit-rate": o.orbitRate = Float(value("--orbit-rate") ?? "") ?? o.orbitRate
            // --- V2.8 (composable growth drives) ---
            case "--growth-uniform": o.growthUniform = Float(value("--growth-uniform") ?? "") ?? o.growthUniform
            case "--growth-light": o.growthLight = Float(value("--growth-light") ?? "") ?? o.growthLight
            case "--growth-rde": o.growthRde = Float(value("--growth-rde") ?? "") ?? o.growthRde
            case "--drive-probe": o.driveProbe = true
            default: break
            }
        }
        // Mode-specific defaults (only when the flag was not given):
        // lockstep: 300 steps, jitter 0.2, capped at 2k cells (brute force O(N^2)).
        // goldens: 400 steps, subdiv 4, jitter 0.25.
        // rde-probe: 2000 steps (static ball, division off).
        if o.lockstep {
            if !given("--steps") { o.steps = 300 }
            if !given("--jitter") { o.jitter = 0.2 }
            if !given("--subdiv") { o.subdiv = 3 }
        }
        if o.goldenCapture != nil || o.goldenCheck != nil {
            if !given("--steps") { o.steps = 400 }
            if !given("--subdiv") { o.subdiv = 4 }
            if !given("--jitter") { o.jitter = 0.25 }
        }
        if o.rdeProbe {
            if !given("--steps") { o.steps = 2000 }
            if !given("--subdiv") { o.subdiv = 4 }
        }
        if o.diffProbe {
            if !given("--steps") { o.steps = 2000 }
            if !given("--subdiv") { o.subdiv = 4 }
            if !given("--differentiate-at") { o.differentiateAt = max(500, o.steps / 4) }
        }
        if o.driveProbe {
            if !given("--steps") { o.steps = 1500 }
            if !given("--subdiv") { o.subdiv = 3 }
        }
        return o
    }

    static func run() -> Never {
        let o = parseOptions()

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("HeadlessRunner: no Metal device")
        }
        print("Device: \(device.name)")

        // The selftest caps growth below maxCells so CPU-side invariant checks
        // stay fast; the capacity-guard code path is identical for any cap.
        let effectiveMaxCells = o.selftest ? min(o.maxCells, 400_000) : o.maxCells
        let sim = CellSim(device: device, queue: queue, maxCells: effectiveMaxCells)
        sim.params.springFactor = o.spring
        sim.params.planarFactor = o.planar
        sim.params.bulgeFactor = o.bulge
        sim.params.foodIncrement = o.food
        sim.params.splitThreshold = o.threshold
        sim.params.divisionEnabled = o.division ? 1 : 0
        sim.params.rngSeed = o.seed
        sim.params.roi = o.roi
        sim.params.repulsionStrength = o.repulsion
        // --- M5 ---
        // V2.8: the enum is a preset-setter for the three additive drives
        // (setGrowthMode); the explicit --growth-* flags below override the
        // enum-set component (a combination is free).
        sim.setGrowthMode(growthModeCode(o.growth))
        if let v = o.growthUniform { sim.params.growthUniform = max(v, 0) }
        if let v = o.growthLight { sim.params.growthLight = max(v, 0) }
        if let v = o.growthRde { sim.params.growthRde = max(v, 0) }
        let lLen = simd_length(o.lightDir)
        guard lLen > 1e-6 else {
            print("--light-dir must be non-zero")
            exit(1)
        }
        let L = o.lightDir / lLen
        sim.params.lightDirX = L.x
        sim.params.lightDirY = L.y
        sim.params.lightDirZ = L.z
        // Convex-combination stability clamps (see nutrientDiffuse/illuminate).
        sim.params.lightDiffusionRate = min(max(o.lightDiffuse, 0), 1)
        sim.params.lightNutrient = max(o.lightNutrient, 0)
        sim.params.lightDecay = min(max(o.lightDecay, 0), 0.999)
        sim.params.lightBase = max(o.lightBase, 0)
        sim.params.lightInterval = UInt32(max(o.lightInterval, 1))
        sim.params.rdeF = o.rdeF
        sim.params.rdeK = o.rdeK
        sim.params.rdeDu = min(max(o.rdeDu, 0), 1)
        sim.params.rdeDv = min(max(o.rdeDv, 0), 1)
        sim.params.rdeBase = max(o.rdeBase, 0)
        sim.params.rdeGain = o.rdeGain
        // M6 diagnostic kill switch for the light-march coarse skip (A/B
        // validation of the occupancy optimization).
        if ProcessInfo.processInfo.environment["CG_NO_COARSE_SKIP"] != nil {
            sim.params.lightCoarseSkip = 0
        }
        // V2.1 fractional illumination softness (light angular radius in
        // radians; 0 = the legacy binary verdict). Clamped to a sane range:
        // the coverage ramp degenerates for lightSoft >= pi.
        sim.params.lightSoft = min(max(o.lightSoft, 0), 1.5)
        // --- V2.2 tint modes + mitosis pulse ---
        sim.params.tintMode = tintModeCode(o.tint)
        sim.params.heatScale = max(o.heatScale, 0)   // <= 1e-6 = shader-side auto default
        sim.params.pulseDecay = min(max(o.pulseDecay, 0), 0.9999)
        // --- V2.1b render radius scale (foam recommends 0.7; 0.5 = walls
        // degenerate to points). Only when the flag is given. Re-applied
        // after state loads below — loadState restores the state's own
        // params, which would otherwise win.
        if o.radiusScaleGiven {
            sim.params.radiusScale = min(max(o.radiusScale, 0.05), 1)
            sim.syncParams()
        }
        // --- V2.3 cell types + two-phase growth ---
        sim.params.lamelRate = max(o.lamelRate, 0)
        sim.params.dalRate = max(o.dalRate, 0)
        sim.params.interiorDivider = max(o.interiorDivider, 1)
        sim.params.phaseSwitchX = UInt32(max(o.phaseX, 0))
        // --- V2.4 differentiation phase ---
        if o.typeStemV >= o.typeCapV {
            print("--type-stem-v (\(o.typeStemV)) must be < --type-cap-v (\(o.typeCapV))")
            exit(1)
        }
        // Rule-matrix mode auto-resolution: a requested differentiation phase
        // (RDE typing) -> region; otherwise band (polar typing / untyped —
        // both are no-ops on type-0 cells). An explicit --type-mode wins.
        if o.typeModeGiven {
            switch o.typeMode {
            case "band": sim.params.typeMode = 0
            case "region": sim.params.typeMode = 1
            default:
                print("--type-mode '\(o.typeMode)' unknown (band | region)")
                exit(1)
            }
        } else {
            sim.params.typeMode = o.differentiateAt > 0 ? 1 : 0
        }
        sim.params.typeCapV = o.typeCapV
        sim.params.typeStemV = o.typeStemV
        sim.params.differentiationPhase = 0   // the phase starts via the step range (growAndReport/bench)
        // --- V2.5 starvation death ---
        sim.params.deathEnabled = o.death ? 1 : 0
        sim.params.deathThreshold = max(o.deathThreshold, 0)
        sim.params.deathSteps = UInt32(max(o.deathSteps, 0))

        if o.diffProbe {
            runDiffProbe(queue: queue, sim: sim, o: o)   // exits
        }
        if o.rdeProbe {
            runRdeProbe(queue: queue, sim: sim, o: o)   // exits
        }
        if o.driveProbe {
            runDriveProbe(queue: queue, sim: sim, o: o)   // exits (V2.8)
        }
        if o.selftest {
            runSelftest(queue: queue, sim: sim, o: o)
        }
        if o.bench || o.benchGrowth {
            runBench(queue: queue, sim: sim, o: o)
        }
        if o.lockstep {
            runLockstep(queue: queue, sim: sim, o: o)
        }
        if let name = o.goldenCapture {
            runGoldenCapture(name, queue: queue, sim: sim, o: o)
        }
        if let name = o.goldenCheck {
            runGoldenCheck(name, queue: queue, sim: sim, o: o)
        }
        if o.renderPath != nil {
            runRender(queue: queue, sim: sim, o: o)   // exits
        }
        if o.animDir != nil {
            runAnimation(queue: queue, sim: sim, o: o)   // exits (M6)
        }
        runHeadless(queue: queue, sim: sim, o: o)
    }

    // MARK: - Headless stats / screenshot

    /// The seeded growth pipeline shared by --headless/--screenshot (M1-M3)
    /// and --render (M4): reset from the CLI config, run the configured steps
    /// in batches, report progress/stats, NaN check. GPU idle on return.
    /// `useSeededJitter`: --render replaces the GUI's unseeded
    /// SimViewModel.randomInBall jitter with HeadlessRunner.seededJitter so
    /// the rendered image is a pure function of the CLI config (reproducible
    /// renders; --screenshot keeps its historical unseeded behaviour).
    private static func growAndReport(queue: MTLCommandQueue, sim: CellSim, o: Options, useSeededJitter: Bool = false) {
        let (base, links) = Icosphere.generate(subdivisions: o.subdiv, spacing: sim.params.linkRestLength)
        let positions = o.jitter > 0
            ? (useSeededJitter ? seededJitter(base, radius: o.jitter, seed: o.seed)
                               : base.map { $0 + SimViewModel.randomInBall(radius: o.jitter) })
            : base

        // CPU-side sanity: mean neighbour distance straight from the generated arrays.
        var cpuSum: Float = 0
        var cpuCount: Float = 0
        var longLinks = 0
        for (i, ring) in links.enumerated() {
            for l in ring {
                let d = simd_distance(positions[i], positions[Int(l)])
                cpuSum += d
                cpuCount += 1
                if d > 2 * sim.params.linkRestLength { longLinks += 1 }
            }
        }
        print("CPU mean link length: \(cpuSum / cpuCount)  links=\(Int(cpuCount))  links>2x rest: \(longLinks)")
        print("cell 0 ring: \(links[0])  count=\(links[0].count)")
        print("Cells: \(positions.count)  (icosphere subdiv \(o.subdiv), jitter \(o.jitter), division \(sim.params.divisionEnabled != 0))")

        sim.reset(positions: positions, links: links)
        // V2.3: phase-0 polar typing when the run requests it — BEFORE any
        // step runs (types are reset-time state; daughters inherit).
        applyTypingIfRequested(sim: sim, o: o)

        func meanLinkLength() -> Float {
            guard let cb = queue.makeCommandBuffer() else { return -1 }
            sim.encodeStats(cb)
            cb.commit()
            cb.waitUntilCompleted()
            sim.syncCellCountFromDevice(allowGridGrowth: true)
            return sim.readMeanLinkLength()
        }

        print("t=0        cells=\(sim.cellCount)  meanLinkLength=\(meanLinkLength())")

        let t0 = CFAbsoluteTimeGetCurrent()
        var done = 0
        // V2.4: the differentiation step-range (--differentiate-at N
        // [--differentiate-until M]) toggles params.differentiationPhase at
        // EXACT step boundaries: batches are split at the boundaries and the
        // params update happens with the GPU idle between command buffers
        // (no mid-flight-params hazard). until == 0 -> the phase never ends
        // host-side.
        var diffPhase = false
        func setDiffPhase(_ on: Bool, at step: Int) {
            diffPhase = on
            sim.params.differentiationPhase = on ? 1 : 0
            sim.syncParams()
            print("t=\(step): differentiation phase \(on ? "STARTED — division frozen, RDE typing active (\(sim.params.growthMode == 2 ? "rde" : "non-rde: growth pause only"))" : "ENDED — growth resumed")")
        }
        var lastTypes: [Int]? = nil   // for the per-type Δ (typing vs division)
        var lastDeaths: Int? = nil    // V2.5: for the deaths-per-window cascade
        var lastDeathStep = 0
        while done < o.steps {
            var batch = min(100, o.steps - done)
            // Split the batch at the next phase boundary inside it.
            var boundary: Int? = nil
            if !diffPhase, o.differentiateAt > done, o.differentiateAt <= done + batch {
                boundary = o.differentiateAt
            }
            if diffPhase, o.differentiateUntil > done, o.differentiateUntil <= done + batch {
                boundary = min(boundary ?? Int.max, o.differentiateUntil)
            }
            if let b = boundary { batch = b - done }
            guard let cb = queue.makeCommandBuffer() else { fatalError("no command buffer") }
            for _ in 0..<batch { sim.step(cb) }
            cb.commit()
            cb.waitUntilCompleted()
            done += batch
            sim.syncCellCountFromDevice(allowGridGrowth: true)
            if !diffPhase, o.differentiateAt > 0, o.differentiateAt == done {
                setDiffPhase(true, at: done)
            }
            if diffPhase, o.differentiateUntil > 0, o.differentiateUntil == done {
                setDiffPhase(false, at: done)
            }
            if done >= o.divStop && sim.params.divisionEnabled != 0 {
                sim.params.divisionEnabled = 0
                sim.syncParams()
                print("t=\(done): division disabled (diagnostic --div-stop)")
            }
            if done % 500 == 0 || done == o.steps {
                // V2.5: with death on the headline count is the LIVE count —
                // dead slots stay allocated (the counter is a high-water mark).
                let liveLabel = sim.params.deathEnabled != 0
                    ? " live=\(sim.liveCount) dead=\(sim.deathCount)"
                    : ""
                print("t=\(done)   cells=\(sim.cellCount)\(liveLabel)  meanLinkLength=\(meanLinkLength())")
                printGrowthCheckpoint(sim: sim)
                if sim.params.deathEnabled != 0 {
                    lastDeaths = printDeathCheckpoint(sim: sim, previousDeaths: lastDeaths,
                                                      windowSteps: done - lastDeathStep)
                    lastDeathStep = done
                }
                if isTyped(sim) || diffPhase {
                    lastTypes = printTypedCheckpoint(sim: sim, previous: lastTypes)
                }
            }
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("Elapsed: \(String(format: "%.2f", elapsed))s  (\(String(format: "%.0f", Double(o.steps) / max(elapsed, 1e-9))) steps/s)")

        // NaN / range check (readback fine at validation scale).
        if sim.cellCount <= 1_000_000 {
            var nanCount = 0
            var maxAbs: Float = 0
            for p in sim.readPositions() {
                if p.x.isNaN || p.y.isNaN || p.z.isNaN { nanCount += 1 }
                maxAbs = max(maxAbs, simd_length(p))
            }
            print("NaN cells: \(nanCount)   max |P|: \(maxAbs)")
        }
        // CG_CHECK=1: the full invariant harness at the end of a headless run
        // (fast topology bisect loop without the whole selftest).
        if ProcessInfo.processInfo.environment["CG_CHECK"] != nil {
            let (pass, lines) = checkInvariants(sim: sim, settled: false, meanLimit: 10)
            for l in lines { print("   \(l)") }
            print("CG_CHECK: \(pass ? "PASS" : "FAIL")")
        }

        // M5 directional-growth diagnostics (the paper-gate metric: does the
        // form elongate along the light? Isotropic forms: ratio ~1, offset ~0).
        let (ratio, offset) = shapeMetrics(sim, lightDir: simd_normalize(
            SIMD3<Float>(sim.params.lightDirX, sim.params.lightDirY, sim.params.lightDirZ)))
        print("shape: axial/tangential extent ratio = \(String(format: "%.3f", ratio))   centroid offset along light = \(String(format: "%.3f", offset)) (rms tangential units)")

        if let dumpDir = o.rdeDump {
            do { try writeChemicalDumps(sim: sim, dir: dumpDir, tag: "final") } catch {
                print("chemical dump FAILED: \(error)")
            }
        }
    }

    /// Per-checkpoint chemical stats for the growth modes (M5; cheap host
    /// readback at validation scale).
    private static func printGrowthCheckpoint(sim: CellSim) {
        let light = SIMD3<Float>(sim.params.lightDirX, sim.params.lightDirY, sim.params.lightDirZ)
        switch sim.params.growthMode {
        case 1:
            let nut = sim.readNutrient()
            let nor = UnsafeBufferPointer(start: sim.normalsPointer, count: sim.cellCount)
            var facing = (sum: Float(0), n: 0)
            var away = (sum: Float(0), n: 0)
            var maxN: Float = 0
            for i in 0..<sim.cellCount {
                let d = simd_dot(simd_normalize(nor[i]), light)
                if d > 0.3 { facing.sum += nut[i]; facing.n += 1 }
                else if d < -0.3 { away.sum += nut[i]; away.n += 1 }
                maxN = max(maxN, nut[i])
            }
            let fm = facing.n > 0 ? facing.sum / Float(facing.n) : 0
            let am = away.n > 0 ? away.sum / Float(away.n) : 0
            print("      nutrient: mean \(String(format: "%.3f", nut.reduce(0, +) / Float(max(nut.count, 1))))  max \(String(format: "%.3f", maxN))  facing(N·L>0.3) \(String(format: "%.3f", fm)) (n=\(facing.n))  away(N·L<-0.3) \(String(format: "%.3f", am)) (n=\(away.n))")
        case 2:
            let u = sim.readChemU()
            let v = sim.readChemV()
            let us = fieldStats(u)
            let vs = fieldStats(v)
            print("      u: mean \(String(format: "%.3f", us.mean)) var \(String(format: "%.4f", us.variance))  v: mean \(String(format: "%.3f", vs.mean)) var \(String(format: "%.4f", vs.variance)) max \(String(format: "%.3f", vs.max))")
        default:
            break
        }
        // V2.2 mitosis-pulse diagnostics (tint mode pulse): how much of the
        // surface is flashing, and how SPATIALLY CLUSTERED the recent
        // divisions are. The light-mode wavefront signature is a high ring
        // R² (recent divisions cluster in the lit band and move with it) vs
        // uniform mode's scattered speckle (uncorrelated neighbours,
        // ring R² ~ 1/valence ≈ 0.17 like any iid field).
        if sim.params.tintMode == 4 {
            let v = sim.cellCount
            if v > 0 {
                let pulse = sim.readPulse()
                let links = UnsafeBufferPointer(start: sim.linksPointer, count: v * CellSim.maxValence)
                let counts = UnsafeBufferPointer(start: sim.linkCountsPointer, count: v)
                var ringAvg = [Float](repeating: 0, count: v)
                for i in 0..<v {
                    let n = Int(counts[i])
                    guard n > 0 else { ringAvg[i] = pulse[i]; continue }
                    var acc: Float = 0
                    let base = i * CellSim.maxValence
                    for k in 0..<n { acc += pulse[Int(links[base + k])] }
                    ringAvg[i] = acc / Float(n)
                }
                let s = fieldStats(pulse)
                let rs = fieldStats(ringAvg)
                let r2 = s.variance > 0 ? rs.variance / s.variance : 0
                let active = pulse.lazy.filter { $0 > 0.05 }.count
                print("      pulse: mean \(String(format: "%.3f", s.mean)) max \(String(format: "%.3f", s.max)) active(>0.05) \(String(format: "%.1f%%", 100 * Float(active) / Float(v)))  ring R² \(String(format: "%.3f", r2)) (clustered ≫ ~0.17)")
            }
        }
    }

    private struct FieldStats {
        var mean: Float = 0
        var variance: Float = 0
        var min: Float = 0
        var max: Float = 0
    }

    private static func fieldStats(_ v: [Float]) -> FieldStats {
        guard !v.isEmpty else { return FieldStats() }
        var s: Float = 0
        var mn = v[0], mx = v[0]
        for x in v { s += x; mn = min(mn, x); mx = max(mx, x) }
        let mean = s / Float(v.count)
        var q: Float = 0
        for x in v { let d = x - mean; q += d * d }
        return FieldStats(mean: mean, variance: q / Float(v.count), min: mn, max: mx)
    }

    /// Directional shape metrics: axial (along the light) vs tangential extent
    /// around the centroid, and the centroid's offset along the light in
    /// tangential-rms units. Used for the paper gates (plant-like forms
    /// elongate toward the light; uniform/organ forms are near-isotropic).
    static func shapeMetrics(_ sim: CellSim, lightDir: SIMD3<Float>) -> (axialRatio: Float, centroidOffset: Float) {
        let positions = sim.readPositions()
        guard positions.count > 2 else { return (0, 0) }
        var c = SIMD3<Float>.zero
        for p in positions { c += p }
        c /= Float(positions.count)
        let L = simd_normalize(lightDir)
        var ax: Float = 0
        var tot: Float = 0
        for p in positions {
            let d = p - c
            let a = simd_dot(d, L)
            ax += a * a
            tot += simd_dot(d, d)
        }
        let n = Float(positions.count)
        let axial = sqrt(ax / n)
        let tangential = sqrt(max(tot / n - ax / n, 0))
        let ratio = tangential > 1e-6 ? axial / tangential : 0
        let offset = simd_dot(c, L) / max(tangential, 1e-6)
        return (ratio, offset)
    }

    /// M5 growth-mode code from the CLI string; exits on an unknown mode.
    private static func growthModeCode(_ s: String) -> UInt32 {
        switch s {
        case "uniform": return 0
        case "light": return 1
        case "rde": return 2
        default:
            print("--growth '\(s)' unknown (uniform | light | rde)")
            exit(1)
        }
    }

    /// V2.2/V2.3 tint-mode code from the CLI string; exits on an unknown mode.
    /// "auto" (the default) reproduces the pre-V2.2 behaviour exactly:
    /// chemical colours in rde mode, neutral albedo otherwise.
    static func tintModeCode(_ s: String) -> UInt32 {
        switch s {
        case "auto", "default": return 0
        case "none": return 1
        case "chemical": return 2
        case "food": return 3
        case "pulse": return 4
        case "type": return 5          // V2.3 type colours
        case "type-pulse": return 6    // V2.3 type colours + mitosis-pulse overlay
        case "starve": return 7        // V2.5 the starvation-death front
        default:
            print("--tint '\(s)' unknown (auto | none | chemical | food | pulse | type | type-pulse | starve)")
            exit(1)
        }
    }

    // MARK: - M5 RDE pattern probe (--rde-probe)

    /// Validates the Gray-Scott RDE on a STATIC ball (division OFF) BEFORE it
    /// drives growth (M5 scope B): patterns must FORM — spatial variance grows
    /// from the seeded noise and the chemical field becomes spatially
    /// organized. Metrics per chemical:
    ///   variance               — pattern amplitude vs the white-noise init
    ///   ring-avg explained var — var(avg over ring)/var: ~1/valence ≈ 0.17
    ///                            for iid noise, → high for structured patterns
    ///   lag-1 link correlation — corr(x_i, x_j) over directed links
    /// Prints a checkpoint table, writes u/v scatter dumps, PASS/FAIL + exit.
    private static func runRdeProbe(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        sim.setGrowthMode(2)
        sim.params.divisionEnabled = 0
        sim.syncParams()
        let (base, links) = Icosphere.generate(subdivisions: o.subdiv, spacing: sim.params.linkRestLength)
        print("rde-probe: STATIC ball (division OFF), subdiv \(o.subdiv) (\(base.count) cells), jitter 0, F=\(sim.params.rdeF) k=\(sim.params.rdeK) Du=\(sim.params.rdeDu) Dv=\(sim.params.rdeDv), \(o.steps) steps")
        sim.reset(positions: base, links: links)
        let dumpDir = o.rdeDump ?? "/private/tmp/opencode/cellulargrowth/m5"
        do { try writeChemicalDumps(sim: sim, dir: dumpDir, tag: "t00000") } catch {
            print("chemical dump FAILED: \(error)")
        }

        func fmt(_ v: Float) -> String { String(format: "%.4f", v) }
        var first: (u: ChemStats, v: ChemStats)?
        var secondLast: (u: ChemStats, v: ChemStats)?
        var last: (u: ChemStats, v: ChemStats)?
        var printed = Set<Int>()
        func snap(_ t: Int) {
            guard !printed.contains(t) else { return }
            printed.insert(t)
            let su = chemStats(sim.readChemU(), links: links)
            let sv = chemStats(sim.readChemV(), links: links)
            print("t=\(t)  u: mean \(fmt(su.mean)) var \(fmt(su.variance)) R² \(fmt(su.ringR2)) lag1 \(fmt(su.lag1))  |  v: mean \(fmt(sv.mean)) var \(fmt(sv.variance)) R² \(fmt(sv.ringR2)) lag1 \(fmt(sv.lag1))")
            if first == nil { first = (su, sv) }
            secondLast = last
            last = (su, sv)
        }
        snap(0)
        var done = 0
        let checkpoints = Set([100, 500]).union(stride(from: 1000, through: o.steps, by: 500))
        while done < o.steps {
            let batch = min(100, o.steps - done)
            runSteps(queue: queue, sim: sim, count: batch)
            done += batch
            if checkpoints.contains(done) { snap(done) }
        }
        snap(o.steps)
        do { try writeChemicalDumps(sim: sim, dir: dumpDir, tag: String(format: "t%05d", o.steps)) } catch {
            print("chemical dump FAILED: \(error)")
        }

        guard let f = first, let sl = secondLast, let l = last, f.v.mean > 1e-9 else {
            print("RDE PROBE FAIL: degenerate statistics")
            exit(1)
        }
        // Pattern-formation evidence, with metrics chosen for CLUSTER seeds
        // (the seeded t=0 field is already spatially structured, so
        // ratio-to-t0 variance/R² gates would be meaningless):
        //   amplification — v mean grows >= 3x (the nonlinear instability ran);
        //   reaction      — u mean drops below 0.9 (substrate actually consumed);
        //   organization  — ring-avg R² of v >= 0.4 at t=N (iid noise sits at
        //                   ~1/valence ≈ 0.17; structured patterns measure high);
        //   stability     — the last two checkpoints agree (a dissipative
        //                   structure, not a decaying transient).
        let amp = l.v.mean / f.v.mean
        let stable = abs(l.v.mean - sl.v.mean) < 0.05 * max(l.v.mean, 1e-6)
        print("verdict: v mean \(fmt(f.v.mean)) -> \(fmt(l.v.mean)) (x\(String(format: "%.1f", amp)), gate >= 3), u mean \(fmt(f.u.mean)) -> \(fmt(l.u.mean)) (gate < 0.9), v R² \(String(format: "%.3f", l.v.ringR2)) (gate >= 0.4), steady \(stable ? "yes" : "NO")")
        let ok = amp >= 3 && l.u.mean < 0.9 && l.v.ringR2 >= 0.4 && stable && l.u.mean.isFinite
        print(ok ? "RDE PROBE PASS" : "RDE PROBE FAIL")
        exit(ok ? 0 : 1)
    }

    /// V2.8: per-hemisphere per-capita growth probe — the drive-combination
    /// validation tool. Grows the CURRENT drive config from a fresh ball and
    /// checkpoints the cell counts on the LIT vs SHADED hemisphere (dot(P,
    /// lightDir) >= 0 vs < 0 — the shared light points along +lightDir, so a
    /// cell on the light side of the origin plane is the lit side to first
    /// order). Prints the per-capita table; ENFORCES gates only for the two
    /// drive signatures that carry a physical expectation:
    ///   - light-gated RDE only (R > 0, L == U == 0, growthMode != 2): the
    ///     lit hemisphere must outgrow the shaded one — the RDE drive term is
    ///     ~0 where litFraction ~ 0 (the V2.8 gating proof);
    ///   - all drives zero (division on): the count must stay FLAT — no food,
    ///     no division (stagnation sanity).
    /// Any other signature prints an informational table and exits 0. Cells
    /// crossing the equator between checkpoints add a little classification
    /// noise — far below the gate margins.
    private static func runDriveProbe(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        sim.params.divisionEnabled = 1
        sim.syncParams()
        let (base, links) = Icosphere.generate(subdivisions: max(o.subdiv, 3), spacing: sim.params.linkRestLength)
        let L = simd_normalize(SIMD3<Float>(sim.params.lightDirX,
                                            sim.params.lightDirY,
                                            sim.params.lightDirZ))
        func hemispheres(_ positions: [SIMD3<Float>]) -> (lit: Int, shade: Int) {
            var lit = 0, shade = 0
            for p in positions {
                if simd_dot(p, L) >= 0 { lit += 1 } else { shade += 1 }
            }
            return (lit, shade)
        }
        sim.reset(positions: base, links: links)
        let initial = hemispheres(sim.readPositions())
        print("drive-probe: \(base.count) cells, drives U=\(sim.params.growthUniform) L=\(sim.params.growthLight) R=\(sim.params.growthRde) (growthMode \(sim.params.growthMode)), light (\(L.x), \(L.y), \(L.z)), \(o.steps) steps")
        print("t=0 lit=\(initial.lit) shade=\(initial.shade)")
        var done = 0
        var last = initial
        while done < o.steps {
            let batch = min(100, o.steps - done)
            runSteps(queue: queue, sim: sim, count: batch)
            done += batch
            if done % 500 == 0 || done == o.steps {
                last = hemispheres(sim.readPositions())
                let litPC = Double(last.lit) / Double(max(initial.lit, 1))
                let shadePC = Double(last.shade) / Double(max(initial.shade, 1))
                print("t=\(done) lit=\(last.lit) (x\(String(format: "%.2f", litPC))) shade=\(last.shade) (x\(String(format: "%.2f", shadePC))) cells=\(sim.cellCount)")
            }
        }
        let litPC = Double(last.lit) / Double(max(initial.lit, 1))
        let shadePC = Double(last.shade) / Double(max(initial.shade, 1))
        let gatedRdeOnly = sim.params.growthRde > 0 && sim.params.growthLight == 0
            && sim.params.growthUniform == 0 && sim.params.growthMode != 2
        let allZero = sim.params.growthUniform == 0 && sim.params.growthLight == 0
            && sim.params.growthRde == 0
        var ok = true
        if gatedRdeOnly {
            // Gating proof: the lit hemisphere must grow substantially while
            // the shaded one stays near flat (the terminator band's partial
            // soft-light visibility may add a few percent of shade-side
            // divisions — the 2x margin absorbs it).
            let gateOK = litPC > 1.5 && litPC > 2.0 * shadePC
            print("gating verdict (R-only, light-gated): lit x\(String(format: "%.2f", litPC)) vs shade x\(String(format: "%.2f", shadePC)) — \(gateOK ? "PASS (lit ≫ shade)" : "FAIL") (gates: lit > 1.5 && lit > 2 × shade)")
            ok = gateOK
        } else if allZero {
            let flat = sim.cellCount == base.count
            print("stagnation verdict (all drives 0): cells \(base.count) -> \(sim.cellCount) — \(flat ? "PASS (no growth without food)" : "FAIL")")
            ok = flat
        } else {
            print("informational probe (no gates for this drive signature)")
        }
        exit(ok ? 0 : 1)
    }

    private struct ChemStats {
        var mean: Float
        var variance: Float
        var min: Float
        var max: Float
        var ringR2: Float   // var(ring average) / var — spatial organization
        var lag1: Float     // correlation between directly linked cells
    }

    private static func chemStats(_ values: [Float], links: [[UInt32]]) -> ChemStats {
        let s = fieldStats(values)
        guard s.variance > 0, !values.isEmpty else {
            return ChemStats(mean: s.mean, variance: 0, min: s.min, max: s.max, ringR2: 0, lag1: 0)
        }
        // Ring averages (spatial smoothing over one link).
        var ringAvg = [Float](repeating: 0, count: values.count)
        var linkSum: Float = 0
        var linkPairs = 0
        for (i, ring) in links.enumerated() {
            guard !ring.isEmpty else { ringAvg[i] = values[i]; continue }
            var acc: Float = 0
            for j in ring {
                acc += values[Int(j)]
                linkSum += (values[i] - s.mean) * (values[Int(j)] - s.mean)
                linkPairs += 1
            }
            ringAvg[i] = acc / Float(ring.count)
        }
        let rs = fieldStats(ringAvg)
        let lag1 = linkPairs > 0 ? linkSum / (Float(linkPairs) * s.variance) : 0
        return ChemStats(mean: s.mean, variance: s.variance, min: s.min, max: s.max,
                         ringR2: rs.variance / s.variance, lag1: lag1)
    }

    // MARK: - V2.4 differentiation probe (--diff-probe)

    /// Validates the DIFFERENTIATION PHASE on a STATIC ball (division OFF):
    /// the RDE settles for --differentiate-at steps (default steps/4), then
    /// the phase turns on — typeTransition converts type-0 cells every step
    /// while the field keeps evolving under the freeze rule (differentiated
    /// cells stop participating; see rdeStep/typeTransition in Shaders.swift).
    /// Reported per checkpoint:
    ///   v variance / ring-avg R² — the pattern the typing reads;
    ///   type counts              — 0/V -> mixed (stem + cap present);
    ///   FLIPS per window         — new transitions since the last checkpoint;
    ///     with type-memory hysteresis a cell changes type AT MOST ONCE
    ///     (0 -> 1 or 0 -> 4, never back), so the flip stream must settle
    ///     to ~0 — the no-flicker gate.
    /// Writes the type-map PNG (type colours) at the phase start and the end.
    /// PASS/FAIL + exit code.
    private static func runDiffProbe(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        sim.setGrowthMode(2)
        sim.params.divisionEnabled = 0
        sim.params.typeMode = 1          // region (RDE typing)
        sim.params.typeCapV = o.typeCapV
        sim.params.typeStemV = o.typeStemV
        sim.syncParams()
        let settle = o.differentiateAt
        let (base, links) = Icosphere.generate(subdivisions: o.subdiv, spacing: sim.params.linkRestLength)
        print("diff-probe: STATIC ball (division OFF), subdiv \(o.subdiv) (\(base.count) cells), F=\(sim.params.rdeF) k=\(sim.params.rdeK) Du=\(sim.params.rdeDu) Dv=\(sim.params.rdeDv), thresholds v>=\(o.typeCapV) cap / v<=\(o.typeStemV) stem, RDE settle \(settle) steps, differentiation \(o.steps - settle) steps")
        sim.reset(positions: base, links: links)
        let dumpDir = o.rdeDump ?? "/private/tmp/opencode/cellulargrowth/v24"
        do { try writeTypeScatter("\(dumpDir)/types_t00000.png", sim: sim) } catch {
            print("type dump FAILED: \(error)")
        }

        func fmt(_ v: Float) -> String { String(format: "%.4f", v) }
        var settleV: ChemStats?
        var prevTypes: [UInt32]?
        var lastFlips = -1
        var lastCounts: [Int] = []
        var flipCurve: [Int] = []   // flips per checkpoint window (the settle curve)
        var memoryOK = true      // type memory: only 0 -> 1 / 0 -> 4 transitions
        var sawFlips = false
        func typeSnap(_ t: Int) {
            let types = sim.readTypes()
            var counts = [Int](repeating: 0, count: 5)
            var flips = 0
            if let prev = prevTypes {
                for i in 0..<min(prev.count, types.count) {
                    let a = prev[i], b = types[i]
                    if a != b {
                        flips += 1
                        if !(a == 0 && (b == 1 || b == 4)) { memoryOK = false }
                    }
                }
                if flips > 0 { sawFlips = true }
            }
            for x in types where Int(x) < 5 { counts[Int(x)] += 1 }
            let vs = chemStats(sim.readChemV(), links: links)
            print("t=\(t)  v: mean \(fmt(vs.mean)) var \(fmt(vs.variance)) R² \(fmt(vs.ringR2))  |  types: none=\(counts[0]) stem=\(counts[1]) lamel=\(counts[2]) dal=\(counts[3]) cap=\(counts[4])  flips=\(flips)")
            prevTypes = types
            lastFlips = flips
            lastCounts = counts
            flipCurve.append(flips)
        }

        // Stage 1: RDE settle (no typing — the field the thresholds read).
        runSteps(queue: queue, sim: sim, count: settle)
        settleV = chemStats(sim.readChemV(), links: links)
        print("settle t=\(settle): v mean \(fmt(settleV!.mean)) var \(fmt(settleV!.variance)) R² \(fmt(settleV!.ringR2)) lag1 \(fmt(settleV!.lag1)) (pattern gate: R² >= 0.4)")
        sim.params.differentiationPhase = 1
        sim.syncParams()
        print("t=\(settle): differentiation phase STARTED — division frozen, RDE typing active")
        typeSnap(settle)
        do { try writeTypeScatter("\(dumpDir)/types_t\(String(format: "%05d", settle)).png", sim: sim) } catch {
            print("type dump FAILED: \(error)")
        }
        var done = settle
        while done < o.steps {
            let batch = min(100, o.steps - done)
            runSteps(queue: queue, sim: sim, count: batch)
            done += batch
            typeSnap(done)
        }
        sim.params.differentiationPhase = 0
        sim.syncParams()
        do {
            try writeTypeScatter("\(dumpDir)/types_t\(String(format: "%05d", o.steps)).png", sim: sim)
            try writeChemicalDumps(sim: sim, dir: dumpDir, tag: String(format: "t%05d", o.steps))
        } catch {
            print("probe dumps FAILED: \(error)")
        }

        // Hygiene invariants (no division here — mostly the NaN/finiteness
        // net over the frozen-chemical dynamics).
        let (invPass, invLines) = checkInvariants(sim: sim, settled: false, meanLimit: 1.5)
        for l in invLines { print("   \(l)") }

        guard let sv = settleV else {
            print("DIFF PROBE FAIL: degenerate statistics")
            exit(1)
        }
        let v = sim.cellCount
        let patternOK = sv.ringR2 >= 0.4
        let mixed = Float(lastCounts[1]) >= 0.05 * Float(v) && Float(lastCounts[4]) >= 0.02 * Float(v)
        // "Flips settle to ~0": the residual trickle of marginal hoverers
        // resolving (the RDE keeps stepping in the phase) must fall to a
        // small fraction of V — far below the O(V)-per-window churn a
        // FLICKERING (hysteresis-free) pattern would produce. Measured on the
        // reference run: the burst window flips ~56% of V, the tail sits at
        // 0-4 cells (<= 0.16%/window). Gates: <= 0.5% of V per window AND
        // <= 5% of the burst window's rate.
        let maxFlips = flipCurve.max() ?? 0
        let flipGate = max(1, Int(0.005 * Float(v)))
        let settled = lastFlips <= flipGate && lastFlips <= maxFlips / 20 && sawFlips
        let typed = Float(lastCounts[1] + lastCounts[4]) / Float(v)
        print("verdict: v R² \(String(format: "%.3f", sv.ringR2)) (gate >= 0.4) \(patternOK ? "PASS" : "FAIL"), types mixed stem \(String(format: "%.1f%%", 100 * Float(lastCounts[1]) / Float(v))) cap \(String(format: "%.1f%%", 100 * Float(lastCounts[4]) / Float(v))) (gates >= 5% / >= 2%) \(mixed ? "PASS" : "FAIL"), typed total \(String(format: "%.1f%%", 100 * typed)), last-window flips \(lastFlips) (gate <= \(flipGate), burst \(maxFlips)) \(settled ? "PASS" : "FAIL"), type memory \(memoryOK ? "PASS" : "FAIL") \(sawFlips ? "" : "(no transitions observed at all)")")
        let ok = patternOK && mixed && settled && memoryOK && invPass
        print(ok ? "DIFF PROBE PASS" : "DIFF PROBE FAIL")
        exit(ok ? 0 : 1)
    }

    // MARK: - M5 chemical dumps

    /// Chemical scatter dumps: rde mode → u and v maps; light mode → the
    /// nutrient map. Diagnostic artefacts (the renderers are untouched).
    static func writeChemicalDumps(sim: CellSim, dir: String, tag: String) throws {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        switch sim.params.growthMode {
        case 1:
            try writeChemicalScatter("\(dir)/nutrient_\(tag).png", sim: sim, values: sim.readNutrient())
        case 2:
            try writeChemicalScatter("\(dir)/u_\(tag).png", sim: sim, values: sim.readChemU())
            try writeChemicalScatter("\(dir)/v_\(tag).png", sim: sim, values: sim.readChemV())
        default:
            print("chemical dumps: nothing to dump in uniform mode")
            return
        }
        print("chemical dumps written to \(dir) (\(tag))")
    }

    /// Orthographic point-splat of a per-cell scalar along +Z (far cells
    /// first, near cells painted over them). Greyscale, the value range
    /// normalized to [min, max]. Diagnostic only.
    private static func writeChemicalScatter(_ path: String, sim: CellSim, values: [Float]) throws {
        let count = sim.cellCount
        guard count == values.count, count > 0 else { return }
        let pos = UnsafeBufferPointer(start: sim.positionsPointer, count: count)
        let rad = UnsafeBufferPointer(start: sim.radiiPointer, count: count)
        var lo = pos[0]
        var hi = pos[0]
        for p in pos { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let spanX = max(hi.x - lo.x, 1e-3)
        let spanY = max(hi.y - lo.y, 1e-3)
        let size = 800
        let pad = 8
        let scale = Float(size - 2 * pad) / max(spanX, spanY)
        var mn = values[0]
        var mx = values[0]
        for v in values { mn = min(mn, v); mx = max(mx, v) }
        let range = max(mx - mn, 1e-6)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                                  bytesPerRow: size * 4, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw NSError(domain: "dump", code: 1)
        }
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        // Far-to-near painting order (orthographic +Z view).
        var order = Array(0..<count)
        order.sort { pos[$0].z < pos[$1].z }
        for i in order {
            let t = (values[i] - mn) / range
            let g = 0.08 + 0.92 * Double(t)
            ctx.setFillColor(CGColor(gray: g, alpha: 1))
            let r = Double(max(rad[i], 0.25) * scale * 0.9)
            let cx = Double(pad) + Double((pos[i].x - lo.x) * scale)
            let cy = Double(size) - Double(pad) - Double((pos[i].y - lo.y) * scale)   // +y up
            ctx.fillEllipse(in: CGRect(x: cx - r, y: cy - r, width: 2 * r, height: 2 * r))
        }
        guard let img = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                         UTType.png.identifier as CFString, 1, nil) else {
            throw NSError(domain: "dump", code: 2)
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else { throw NSError(domain: "dump", code: 3) }
    }

    /// V2.4: orthographic point-splat coloured by CELL TYPE — the
    /// typeTransition map (stem white, lamel warm white, dal cool white, cap
    /// red, type 0 neutral — the SAME palette as the cellTint type mode, so
    /// the dump and the renders agree). Same projection as
    /// writeChemicalScatter. Diagnostic only.
    private static func writeTypeScatter(_ path: String, sim: CellSim) throws {
        let count = sim.cellCount
        let types = sim.readTypes()
        guard count == types.count, count > 0 else { return }
        let pos = UnsafeBufferPointer(start: sim.positionsPointer, count: count)
        let rad = UnsafeBufferPointer(start: sim.radiiPointer, count: count)
        var lo = pos[0]
        var hi = pos[0]
        for p in pos { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let spanX = max(hi.x - lo.x, 1e-3)
        let spanY = max(hi.y - lo.y, 1e-3)
        let size = 800
        let pad = 8
        let scale = Float(size - 2 * pad) / max(spanX, spanY)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                                  bytesPerRow: size * 4, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw NSError(domain: "dump", code: 1)
        }
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        var order = Array(0..<count)
        order.sort { pos[$0].z < pos[$1].z }
        for i in order {
            let t = Int(min(types[i], 4))
            let c: (Double, Double, Double) = t == 1 ? (1.00, 1.00, 1.00)     // stem
                : t == 2 ? (1.00, 0.88, 0.72)                                 // lamel
                : t == 3 ? (0.72, 0.86, 1.00)                                 // dal
                : t == 4 ? (0.80, 0.10, 0.10)                                 // cap
                : (0.87, 0.85, 0.82)                                          // none
            ctx.setFillColor(CGColor(red: c.0, green: c.1, blue: c.2, alpha: 1))
            let r = Double(max(rad[i], 0.25) * scale * 0.9)
            let cx = Double(pad) + Double((pos[i].x - lo.x) * scale)
            let cy = Double(size) - Double(pad) - Double((pos[i].y - lo.y) * scale)   // +y up
            ctx.fillEllipse(in: CGRect(x: cx - r, y: cy - r, width: 2 * r, height: 2 * r))
        }
        guard let img = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                         UTType.png.identifier as CFString, 1, nil) else {
            throw NSError(domain: "dump", code: 2)
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else { throw NSError(domain: "dump", code: 3) }
    }

    /// V2.2: an EXPLICIT CLI tint request overrides the tint mode a loaded
    /// state carries (tintMode lives in SimParams — the cellTint kernel is
    /// part of the step pipeline — but it is a RENDER knob: --tint pulse on a
    /// state-load must render pulses, not the mode the state was saved with).
    /// Without the flag the state's stored mode wins — that is what makes
    /// golden snapshot replays byte-exact without repeating the flags.
    private static func applyTintOverrideAfterLoad(sim: CellSim, o: Options) {
        guard o.tint != "auto" || o.heatScaleGiven || o.pulseDecayGiven else { return }
        if o.tint != "auto" { sim.params.tintMode = tintModeCode(o.tint) }
        if o.heatScaleGiven { sim.params.heatScale = max(o.heatScale, 0) }
        if o.pulseDecayGiven { sim.params.pulseDecay = min(max(o.pulseDecay, 0), 0.9999) }
        sim.syncParams()
        sim.refreshTint()
    }

    /// V2.1b: re-apply --radius-scale AFTER a state load (loadState restores
    /// the state's own SimParams). The caller must refresh the radii (compute
    /// computeRadii runs with the new scale) before rendering — both render
    /// paths run a refreshRadii command buffer right after this.
    private static func applyRadiusScaleAfterLoad(sim: CellSim, o: Options) {
        guard o.radiusScaleGiven else { return }
        sim.params.radiusScale = min(max(o.radiusScale, 0.05), 1)
        sim.syncParams()
    }

    /// V2.3: apply the phase-0 polar typing when the run requests it
    /// (--type-poles/--type-t1/--type-t2, or a preset that carries them).
    /// Called right after every reset that starts a typed run (and after a
    /// state load — re-bands the loaded geometry, the GUI re-type semantics).
    private static func applyTypingIfRequested(sim: CellSim, o: Options) {
        guard let axis = o.typePoles else { return }
        sim.applyPolarTypes(axis: axis, t1: o.typeT1, t2: o.typeT2)
        let t = sim.readTypes()
        var counts = [Int](repeating: 0, count: 5)
        for x in t where Int(x) < 5 { counts[Int(x)] += 1 }
        print("typing: axis (\(axis.x),\(axis.y),\(axis.z)) t1 \(o.typeT1) t2 \(o.typeT2) -> none=\(counts[0]) stem=\(counts[1]) lamel=\(counts[2]) dal=\(counts[3]) cap=\(counts[4])")
    }

    /// V2.3 typed-growth checkpoint (printed once per checkpoint when the
    /// run is typed): per-type counts (Σ must equal V — every cell typed
    /// exactly once, daughters inherit), per-type MEAN PULSE (the phase-2
    /// validation: only voeg cells divide in phase 2, so the pulse mass must
    /// sit on lamel/dal), and the phase state. V2.4: `previous` (the counts
    /// of the previous checkpoint) adds the per-type Δ — during the
    /// differentiation phase Δ = newly typed cells (typeTransition), during
    /// growth Δ = divisions per type (daughters inherit the parent's type,
    /// so a type's count grows by exactly one per division of that type —
    /// the per-region division-rate metric). Returns the current counts.
    @discardableResult
    private static func printTypedCheckpoint(sim: CellSim, previous: [Int]? = nil) -> [Int] {
        let v = sim.cellCount
        guard v > 0 else { return [] }
        let types = sim.readTypes()
        let dead = sim.readDeadFlags()   // V2.5: dead slots carry stale types — excluded
        var counts = [Int](repeating: 0, count: 5)
        var badType = 0
        for i in 0..<v where dead[i] == 0 {
            let k = Int(types[i])
            if k < 5 { counts[k] += 1 } else { badType += 1 }
        }
        let pulse = sim.readPulse()
        var pulseSum = [Float](repeating: 0, count: 5)
        for i in 0..<v where dead[i] == 0 {
            let t = Int(min(types[i], 4))
            pulseSum[t] += pulse[i]
        }
        func avg(_ s: Float, _ n: Int) -> String {
            n > 0 ? String(format: "%.3f", s / Float(n)) : "  -  "
        }
        let div = sim.divisionCounter
        let x = Int(sim.params.phaseSwitchX)
        let phase = (x != 0 && div >= x) ? 2 : 1
        // Valence pressure diagnostic (V2.3): divisions insert daughters into
        // plane neighbours' rings, so cells at the growth front (and the
        // dormant band cells beside it) accumulate links. A cell at
        // maxValence defers its neighbours' splits — a saturated band freezes
        // edge growth (documented failure mode; watched in V2.3).
        let countsBuf = UnsafeBufferPointer(start: sim.linkCountsPointer, count: v)
        var valenceHist = [Int](repeating: 0, count: Int(CellSim.maxValence) + 1)
        var maxVal = 0
        for i in 0..<v where dead[i] == 0 {
            let k = Int(countsBuf[i])
            if k <= Int(CellSim.maxValence) { valenceHist[k] += 1 }
            maxVal = max(maxVal, k)
        }
        let atCap = valenceHist[Int(CellSim.maxValence)]
        // Which types carry the valence pressure? (The dormant band cells
        // cannot relieve their own valence by dividing — the suspected
        // saturation mechanism.)
        var satByType = [Int](repeating: 0, count: 5)
        for i in 0..<v where dead[i] == 0 && Int(countsBuf[i]) >= Int(CellSim.maxValence) {
            let t = Int(min(types[i], 4))
            satByType[t] += 1
        }
        print("      valence: max=\(maxVal) at cap(\(CellSim.maxValence))=\(atCap) (\(String(format: "%.1f%%", 100 * Float(atCap) / Float(max(sim.liveCount, 1))))))  v14+ = \(valenceHist[14] + valenceHist[15] + valenceHist[16])  sat by type: none=\(satByType[0]) stem=\(satByType[1]) lamel=\(satByType[2]) dal=\(satByType[3]) cap=\(satByType[4])")
        let diff = sim.params.differentiationPhase != 0
        let deltaLabel = diff ? "Δ typing" : "Δ divisions"
        let deltas = previous != nil
            ? "  \(deltaLabel): none=\(counts[0] - previous![0]) stem=\(counts[1] - previous![1]) lamel=\(counts[2] - previous![2]) dal=\(counts[3] - previous![3]) cap=\(counts[4] - previous![4])"
            : ""
        print("      types: none=\(counts[0]) stem=\(counts[1]) lamel=\(counts[2]) dal=\(counts[3]) cap=\(counts[4]) Σ=\(sim.liveCount)\(badType == 0 ? "" : " BAD=\(badType)")  divisions=\(div) phase=\(phase)\(diff ? " DIFFERENTIATING" : "")\(deltas)")
        print("      pulse/type: none \(avg(pulseSum[0], counts[0]))  stem \(avg(pulseSum[1], counts[1]))  lamel \(avg(pulseSum[2], counts[2]))  dal \(avg(pulseSum[3], counts[3]))  cap \(avg(pulseSum[4], counts[4]))")
        return counts
    }

    /// True when the live state carries ANY typed cell (types are checked
    /// once per checkpoint — cheap host readback at validation scale).
    private static func isTyped(_ sim: CellSim) -> Bool {
        sim.readTypes().contains { $0 != 0 }
    }

    /// Decide-replay helper: walks divisionExecute's FULL precondition chain
    /// for one would-claim cell — the multi-pair retry included (top-8
    /// most-opposite pairs, each with the empty-arc swap) — and returns
    /// "none (would execute)" when some pair passes, else the last failing
    /// reason. Debug-only (the death checkpoint's decide replay).
    private static func replayExecutePrecondition(sim: CellSim, cell: Int,
                                                  linksBuf: UnsafeBufferPointer<UInt32>,
                                                  countsBuf: UnsafeBufferPointer<UInt32>,
                                                  foodBuf: UnsafeBufferPointer<Float>,
                                                  dead: [UInt32]) -> String {
        let maxV = CellSim.maxValence
        let posBuf = UnsafeBufferPointer(start: sim.positionsPointer, count: sim.cellCount)
        let n = Int(countsBuf[cell])
        let base = cell * maxV
        var ringPos = [SIMD3<Float>](repeating: .zero, count: n)
        var ringCell = [Int](repeating: 0, count: n)
        var dirs = [SIMD3<Float>](repeating: .zero, count: n)
        for k in 0..<n {
            ringCell[k] = Int(linksBuf[base + k])
            ringPos[k] = posBuf[ringCell[k]]
            let d = ringPos[k] - posBuf[cell]
            let l = simd_length(d)
            if l < 1e-9 { return "zero-length link" }
            dirs[k] = d / l
        }
        var acc = SIMD3<Float>.zero
        var prev = ringPos[n - 1] - posBuf[cell]
        for k in 0..<n {
            let cur = ringPos[k] - posBuf[cell]
            acc += simd_cross(prev, cur)
            prev = cur
        }
        if simd_dot(acc, acc) < 1e-20 { return "degenerate ring normal" }
        let N = simd_normalize(acc)
        // All pairs, most-opposite first (the GPU keeps the best 8; the
        // replay walks all — a superset, so "none" here implies "none" there).
        var pairs: [(dot: Float, a: Int, b: Int)] = []
        for a in 0..<n {
            for b in (a + 1)..<n { pairs.append((simd_dot(dirs[a], dirs[b]), a, b)) }
        }
        pairs.sort { $0.dot != $1.dot ? $0.dot < $1.dot : ($0.a != $1.a ? $0.a < $1.a : $0.b < $1.b) }
        var lastFail = "no pairs"
        for pr in pairs {
            let w0 = dirs[pr.a] - dirs[pr.b]
            let wLen = simd_length(w0)
            if wLen < 1e-5 { lastFail = "plane links parallel"; continue }
            let m0 = simd_cross(N, w0 / wLen)
            let m2 = simd_dot(m0, m0)
            if m2 < 1e-12 { lastFail = "N parallel to axis"; continue }
            let m = m0 / sqrt(m2)
            var reason = ""
            var res = classifyHost(ringPos, ringCell, n, pr.a, pr.b, posBuf[cell], m, &reason)
            if !res.ok && reason == "empty daughter arc" {
                // Empty between-arc: the reversed pair (m flips with w).
                var reason2 = ""
                res = classifyHost(ringPos, ringCell, n, pr.b, pr.a, posBuf[cell], -m, &reason2)
                reason = reason2
            }
            if !res.ok { lastFail = reason; continue }
            // Valence guards (the pre-scans are skipped in the replay — the
            // classification + guards are the observed jam causes).
            let p1 = ringCell[pr.a], p2 = ringCell[pr.b]
            if res.aCount + 3 > maxV || res.bCount + 3 > maxV {
                lastFail = "parent/daughter valence guard"; continue
            }
            if Int(countsBuf[p1]) + 1 > maxV || Int(countsBuf[p2]) + 1 > maxV {
                lastFail = "plane-neighbour valence guard"; continue
            }
            return "none (would execute)"
        }
        return lastFail
    }

    /// Host twin of classifyCleavageArc (see Shaders.swift): returns
    /// (ok, bCount, aCount); on !ok the reason string says why.
    private static func classifyHost(_ ringPos: [SIMD3<Float>], _ ringCell: [Int], _ n: Int,
                                     _ k1: Int, _ k2: Int, _ P: SIMD3<Float>, _ m: SIMD3<Float>,
                                     _ reason: inout String) -> (ok: Bool, bCount: Int, aCount: Int) {
        let k2rot = (k2 + n - k1) % n
        var bCount = 0
        for t in 1..<k2rot {
            let r = (k1 + t) % n
            if simd_dot(ringPos[r] - P, m) <= 0 {
                reason = "arc not contiguous (B-side entry at rotated offset \(t), plane \(k1)/\(k2) of \(n))"
                return (false, 0, 0)
            }
            bCount += 1
        }
        if bCount == 0 {
            reason = "empty daughter arc"
            return (false, 0, 0)
        }
        var aCount = 0
        for t in (k2rot + 1)..<n {
            let r = (k1 + t) % n
            if simd_dot(ringPos[r] - P, m) > 0 {
                reason = "arc not contiguous (A-side entry at rotated offset \(t), plane \(k1)/\(k2) of \(n))"
                return (false, bCount, aCount)
            }
            aCount += 1
        }
        reason = ""
        return (true, bCount, aCount)
    }

    /// V2.5 death-dynamics checkpoint (printed when death is enabled): the
    /// deaths-per-window cascade (sparse at first, bursts where starvation
    /// clusters — the shaded regions die together once their climb time
    /// exceeds the window), the starving-counter distribution over LIVE
    /// cells, and the current candidate count. Returns the cumulative death
    /// count for the next window's delta.
    @discardableResult
    private static func printDeathCheckpoint(sim: CellSim, previousDeaths: Int?,
                                             windowSteps: Int) -> Int {
        let deaths = sim.deathCounter
        var window = ""
        if let prev = previousDeaths, windowSteps > 0 {
            let d = deaths - prev
            window = "  deaths/window \(d) (\(String(format: "%.2f", Double(d) / Double(windowSteps)))/step)"
        }
        let starving = sim.readStarving()
        let dead = sim.readDeadFlags()
        let steps = max(Int(sim.params.deathSteps), 1)
        var sum = 0; var mx = 0; var half = 0; var cand = 0; var liveN = 0
        for i in 0..<sim.cellCount where dead[i] == 0 {
            liveN += 1
            let s = Int(starving[i])
            sum += s; mx = max(mx, s)
            if s >= steps / 2 { half += 1 }
            if s >= steps { cand += 1 }
        }
        let mean = liveN > 0 ? Float(sum) / Float(liveN) : 0
        // Live food ledger stats: the division fuel. A frozen form with
        // nobody starving and no divisions shows up here (e.g. everything
        // pinned AT the threshold — a decide-side stall).
        let foodBuf = UnsafeBufferPointer(start: sim.foodPointer, count: sim.cellCount)
        var fSum: Float = 0, fMin: Float = .infinity, fMax: Float = -.infinity, above = 0
        for i in 0..<sim.cellCount where dead[i] == 0 {
            let f = foodBuf[i]
            fSum += f; fMin = min(fMin, f); fMax = max(fMax, f)
            if f > sim.params.splitThreshold { above += 1 }
        }
        print("      death: total \(deaths)  live \(sim.liveCount)/\(sim.cellCount)  divisions=\(sim.divisionCounter)\(window)")
        print("      food(live): mean \(String(format: "%.3f", fSum / Float(max(liveN, 1)))) min \(String(format: "%.3f", fMin)) max \(String(format: "%.3f", fMax))  above splitThreshold \(above)")
        // Decide-side state: live valence histogram + how many claims the
        // division decide currently holds (a frozen form with everyone above
        // the threshold and zero claims = a decide-side stall).
        let countsBuf = UnsafeBufferPointer(start: sim.linkCountsPointer, count: sim.cellCount)
        var hist = [Int](repeating: 0, count: Int(CellSim.maxValence) + 1)
        var vMax = 0
        for i in 0..<sim.cellCount where dead[i] == 0 {
            let k = Int(countsBuf[i])
            if k <= Int(CellSim.maxValence) { hist[k] += 1 }
            vMax = max(vMax, k)
        }
        let splitSum = UnsafeBufferPointer(start: sim.splitFlagsPointer, count: sim.cellCount)
            .reduce(0) { $0 + Int($1) }
        let histLine = (3...min(16, CellSim.maxValence)).map { "v\($0)=\(hist[$0])" }.joined(separator: " ")
        print("      valence(live): max=\(vMax) \(histLine)  splitFlags set: \(splitSum)  divEnabled=\(sim.params.divisionEnabled) diffPhase=\(sim.params.differentiationPhase)")
        // Decide replay on the CPU for the 3 lowest-indexed live cells: why
        // does the global index minimum not claim? Walks the SAME predicate
        // (food > threshold, lower index, distance <= 2) and reports the
        // blocking cell + its dead/alive state.
        let linksBuf = UnsafeBufferPointer(start: sim.linksPointer, count: sim.cellCount * CellSim.maxValence)
        let thr = sim.params.splitThreshold
        // Decide replay (one summary line): walks the division decide's
        // predicate + the execute preconditions for the lowest-indexed live
        // cells. In a healthy state everything is below threshold or blocked;
        // an anomaly (a would-claim cell whose execute preconditions fail on
        // ALL candidate pairs) prints its reason — that is the jam signature.
        var replayNote = "all below threshold"
        var liveSeen = 0
        for i in 0..<sim.cellCount {
            guard dead[i] == 0 else { continue }
            liveSeen += 1
            if liveSeen > 3 { break }
            let n = Int(countsBuf[i])
            guard foodBuf[i] > thr else { continue }
            var blocker: (idx: Int, dead: Bool, dist: Int)? = nil
            let base = i * CellSim.maxValence
            scan: for k in 0..<n {
                let j = Int(linksBuf[base + k])
                if foodBuf[j] > thr && j < i { blocker = (j, dead[j] != 0, 1); break scan }
                let jn = Int(countsBuf[j])
                let jbase = j * CellSim.maxValence
                for k2 in 0..<jn {
                    let l = Int(linksBuf[jbase + k2])
                    if l != i && foodBuf[l] > thr && l < i { blocker = (l, dead[l] != 0, 2); break scan }
                }
            }
            if let b = blocker {
                replayNote = "cell \(i) blocked by \(b.idx) (dead=\(b.dead), dist \(b.dist))"
            } else {
                let why = replayExecutePrecondition(sim: sim, cell: i, linksBuf: linksBuf,
                                                    countsBuf: countsBuf, foodBuf: foodBuf, dead: dead)
                replayNote = why == "none (would execute)"
                    ? "cell \(i) would claim+execute (healthy)"
                    : "JAM: cell \(i) would claim, execute stuck: \(why)"
            }
            break   // the lowest-indexed above-threshold cell tells the story
        }
        print("      decide-replay: \(replayNote)")
        print("      starving: mean \(String(format: "%.1f", mean)) max \(mx)  >=50% of window \(half) (\(String(format: "%.1f%%", 100 * Float(half) / Float(max(liveN, 1)))))  candidates(>=\(steps)) \(cand)")
        return deaths
    }

    private static func runHeadless(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        // M6: --state-load works in the headless path too — render/screenshot
        // a stored state instead of growing (matches the --render workflow).
        if let loadPath = o.stateLoad {
            do {
                let snap = try sim.loadState(from: loadPath)
                let modes = ["uniform", "light", "rde"]
                // V2.8: also show the active drives — a pre-V2.8 params block
                // was mapped from the file's growth mode (a full 216-byte
                // block carries its own drives).
                print("state loaded: \(loadPath) (CGST v\(snap.version), \(sim.cellCount) cells, mode \(modes[Int(min(snap.growthMode, 2))]), drives U=\(sim.params.growthUniform) L=\(sim.params.growthLight) R=\(sim.params.growthRde))")
                // V2.5: a death-era state restores its dead slots + starving
                // counters + death counter; live = counter - deaths.
                if sim.deathCount > 0 {
                    print("   death state: \(sim.deathCount) dead slots, \(sim.liveCount) live, deathCounter \(sim.deathCounter)")
                }
                applyTintOverrideAfterLoad(sim: sim, o: o)
                applyRadiusScaleAfterLoad(sim: sim, o: o)
                // V2.3: --type-poles re-bands a LOADED state (the GUI re-type
                // semantics); without the flag the state's stored types win.
                applyTypingIfRequested(sim: sim, o: o)
                if isTyped(sim) { printTypedCheckpoint(sim: sim) }   // persisted types + phase
            } catch {
                print("state load FAILED: \(error)")
                exit(1)
            }
        } else {
            // V2.5 fix: the headless path is DOCUMENTED as seeded (file
            // header: "All headless jitter is seeded (deterministic runs)")
            // but called growAndReport without useSeededJitter, so it ran the
            // GUI's unseeded randomInBall — two identical --headless
            // invocations started from different positions and diverged
            // (found while validating the V2.5 death path: two-run state
            // compares differed at step 0). --screenshot keeps its
            // documented historical unseeded behaviour; --render was already
            // seeded.
            growAndReport(queue: queue, sim: sim, o: o, useSeededJitter: true)
        }

        // M5: --state-save now works in the headless path too (it was only
        // handled by --render before; the flag silently did nothing here).
        if let savePath = o.stateSave {
            do {
                try sim.saveState(to: savePath)
                print("state saved: \(savePath) (\(sim.cellCount) cells, CGST v3 full state)")
            } catch {
                print("state save FAILED: \(error)")
                exit(1)
            }
        }

        if let path = o.screenshotPath {
            // Ensure every cell (including daughters from the final step) has a
            // current radius before rendering.
            if let cb = queue.makeCommandBuffer() {
                sim.refreshRadii(cb)
                cb.commit()
                cb.waitUntilCompleted()
            }
            if o.screenshotStyle != "solid" && o.screenshotStyle != "xray" && o.screenshotStyle != "foam" {
                print("--screenshot-style '\(o.screenshotStyle)' unknown; using solid")
            }
            renderScreenshot(sim: sim, queue: queue, to: path,
                             style: (o.screenshotStyle == "xray" || o.screenshotStyle == "foam")
                                    ? o.screenshotStyle : "solid",
                             foamOpacity: foamOpacityFromEnv())
        }

        exit(0)
    }

    // MARK: - M4 offline ray-traced render

    /// --render PATH: grow the configured state (same seeded pipeline as
    /// --headless), then ray-trace it offscreen in the chosen style. The grid
    /// for the final state is built ONCE here (the sim rebuilds it per step;
    /// rendering traverses a static snapshot of it).
    /// V2.1b foam look knobs via env (look iteration + bench A/B without a
    /// CLI contract change; goldens record their explicit values in the meta).
    static func foamOpacityFromEnv() -> Float {
        ProcessInfo.processInfo.environment["CG_FOAM_OPACITY"].flatMap { Float($0) } ?? 0.5
    }
    static func foamLayersFromEnv() -> Int {
        ProcessInfo.processInfo.environment["CG_FOAM_LAYERS"].flatMap { Int($0) } ?? 6
    }
    static func foamWallFromEnv() -> Float {
        ProcessInfo.processInfo.environment["CG_FOAM_WALL"].flatMap { Float($0) } ?? 0.25
    }

    private static func runRender(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        if let loadPath = o.stateLoad {
            // Render a STORED state (--state-load): skips the growth entirely
            // (paper-gate workflow: same state rendered in both styles).
            do {
                let snap = try sim.loadState(from: loadPath)
                let modes = ["uniform", "light", "rde"]
                // V2.8: show the active drives (see runHeadless).
                print("state loaded: \(loadPath) (CGST v\(snap.version), \(sim.cellCount) cells, mode \(modes[Int(min(snap.growthMode, 2))]), drives U=\(sim.params.growthUniform) L=\(sim.params.growthLight) R=\(sim.params.growthRde))")
                // V2.5: a death-era state restores its dead slots + starving
                // counters + death counter; live = counter - deaths.
                if sim.deathCount > 0 {
                    print("   death state: \(sim.deathCount) dead slots, \(sim.liveCount) live, deathCounter \(sim.deathCounter)")
                }
                applyTintOverrideAfterLoad(sim: sim, o: o)
                applyRadiusScaleAfterLoad(sim: sim, o: o)
                applyTypingIfRequested(sim: sim, o: o)   // V2.3: re-band on request
            } catch {
                print("state load FAILED: \(error)")
                exit(1)
            }
        } else {
            growAndReport(queue: queue, sim: sim, o: o, useSeededJitter: true)
        }

        // Fresh radii (daughters from the final step) + one grid rebuild for
        // the final state, in a single waited command buffer. States rendered
        // with the DIRECTIONAL unified light additionally recompute the
        // per-cell illumination fractions (litFraction) — a state snapshot
        // stores geometry, not the illumination. The light mode is decided
        // BEFORE the prep CB so the refresh can be encoded into it.
        let sharedLight = SIMD3<Float>(sim.params.lightDirX, sim.params.lightDirY, sim.params.lightDirZ)
        var directional: SIMD3<Float>? = nil
        switch o.renderLight {
        case "growth":
            directional = sharedLight
            print("render light: directional (shared growth light)")
        case "unbiased":
            directional = nil
            print("render light: unbiased omnidirectional (paper default)")
        case "auto":
            directional = sim.params.growthMode == 1 ? sharedLight : nil
            print("render light: auto -> \(directional != nil ? "directional (light-growth mode)" : "unbiased omnidirectional")")
        default:
            print("--render-light '\(o.renderLight)' unknown (auto | growth | unbiased)")
            exit(1)
        }
        if let cb = queue.makeCommandBuffer() {
            sim.refreshRadii(cb)
            sim.encodeGridRebuild(cb)
            if directional != nil {
                sim.encodeIlluminationRefresh(cb)
            }
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { print("render prep CB ERROR: \(e)") }
        }
        sim.syncCellCountFromDevice()   // host mirror consistent (GPU idle)

        var settings = RayRenderSettings()
        switch o.style {
        case "solid": settings.style = .solid
        case "xray": settings.style = .xray
        case "foam": settings.style = .foam
        case "foam-xray": settings.style = .foamXray
        default:
            print("--style '\(o.style)' unknown (solid | xray | foam | foam-xray); using solid")
            settings.style = .solid
        }
        // M5 unified light: decided above (before the prep CB, so the
        // illumination fractions can be refreshed in it). X-ray has no
        // lighting — unaffected.
        settings.directionalLight = directional
        settings.width = o.width
        settings.height = o.height
        settings.samples = o.samples
        // V2.1 render knobs (--shadows/--xray-color; the CG_* env overrides
        // exist for look iteration and bench A/B without touching the CLI
        // contract — goldens record their explicit values).
        settings.shadows = max(o.shadows, 0)
        settings.xrayColored = o.xrayColor
        // V2.1b foam knobs (env-driven look iteration; the meta records them).
        settings.foamOpacity = max(min(foamOpacityFromEnv(), 1), 0.02)
        settings.foamLayers = max(1, min(foamLayersFromEnv(), 8))
        settings.foamWall = max(foamWallFromEnv(), 0)
        if let es = ProcessInfo.processInfo.environment["CG_SHADOW_SAMPLES"], let n = Int(es) {
            settings.shadowSamples = max(0, min(n, 64))
        }
        if o.tiled {
            print("--tiled: tiling is always applied (RayRenderer.tileSize = \(RayRenderer.tileSize)²); the flag is an explicit marker")
        }
        // Tuning overrides (look iteration without rebuilds; goldens record
        // their explicit values in the meta, so these never affect checks).
        func envFloat(_ key: String, _ current: Float) -> Float {
            ProcessInfo.processInfo.environment[key].flatMap { Float($0) } ?? current
        }
        settings.aoRadius = envFloat("CG_AO_RADIUS", settings.aoRadius)
        settings.aoStrength = envFloat("CG_AO_STRENGTH", settings.aoStrength)
        settings.aoFloor = envFloat("CG_AO_FLOOR", settings.aoFloor)
        settings.xrayExposure = envFloat("CG_XRAY_EXPOSURE", settings.xrayExposure)
        settings.xrayGamma = envFloat("CG_XRAY_GAMMA", settings.xrayGamma)
        settings.densityScale = envFloat("CG_DENSITY_SCALE", settings.densityScale)
        settings.profileSupport = envFloat("CG_PROFILE_SUPPORT", settings.profileSupport)

        // M6 stereo pairs (paper §6): two renders with the eye shifted along
        // the camera RIGHT axis by ±interocular/2 (parallel-axis stereo — no
        // toe-in, so there is no keystone distortion), composited side by
        // side (left eye on the left). Separation is a FRACTION of the scene
        // scale (the bounding radius), default 0.03 — the paper only says
        // "cameras with an interocular separation".
        if o.stereo {
            settings.stereo = true
            settings.interocular = o.interocular
        }

        // Same framing convention as --screenshot: camera distance from the
        // structure's max |P| (OrbitCamera yaw/pitch/fov defaults shared).
        var maxAbs: Float = 0
        for p_ in sim.readPositions() { maxAbs = max(maxAbs, simd_length(p_)) }
        settings.camera.distance = maxAbs * 2.6

        let renderer = RayRenderer(device: sim.device, queue: queue, sim: sim)
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = renderer.render(settings: settings)
        let wall = CFAbsoluteTimeGetCurrent() - t0
        let styleNames = ["solid", "xray", "foam", "foam-xray"]
        print("Render: \(styleNames[Int(min(settings.style.rawValue, 3))]) \(result.width)x\(result.height)\(o.stereo ? " (stereo pair, interocular \(o.interocular)x scene)" : "") samples=\(settings.samples) cells=\(sim.cellCount)  wall \(String(format: "%.2f", wall))s  gpu \(String(format: "%.2f", result.gpuSeconds))s")
        if let savePath = o.stateSave {
            do {
                try sim.saveState(to: savePath)
                print("state saved: \(savePath) (\(sim.cellCount) cells, CGST v3 full state)")
            } catch {
                print("state save FAILED: \(error)")
                exit(1)
            }
        }
        if ProcessInfo.processInfo.environment["CGDBG"] != nil {
            renderer.debugVerify(bytes: result.bytes, settings: settings, pixelCount: 64)
        }
        do {
            let dir = (o.renderPath! as NSString).deletingLastPathComponent
            if !dir.isEmpty { try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true) }
            try writePNG(result.bytes, width: result.width, height: result.height, to: o.renderPath!)
            print("Render written: \(o.renderPath!)")
        } catch {
            print("Render failed: \(error)")
            exit(1)
        }
        if ProcessInfo.processInfo.environment["CGDBG"] != nil {
            // Debug: raw tile-assembled BGRA bytes next to the PNG (pixel-order
            // experiments without a PNG decoder).
            let rawPath = o.renderPath! + ".raw"
            result.bytes.withUnsafeBytes { src in
                FileManager.default.createFile(atPath: rawPath,
                                               contents: Data(bytes: src.baseAddress!, count: result.bytes.count))
            }
            print("CGDBG: raw BGRA bytes written to \(rawPath)")
        }
        exit(0)
    }

    // MARK: - M6 animation export / V2.6 video export (shared core)
    //
    // Paper §6: "High definition animations taken by rendering images at
    // equally spaced time intervals".

    /// V2.6: the deterministic offscreen video-export core shared by the CLI
    /// --anim and the GUI "Export video…" button (factored from the M6
    /// routine — ONE implementation, two callers; never duplicate the frame
    /// loop). The sim must already sit in the state to export (GPU idle):
    /// the CLI caller resets it from the CLI config first, the GUI caller
    /// exports the LIVE state. `stepsPerFrame` steps run between frames;
    /// every frame is re-framed (legacy max|P|·2.6, or the V2.6
    /// follow-centroid + circle choreography) and rendered with the OFFLINE
    /// ray tracer; frames land as frame_0000.png … and ONE H.264 .mp4 is
    /// assembled (the M6 .mov container was REPLACED in V2.6: .mp4 plays
    /// everywhere and keeps a single code path; codec H.264 + 30 fps
    /// cadence unchanged).
    struct VideoExportConfig {
        var dir: String
        var frames: Int
        var stepsPerFrame: Int
        var fps: Int32 = 30
        var settings: RayRenderSettings
        // V2.6 choreography — deterministic per-frame increments, snap-on-start
        // (frame 0 centers the form; smoothing takes over from frame 1).
        var followCentroid = false
        var followSmoothing: Float = 0.15
        var orbitRate: Float = 0
        /// GUI progress hook (frames done, total). Nil = CLI printing only.
        var onFrame: ((Int, Int) -> Void)? = nil
        /// V2.7: polled at the TOP of every frame iteration — true ends the
        /// frame loop early; whatever frames completed are still assembled
        /// into the movie afterwards (a stopped export ALWAYS yields a video
        /// of what it rendered). Nil = never abort (the CLI --anim path).
        /// Read on the export thread, written from the GUI's Stop button —
        /// the codebase's plain-var cross-thread pattern (same as the
        /// Renderer's isPlaying/stepsPerFrame); the worst case is one extra
        /// rendered frame.
        var shouldStop: (() -> Bool)? = nil
        /// V2.7 GUI live preview: called after each frame PNG lands (on the
        /// export thread) with the PNG path + (done, total). The GUI decodes
        /// + downscales OFF-main and shows the latest frame in the panel —
        /// no file-manager hunting. Nil = CLI printing only.
        var onPreview: ((_ pngPath: String, _ done: Int, _ total: Int) -> Void)? = nil
    }

    struct VideoExportResult {
        var dir: String
        var frames: Int
        var moviePath: String?      // nil = assembly failed (the PNG sequence IS complete)
        var totalWall: Double
        var totalGpu: Double
        /// Camera-choreography (centroid + framing + orbit) host cost, ms/frame.
        var choreographyMsPerFrame: Double
        /// Wall time spent in the AVFoundation encode (movie only).
        var movieEncodeWall: Double
        /// V2.7: true when shouldStop ended the loop before the requested
        /// frame count — the movie covers `frames` frames only (partial
        /// export; the GUI says so in the status line).
        var stopped: Bool = false
        /// V2.7: true when the export sim hit its maxCells capacity mid-run
        /// (division is a device-side hard cap; growth stalls from that
        /// frame on — the capacity estimate under-sized; see
        /// SimViewModel.exportCapacityEstimate).
        var capacityCapped: Bool = false
    }

    /// The shared frame loop. Throws on PNG/prep failures; a failed movie
    /// assembly is reported (print + moviePath nil) but NOT thrown — the
    /// complete PNG sequence remains usable (the M6 fallback contract).
    static func exportAnimation(queue: MTLCommandQueue, sim: CellSim,
                                config: VideoExportConfig) throws -> VideoExportResult {
        let cfg = config
        let frames = max(1, cfg.frames)
        let stepsPerFrame = max(0, cfg.stepsPerFrame)
        try FileManager.default.createDirectory(atPath: cfg.dir, withIntermediateDirectories: true)
        print("video export: \(frames) frames every \(stepsPerFrame) steps, style \(cfg.settings.style), \(cfg.settings.width)x\(cfg.settings.height), into \(cfg.dir)")

        let renderer = RayRenderer(device: sim.device, queue: queue, sim: sim)
        var settings = cfg.settings
        var smoothedTarget: SIMD3<Float>? = nil   // V2.6 follow state (frame 0 snaps)
        var totalGpu = 0.0
        var choreoSeconds = 0.0
        let totalWall = CFAbsoluteTimeGetCurrent()
        let pngPaths = (0..<frames).map { "\(cfg.dir)/frame_\(String(format: "%04d", $0)).png" }
        var rendered = 0
        var stopped = false
        var capacityCapped = false

        for f in 0..<frames {
            // V2.7: the stop flag is checked BETWEEN frames — a frame in
            // flight always completes (and lands in the video).
            if let stop = cfg.shouldStop, stop() {
                stopped = true
                print("video export: stop requested — finishing with the \(rendered) frames rendered so far")
                break
            }
            if f > 0 && stepsPerFrame > 0 {
                runSteps(queue: queue, sim: sim, count: stepsPerFrame)
            }
            // Render prep at the exact current count: fresh radii + ONE grid
            // rebuild + (directional light) refreshed illumination verdicts.
            if let cb = queue.makeCommandBuffer() {
                sim.refreshRadii(cb)
                sim.encodeGridRebuild(cb)
                if settings.directionalLight != nil {
                    sim.encodeIlluminationRefresh(cb)
                }
                cb.commit()
                cb.waitUntilCompleted()
                if let e = cb.error { print("export frame \(f) prep CB ERROR: \(e)") }
            }
            sim.syncCellCountFromDevice()
            // V2.7 capacity watchdog: the export sim's maxCells is an
            // ESTIMATE (SimViewModel.exportCapacityEstimate). Division is a
            // device-side hard cap, so hitting it stalls growth (no crash);
            // warn once so the status line can say the export is capacity-
            // limited. The CLI --anim path runs on the caller's sim (its own
            // --max-cells choice) and gets the same honest warning.
            if !capacityCapped && sim.cellCount >= sim.maxCells {
                capacityCapped = true
                print("video export WARNING: export sim reached its maxCells capacity (\(sim.maxCells)) — growth stalls from here")
            }

            // Camera: legacy framing or the V2.6 choreography — both are
            // pure functions of the frame index (deterministic exports).
            let tChoreo = CFAbsoluteTimeGetCurrent()
            if cfg.followCentroid {
                // Follow-centroid: target = smoothed LIVE-cell centroid;
                // distance frames on the CENTROID (max live distance from
                // it) — max |P| from the ORIGIN would overestimate once the
                // form grows/drifts away from its birthplace.
                let (c, radius) = sim.liveCentroidAndRadius()
                let a = min(max(cfg.followSmoothing, 0), 1)
                smoothedTarget = smoothedTarget.map { $0 + (c - $0) * a } ?? c
                settings.camera.target = smoothedTarget!
                settings.camera.distance = radius * 2.6
            } else {
                // Legacy framing (M6): distance from max |P| over the LIVE
                // cells, target = origin. Dead slots (V2.5) hold stale
                // positions and are excluded — for death-free runs this is
                // byte-identical to the pre-V2.6 loop over all cells.
                var maxAbs: Float = 0
                let pos = sim.positionsBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
                let dead = sim.deadFlagsPointer
                for i in 0..<sim.cellCount where dead[i] == 0 {
                    maxAbs = max(maxAbs, simd_length(pos[i]))
                }
                settings.camera.distance = maxAbs * 2.6
            }
            if cfg.orbitRate != 0 { settings.camera.yaw += cfg.orbitRate }
            choreoSeconds += CFAbsoluteTimeGetCurrent() - tChoreo

            let t0 = CFAbsoluteTimeGetCurrent()
            let result = renderer.render(settings: settings)
            let wall = CFAbsoluteTimeGetCurrent() - t0
            totalGpu += result.gpuSeconds
            do {
                try writePNG(result.bytes, width: result.width, height: result.height, to: pngPaths[f])
            } catch {
                throw NSError(domain: "anim", code: 10,
                              userInfo: [NSLocalizedDescriptionKey: "frame \(f) PNG: \(error)"])
            }
            rendered = f + 1
            cfg.onPreview?(pngPaths[f], rendered, frames)
            cfg.onFrame?(rendered, frames)
            print("frame \(f)/\(frames - 1) t=\(f * stepsPerFrame) cells=\(sim.cellCount)  wall \(String(format: "%.2f", wall))s  gpu \(String(format: "%.2f", result.gpuSeconds))s")
        }

        // Assemble the H.264 movie (default 30 fps; frames at equally spaced
        // step intervals = the paper's "equally spaced time intervals").
        // V2.7: assemble from the frames ACTUALLY rendered — an aborted
        // export still yields a complete video of its prefix (encode AFTER
        // the frame loop: a stop then only truncates the frame list, the
        // writer lifecycle stays single-path).
        let movPath = "\(cfg.dir)/animation.mp4"
        var moviePath: String? = nil
        var encodeWall = 0.0
        if rendered > 0 {
            do {
                let tEnc = CFAbsoluteTimeGetCurrent()
                let url = try assembleMovie(frames: Array(pngPaths[0..<rendered]),
                                            width: settings.width, height: settings.height,
                                            fps: cfg.fps, to: movPath)
                encodeWall = CFAbsoluteTimeGetCurrent() - tEnc
                moviePath = url.path
            } catch {
                // Honest fallback: the PNG sequence is complete; say what failed.
                print("MP4 assembly FAILED (\(error)) — the PNG sequence in \(cfg.dir) is complete; assemble manually e.g.:\n  ffmpeg -framerate \(cfg.fps) -i \(cfg.dir)/frame_%04d.png -c:v libx264 -pix_fmt yuv420p \(cfg.dir)/animation.mp4")
            }
        } else {
            print("video export: no frames rendered — no movie assembled")
        }
        let total = CFAbsoluteTimeGetCurrent() - totalWall
        print("video export done: \(rendered)/\(frames) frames\(moviePath != nil ? " + \(moviePath!)" : "")  total wall \(String(format: "%.2f", total))s  render gpu \(String(format: "%.2f", totalGpu))s  choreography \(rendered > 0 ? String(format: "%.2f", choreoSeconds * 1000 / Double(rendered)) : "n/a") ms/frame  movie encode \(String(format: "%.2f", encodeWall))s")
        return VideoExportResult(dir: cfg.dir, frames: rendered, moviePath: moviePath,
                                 totalWall: total, totalGpu: totalGpu,
                                 choreographyMsPerFrame: rendered > 0 ? choreoSeconds * 1000 / Double(rendered) : 0,
                                 movieEncodeWall: encodeWall,
                                 stopped: stopped, capacityCapped: capacityCapped)
    }

    /// --anim DIR --anim-steps N --anim-frames M [--follow-centroid]
    /// [--orbit-rate F]: grow the configured state and render a frame every
    /// N steps with the OFFLINE ray tracer (style / size / samples from
    /// --style/--width/--height/--samples) into DIR as frame_0000.png …,
    /// then assemble ONE H.264 .mp4 (AVFoundation, 30 fps). Frame 0 is the
    /// initial state; frame f is at step f·N. Legacy camera: distance
    /// re-framed per frame (max |P| × 2.6, the shared convention).
    /// --follow-centroid follows the LIVE cells' smoothed centroid instead
    /// (distance framed on the centroid), --orbit-rate F circles at F
    /// radians per rendered frame. Both are V2.6 OPT-IN headless.
    private static func runAnimation(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        let dir = o.animDir!
        print("animation: \(max(1, o.animFrames)) frames every \(max(0, o.animSteps)) steps, style \(o.style), \(o.width)x\(o.height), into \(dir)")

        // Same deterministic setup as --render (seeded jitter).
        let (base, links) = Icosphere.generate(subdivisions: o.subdiv, spacing: sim.params.linkRestLength)
        let positions = o.jitter > 0 ? seededJitter(base, radius: o.jitter, seed: o.seed) : base
        sim.reset(positions: positions, links: links)
        applyTypingIfRequested(sim: sim, o: o)   // V2.3: typed animations

        var settings = RayRenderSettings()
        switch o.style {
        case "solid": settings.style = .solid
        case "xray": settings.style = .xray
        case "foam": settings.style = .foam
        case "foam-xray": settings.style = .foamXray
        default:
            print("--style '\(o.style)' unknown (solid | xray | foam | foam-xray); using solid")
            settings.style = .solid
        }
        settings.width = o.width
        settings.height = o.height
        settings.samples = o.samples
        // V2.1: the animation honours the same render knobs as --render.
        settings.shadows = max(o.shadows, 0)
        settings.xrayColored = o.xrayColor
        // V2.1b foam knobs (env-driven, as in --render).
        settings.foamOpacity = max(min(foamOpacityFromEnv(), 1), 0.02)
        settings.foamLayers = max(1, min(foamLayersFromEnv(), 8))
        settings.foamWall = max(foamWallFromEnv(), 0)
        // M6: the animation honours --render-light exactly like --render
        // (auto = directional only in light-growth mode). For paper-style
        // artefacts pass --render-light unbiased.
        switch o.renderLight {
        case "growth":
            settings.directionalLight = SIMD3<Float>(sim.params.lightDirX, sim.params.lightDirY, sim.params.lightDirZ)
        case "unbiased":
            settings.directionalLight = nil
        case "auto":
            settings.directionalLight = sim.params.growthMode == 1
                ? SIMD3<Float>(sim.params.lightDirX, sim.params.lightDirY, sim.params.lightDirZ)
                : nil
        default:
            print("--render-light '\(o.renderLight)' unknown (auto | growth | unbiased)")
            exit(1)
        }

        do {
            let result = try exportAnimation(queue: queue, sim: sim, config: VideoExportConfig(
                dir: dir, frames: o.animFrames, stepsPerFrame: o.animSteps,
                settings: settings,
                followCentroid: o.followCentroid,
                followSmoothing: o.followSmoothing,
                orbitRate: o.orbitRate))
            // The final grown state (post-frames) — saved even when the movie
            // assembly failed (the PNG sequence is the primary artefact).
            if let savePath = o.stateSave {
                do {
                    try sim.saveState(to: savePath)
                    print("state saved: \(savePath) (\(sim.cellCount) cells, CGST v3 full state)")
                } catch {
                    print("state save FAILED: \(error)")
                    exit(1)
                }
            }
            _ = result
        } catch {
            print("animation FAILED: \(error)")
            exit(1)
        }
        exit(0)
    }

    /// Encode a numbered PNG sequence into an H.264 .mp4 via AVFoundation
    /// (AVAssetWriter + pixel-buffer adaptor; BGRA in, yuv420p out). Works
    /// headless — no GUI/app bundle required. V2.6: the container is .mp4
    /// (the M6 .mov was REPLACED — universal playback, one code path; the
    /// H.264 codec and the frame cadence are unchanged).
    /// V2.7: the output extension is FORCED to .mp4 and the actual written
    /// URL is returned — the writer's fileType and the on-disk name must
    /// agree, and the V2.6 user report of a ".m4v" showed the name/path
    /// plumbing deserves a hard guarantee. (Investigation: no .m4v write
    /// path exists anywhere in this code — grep across history — the writer
    /// writes exactly the URL it is given with fileType .mp4; most plausibly
    /// the .m4v the user saw was QuickTime's MPEG-4 video document type
    /// (same container family, public.mpeg-4 UTI) shown for the .mp4 file,
    /// or a save/export through QuickTime, which defaults to .m4v. Forcing
    /// the extension here means the on-disk name ALWAYS matches the
    /// documented .mp4 container no matter the caller.)
    @discardableResult
    private static func assembleMovie(frames: [String], width: Int, height: Int,
                                      fps: Int32, to path: String) throws -> URL {
        var url = URL(fileURLWithPath: path)
        if url.pathExtension.lowercased() != "mp4" {
            url = url.deletingPathExtension().appendingPathExtension("mp4")
        }
        try? FileManager.default.removeItem(at: url)   // overwrite a previous run
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: max(2_000_000, width * height * 4),   // ~4 bit/px
            ],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
                                                           sourcePixelBufferAttributes: attrs)
        guard writer.canAdd(input) else {
            throw NSError(domain: "anim", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter rejected the video input"])
        }
        writer.add(input)
        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "anim", code: 2)
        }
        writer.startSession(atSourceTime: .zero)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw NSError(domain: "anim", code: 3)
        }
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
            | CGBitmapInfo.byteOrder32Little.rawValue   // BGRA, same as the PNG writer

        for (f, pngPath) in frames.enumerated() {
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: pngPath) as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                throw NSError(domain: "anim", code: 4,
                              userInfo: [NSLocalizedDescriptionKey: "cannot decode \(pngPath)"])
            }
            var pb: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                                      attrs as CFDictionary, &pb) == kCVReturnSuccess,
                  let pixelBuffer = pb else {
                throw NSError(domain: "anim", code: 5,
                              userInfo: [NSLocalizedDescriptionKey: "pixel buffer alloc failed"])
            }
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                                   width: width, height: height, bitsPerComponent: 8,
                                   bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                                   space: colorSpace, bitmapInfo: bitmapInfo) {
                ctx.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.005)
            }
            let time = CMTime(value: CMTimeValue(f), timescale: fps)
            guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                throw writer.error ?? NSError(domain: "anim", code: 6,
                                              userInfo: [NSLocalizedDescriptionKey: "append failed at frame \(f)"])
            }
        }
        input.markAsFinished()
        let sema = DispatchSemaphore(value: 0)
        writer.finishWriting { sema.signal() }
        sema.wait()
        guard writer.status == .completed else {
            throw writer.error ?? NSError(domain: "anim", code: 7,
                                          userInfo: [NSLocalizedDescriptionKey: "writer status \(writer.status.rawValue)"])
        }
        return url
    }

    // MARK: - Bench

    /// Lightweight timing sweep: fixed steps per subdivision level, CPU wall
    /// clock + GPU time (commandBuffer.gpuStartTime/gpuEndTime). With
    /// --bench-growth, division is enabled and initial/final cell counts are
    /// reported (the per-step cost then covers the growing count).
    private static func runBench(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        let levels = [4, 5, 6, 7, 8]   // 2 562 … 655 362 cells
        let steps = o.benchSteps
        if o.benchGrowth {
            sim.params.divisionEnabled = 1
            print("bench: \(steps) steps/level, jitter 0, division ON (food \(sim.params.foodIncrement), threshold \(sim.params.splitThreshold))")
        } else {
            sim.params.divisionEnabled = 0
            print("bench: \(steps) steps/level, jitter 0, division off (forces only)")
        }
        print("repulsion: roi \(sim.params.roi), strength \(sim.params.repulsionStrength) (\(sim.params.repulsionStrength > 0 ? "grid rebuilt every step" : "off"))")
        let modeNames = ["uniform", "light", "rde"]
        print("growth mode: \(modeNames[Int(min(sim.params.growthMode, 2))]) (food \(sim.params.foodIncrement), threshold \(sim.params.splitThreshold))")
        // V2.8: the active drives (the enum is only the preset-setter; the
        // drives decide the chemical-block schedule this sweep measures).
        print("drives: U=\(sim.params.growthUniform) L=\(sim.params.growthLight) R=\(sim.params.growthRde) (march \(sim.params.growthLight > 0 || (sim.params.growthRde > 0 && !(sim.params.growthMode == 2 && sim.params.growthUniform == 0 && sim.params.growthLight == 0)) ? "scheduled" : "off"), rdeStep \(sim.params.growthRde > 0 || sim.params.growthMode == 2 ? "scheduled" : "off"))")
        // V2.5: the death block's cost arm is a --bench run with --death and a
        // huge --death-steps (the counter + decide passes run every step, no
        // deaths execute — the delta vs the death-off baseline is the block
        // cost at matched counts).
        print("death: \(sim.params.deathEnabled != 0 ? "on (threshold \(sim.params.deathThreshold), window \(sim.params.deathSteps))" : "off")")
        // V2.3: typing is applied per level (after each reset) when requested
        // — the typed sweep measures the decide-kernel rule matrix + the
        // foodStep rate classification against the untyped baseline.
        if o.typePoles != nil {
            print("typing: axis \(o.typePoles!) t1 \(o.typeT1) t2 \(o.typeT2), phaseX \(sim.params.phaseSwitchX), lamelRate \(sim.params.lamelRate), dalRate \(sim.params.dalRate)")
        }
        // V2.4: the differentiation step-range is honoured BEFORE the timed
        // window (see the per-level hook below): at = the phase start,
        // until = growth resume (0 = the phase stays ON through the timed
        // window — that arm measures the differentiation-pass cost at
        // matched counts).
        if o.differentiateAt > 0 {
            print("differentiation phase: at \(o.differentiateAt), until \(o.differentiateUntil == 0 ? "∞ (stays on through the timed window)" : "\(o.differentiateUntil)"), thresholds v>=\(sim.params.typeCapV) cap / v<=\(sim.params.typeStemV) stem, typeMode \(sim.params.typeMode == 1 ? "region" : "band")")
        }
        print("cells            | ms/step cpu | ms/step gpu | steps/s")
        for level in levels {
            let (positions, links) = Icosphere.generate(subdivisions: level, spacing: sim.params.linkRestLength)
            sim.reset(positions: positions, links: links)
            applyTypingIfRequested(sim: sim, o: o)   // V2.3: typed bench levels
            // V2.4 phase hook: enter the differentiation phase, run its
            // (untimed) portion until --differentiate-until, then resume
            // growth for the timed window. With until == 0 no untimed steps
            // run and the phase covers the timed window instead (the cost
            // arm: typeTransition + one extra tint pass per step, division
            // frozen — the cells column stays flat in a --bench-growth run).
            if o.differentiateAt > 0 {
                sim.params.differentiationPhase = 1
                sim.syncParams()
                let phaseSteps = o.differentiateUntil > o.differentiateAt
                    ? o.differentiateUntil - o.differentiateAt : 0
                if phaseSteps > 0 {
                    if let cb = queue.makeCommandBuffer() {
                        for _ in 0..<phaseSteps { sim.step(cb) }
                        cb.commit()
                        cb.waitUntilCompleted()
                    }
                    sim.params.differentiationPhase = 0
                    sim.syncParams()
                }
            }
            let initial = sim.cellCount

            // warmup
            if let cb = queue.makeCommandBuffer() {
                for _ in 0..<10 { sim.step(cb) }
                cb.commit()
                cb.waitUntilCompleted()
            }

            let t0 = CFAbsoluteTimeGetCurrent()
            var gpuTime = 0.0
            // CG_BENCH_SPLIT diagnostics (V2.5): the measured gpu window
            // (gpuStartTime..gpuEndTime) spans the WHOLE batch — if the CPU
            // cannot encode faster than the GPU executes, the GPU DRAINS the
            // queue mid-batch and the window silently includes encoder-starve
            // idle time. Splitting encode / wait wall time per batch exposes
            // that: gpu-window >> gpu-busy means the number is CPU-bound and
            // NOT a GPU cost. Gated by env (CG_FOAM_* precedent: bench A/B
            // without a CLI contract change); default output is unchanged.
            var encodeTime = 0.0
            var waitTime = 0.0
            var batchGpu: [Double] = []
            var done = 0
            while done < steps {
                let batch = min(10, steps - done)
                let tEnc0 = CFAbsoluteTimeGetCurrent()
                guard let cb = queue.makeCommandBuffer() else { break }
                for _ in 0..<batch { sim.step(cb) }
                cb.commit()
                encodeTime += CFAbsoluteTimeGetCurrent() - tEnc0
                let tWait0 = CFAbsoluteTimeGetCurrent()
                cb.waitUntilCompleted()
                waitTime += CFAbsoluteTimeGetCurrent() - tWait0
                let g = cb.gpuEndTime - cb.gpuStartTime
                gpuTime += g
                batchGpu.append(g / Double(batch) * 1000)
                done += batch
            }
            let dt = CFAbsoluteTimeGetCurrent() - t0
            sim.syncCellCountFromDevice(allowGridGrowth: true)
            let cells = o.benchGrowth ? "\(initial) -> \(sim.cellCount)" : "\(initial)"
            print("\(cells) | \(String(format: "%.3f", dt / Double(steps) * 1000)) | \(String(format: "%.3f", gpuTime / Double(steps) * 1000)) | \(String(format: "%.0f", Double(steps) / dt))")
            print("   \(sim.gridMemoryDescription())")
            if ProcessInfo.processInfo.environment["CG_BENCH_BATCHES"] != nil {
                // Per-batch gpu window (ms/step): exposure of GPU-scheduler
                // interference from other GPU clients (WindowServer etc.).
                // A level whose batches disagree by >2x is contention-tainted;
                // the MIN batch is the best exclusive-GPU estimate.
                let sorted = batchGpu.sorted()
                let median = sorted[sorted.count / 2]
                print("   batches: \(batchGpu.map { String(format: "%.3f", $0) }.joined(separator: " ")) | median \(String(format: "%.3f", median)) min \(String(format: "%.3f", sorted.first!))")
            }
            if ProcessInfo.processInfo.environment["CG_BENCH_SPLIT"] != nil {
                print("   split: encode \(String(format: "%.3f", encodeTime / Double(steps) * 1000)) + wait \(String(format: "%.3f", waitTime / Double(steps) * 1000)) = wall \(String(format: "%.3f", (encodeTime + waitTime) / Double(steps) * 1000)) ms/step (gpu window \(String(format: "%.3f", gpuTime / Double(steps) * 1000)))")
            }
        }
        exit(0)
    }

    // MARK: - Deterministic helpers (M3)

    /// Seeded jitter: same distribution shape as SimViewModel.randomInBall
    /// (rejection sampling in a ball) but driven by a SplitMix64 stream, so
    /// headless runs (selftest, lockstep, goldens) are reproducible.
    static func seededJitter(_ base: [SIMD3<Float>], radius: Float, seed: UInt32) -> [SIMD3<Float>] {
        var s = UInt64(seed) &+ 0x9E3779B97F4A7C15
        func nextU64() -> UInt64 {
            s = s &+ 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        func u01() -> Float { Float(nextU64() >> 40) * (1.0 / Float(1 << 24)) }
        return base.map { p in
            while true {
                let v = SIMD3<Float>(u01() * 2 - 1, u01() * 2 - 1, u01() * 2 - 1)
                if simd_length_squared(v) <= 1 { return p + v * radius }
            }
        }
    }

    // MARK: - CPU-lockstep golden reference (--lockstep)

    /// Deviation stats between two per-cell vector fields.
    private static func deviationStats(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>])
        -> (mean: Float, max: Float, nan: Bool) {
        precondition(a.count == b.count, "field length mismatch")
        var sum: Float = 0
        var mx: Float = 0
        var nan = false
        for i in 0..<a.count {
            let d = simd_length(a[i] - b[i])
            if d.isNaN { nan = true; continue }
            sum += d
            mx = max(mx, d)
        }
        return a.count > 0 ? (sum / Float(a.count), mx, nan) : (0, 0, nan)
    }

    /// Tolerances for the GPU-vs-CPU comparison. The GPU sums the repulsion
    /// contributions in spatial-hash bin order, the CPU brute force in cell
    /// index order; float addition is not associative and the GPU's rsqrt/
    /// sqrt differ from the CPU's by an ulp, so bitwise equality is
    /// impossible. What the gates assert is that the deviation stays at
    /// float-noise scale instead of growing into semantic divergence (wrong
    /// candidate sets, wrong clamps). Calibrated on the reference run
    /// (subdiv 3, jitter 0.2, roi 1.6, strength 0.1, 300 steps — M3 report):
    /// gates sit well above the measured values and far below any visually
    /// meaningful deviation (1e-2 = 1% of linkRestLength).
    private static let lockstepProbeMeanGate: Float = 1e-6
    private static let lockstepProbeMaxGate: Float = 1e-4
    private static let lockstepMeanGate: Float = 1e-4
    private static let lockstepMaxGate: Float = 1e-2

    /// Runs the exact GPU step semantics on the CPU (brute-force repulsion,
    /// index-order summation) in lockstep with the GPU, division OFF, and
    /// compares per-cell positions at every checkpoint. PASS/FAIL + exit code.
    private static func runLockstep(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        // <= ~2.6k cells: brute-force repulsion is O(N^2) per CPU step
        // (subdiv 4 = 2562 cells ~ 6.6M pair checks/step, seconds for the
        // 50-step trajectory — acceptable for a validation mode).
        let subdiv = min(o.subdiv, 4)
        sim.params.divisionEnabled = 0
        // M5: force the uniform growth mode. The chemical kernels write only
        // the nutrient/u/v buffers (never positions/links), so the trajectory
        // comparison would stay valid even in light/rde mode — but the force
        // + repulsion semantics under test must be the M1-M4 path, unchanged.
        // V2.8: setGrowthMode also sets the drive preset (U=1 here — moot:
        // division is off, foodStep never runs in lockstep).
        sim.setGrowthMode(0)
        // V2.5: force death off — the CPU reference models neither division
        // nor death (both change topology; the comparison is index-labelled).
        sim.params.deathEnabled = 0
        sim.syncParams()

        let (base, links) = Icosphere.generate(subdivisions: subdiv, spacing: sim.params.linkRestLength)
        let positions = o.jitter > 0 ? seededJitter(base, radius: o.jitter, seed: o.seed) : base
        print("lockstep: subdiv \(subdiv) (\(positions.count) cells), division OFF, jitter \(o.jitter) (seeded), repulsion roi \(o.roi) strength \(o.repulsion), \(o.steps) steps")
        print("gates: probe mean < \(lockstepProbeMeanGate), max < \(lockstepProbeMaxGate); trajectory mean < \(lockstepMeanGate), max < \(lockstepMaxGate) (x linkRestLength = \(sim.params.linkRestLength))")
        sim.reset(positions: positions, links: links)

        var cpu = CPUReference(positions: positions, links: links,
                               linkRestLength: sim.params.linkRestLength,
                               springFactor: sim.params.springFactor,
                               planarFactor: sim.params.planarFactor,
                               bulgeFactor: sim.params.bulgeFactor,
                               roi: sim.params.roi,
                               repulsionStrength: sim.params.repulsionStrength)

        var allPass = true

        // --- Probe: grid rebuild + accumulate only; compare the collision
        // field cell-by-cell against brute force BEFORE any feedback
        // amplification. Isolates spatial-hash correctness from trajectory chaos.
        if let cb = queue.makeCommandBuffer() {
            sim.encodeRepulsionProbe(cb)
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { print("probe CB ERROR: \(e)") }   // STAP 0: fault check
        }
        let cpuCol = cpu.collisionOffsets()
        let probe = deviationStats(sim.readCollisionOffsets(), cpuCol)
        print("probe t=0 (collisionOffset): mean |Δ| = \(String(format: "%.3e", probe.mean)), max |Δ| = \(String(format: "%.3e", probe.max))")
        if ProcessInfo.processInfo.environment["CELLDBG"] != nil {
            // --- STAP 0: params ground truth (what the GPU sees vs Swift) ---
            let echoed = sim.dbgEchoParams()
            print("dbg paramEcho (GPU)  : gridBits=(\(echoed[0]),\(echoed[1]),\(echoed[2])) maxValence=\(echoed[3]) cellCount=\(echoed[4]) maxCells=\(echoed[5]) div=\(echoed[6]) seed=\(echoed[7]) roi=\(Float(bitPattern: echoed[8])) strength=\(Float(bitPattern: echoed[9]))")
            print("dbg swift (SimParams): gridBits=(\(sim.params.gridBitsX),\(sim.params.gridBitsY),\(sim.params.gridBitsZ)) maxValence=\(sim.params.maxValence) cellCount=\(sim.params.cellCount) maxCells=\(sim.params.maxCells) div=\(sim.params.divisionEnabled) seed=\(sim.params.rngSeed) roi=\(sim.params.roi) strength=\(sim.params.repulsionStrength)")

            // --- STAP 1: hash fingerprint for cell 0 (reads the probe CB's
            // post-accumulate grid state in its own waited CB) ---
            let fp = sim.dbgQueryFingerprint()
            let bx = Int(sim.params.gridBitsX), by = Int(sim.params.gridBitsY), bz = Int(sim.params.gridBitsZ)
            func hsh(_ c: SIMD3<Int>) -> Int {
                (c.x & ((1 << bx) - 1)) | ((c.y & ((1 << by) - 1)) << bx) | ((c.z & ((1 << bz) - 1)) << (bx + by))
            }
            let P0 = cpu.positions[0]
            let bc0 = SIMD3<Int>(Int(floor(P0.x / sim.params.roi)), Int(floor(P0.y / sim.params.roi)), Int(floor(P0.z / sim.params.roi)))
            var expected: [Int] = []
            for dz in -1...1 { for dy in -1...1 { for dx in -1...1 { expected.append(hsh(bc0 &+ SIMD3<Int>(dx, dy, dz))) } } }
            let gpuHashes = Array(fp[0..<27])
            let mismatches = zip(gpuHashes, expected).filter { Int($0.0) != $0.1 }.count
            print("dbg fingerprint cell 0: unique=\(fp[27]) visits=\(fp[28]) inRoiPassing=\(fp[29]) hashMismatches vs swift=\(mismatches)")
        }
        if probe.nan || probe.mean > lockstepProbeMeanGate || probe.max > lockstepProbeMaxGate {
            print("probe FAIL (gates mean < \(lockstepProbeMeanGate), max < \(lockstepProbeMaxGate))")
            allPass = false
        } else {
            print("probe PASS")
        }

        // --- Trajectory: advance both implementations in sync, compare every
        // 50 steps. Gate on the WORST checkpoint (deviation grows with chaos;
        // the final step is usually the binding one).
        let checkEvery = 50
        var worstMean: Float = 0
        var worstMax: Float = 0
        var done = 0
        while done < o.steps {
            let batch = min(checkEvery, o.steps - done)
            runSteps(queue: queue, sim: sim, count: batch)
            for _ in 0..<batch { cpu.step() }
            done += batch
            let stats = deviationStats(sim.readPositions(), cpu.positions)
            worstMean = max(worstMean, stats.mean)
            worstMax = max(worstMax, stats.max)
            print("t=\(done)  mean |ΔP| = \(String(format: "%.3e", stats.mean)), max |ΔP| = \(String(format: "%.3e", stats.max))")
            if stats.nan {
                print("t=\(done)  FAIL: NaN in deviation")
                allPass = false
            }
        }
        let meanOK = worstMean <= lockstepMeanGate
        let maxOK = worstMax <= lockstepMaxGate
        print("worst checkpoint: mean |ΔP| = \(String(format: "%.3e", worstMean)) (gate \(lockstepMeanGate)) \(meanOK ? "PASS" : "FAIL")")
        print("worst checkpoint: max  |ΔP| = \(String(format: "%.3e", worstMax)) (gate \(lockstepMaxGate)) \(maxOK ? "PASS" : "FAIL")")
        if !meanOK || !maxOK { allPass = false }
        print(allPass ? "LOCKSTEP PASS" : "LOCKSTEP FAIL")
        exit(allPass ? 0 : 1)
    }

    // MARK: - Golden-image regression (--golden-capture / --golden-check)

    static let goldenDir = "docs/golden"

    /// A deterministic render configuration. Division is forced OFF for
    /// goldens: daughter slot allocation is GPU-schedule-dependent (M2
    /// report), so a growing run cannot yet be reproduced pixel-exactly —
    /// documented limitation, revisit if division ever becomes
    /// order-deterministic. Everything else (seeded jitter, fixed camera
    /// derived from max |P|, fixed light, fixed resolution) makes the
    /// rendered pixels a pure function of this config.
    ///
    /// M4: `render != nil` selects the offline ray tracer (Shaders.raytrace)
    /// instead of the raster impostor path, and re-enables division (grown
    /// golden) — see RenderGoldenSettings. The canonical string and the meta
    /// only gain the render keys when render != nil, so the M3 goldens'
    /// configHash values are unchanged.
    struct GoldenConfig {
        var steps: Int
        var subdiv: Int
        var jitter: Float
        var spring: Float
        var planar: Float
        var bulge: Float
        var roi: Float
        var repulsion: Float
        var seed: UInt32
        var width: Int
        var height: Int
        var meanAbsTolerance: Float   // mean |pixel diff| over all bytes, [0,1] units
        var render: RenderGoldenSettings? = nil

        static let defaultTolerance: Float = 2.0 / 255.0

        var canonical: String {
            var s = "subdiv=\(subdiv);jitter=\(fmt(jitter));spring=\(fmt(spring));planar=\(fmt(planar));"
            + "bulge=\(fmt(bulge));roi=\(fmt(roi));repulsion=\(fmt(repulsion));seed=\(seed);"
            + "steps=\(steps);width=\(width);height=\(height);tolerance=\(fmt(meanAbsTolerance))"
            if let r = render {
                s += ";style=\(r.style);samples=\(r.samples);maxCells=\(r.maxCells);div=\(r.division ? 1 : 0);"
                + "aoRadius=\(fmt(r.aoRadius));aoStrength=\(fmt(r.aoStrength));aoFloor=\(fmt(r.aoFloor));"
                + "xrayExposure=\(fmt(r.xrayExposure));xrayGamma=\(fmt(r.xrayGamma));densityScale=\(fmt(r.densityScale));"
                + "profileSupport=\(fmt(r.profileSupport));snapshot=\(r.snapshot ? 1 : 0);"
                + "aaSeed=\(r.seed)"
                if let v = r.v21 {
                    s += ";v21=1;growth=\(v.growth);food=\(fmt(v.food));threshold=\(fmt(v.threshold));"
                    + "rdeF=\(fmt(v.rdeF));rdeK=\(fmt(v.rdeK));rdeDu=\(fmt(v.rdeDu));rdeDv=\(fmt(v.rdeDv));"
                    + "rdeBase=\(fmt(v.rdeBase));rdeGain=\(fmt(v.rdeGain));renderLight=\(v.renderLight);"
                    + "edges=\(fmt(v.edges));edgeWidth=\(fmt(v.edgeWidth));shadows=\(fmt(v.shadows));"
                    + "shadowSamples=\(v.shadowSamples);xrayColor=\(v.xrayColor ? 1 : 0);lightSoft=\(fmt(v.lightSoft))"
                }
                if let v = r.v22 {
                    s += ";v22=1;tint=\(v.tint);heatScale=\(fmt(v.heatScale));pulseDecay=\(fmt(v.pulseDecay))"
                }
                if let b = r.v21b {
                    s += ";v21b=1;foamOpacity=\(fmt(b.foamOpacity));foamLayers=\(b.foamLayers);"
                        + "foamWall=\(fmt(b.foamWall));radiusScale=\(fmt(b.radiusScale))"
                }
            }
            return s
        }
        private func fmt(_ v: Float) -> String { String(format: "%.6f", v) }
    }

    /// Render-specific settings for ray-traced goldens (M4). Every knob that
    /// affects the rendered pixels is recorded, so --golden-check reproduces
    /// the capture even if the RayRenderSettings defaults drift later.
    /// snapshot: the grown structure is NOT reproducible run-to-run (daughter
    /// slot allocation is GPU-schedule-dependent, M2 report — measured: two
    /// identical growth runs differ in 43.7% of rendered pixels), so grown
    /// goldens store the final state (positions + links + linkCounts) in
    /// <name>.state and --golden-check LOADS it instead of re-growing. The
    /// growth config keys stay in the meta as provenance.
    struct RenderGoldenSettings {
        var style: String        // "solid" | "xray"
        var samples: Int
        var maxCells: Int        // growth cap (grown goldens run division ON)
        var division: Bool
        var aoRadius: Float
        var aoStrength: Float
        var aoFloor: Float
        var xrayExposure: Float
        var xrayGamma: Float
        var densityScale: Float
        var profileSupport: Float
        var snapshot: Bool
        var seed: UInt32           // AA jitter seed (every pixel-affecting knob is recorded)
        // --- V2.1 (nil for pre-V2.1 metas: their canonical string and hash
        // must stay byte-identical, so the new keys are only appended when a
        // V2.1 capture declares any non-default V2.1 knob) ---
        var v21: RenderGoldenV21? = nil
        // --- V2.2 (same rule: only appended when a V2.2 knob is declared) ---
        var v22: RenderGoldenV22? = nil
        // --- V2.1b (same rule; the retired edges/edgeWidth keys stay in the
        // V2.1 block's FORMAT for the old metas but are no longer rendered) ---
        var v21b: RenderGoldenV21b? = nil
    }

    /// V2.1 golden knobs: the new render flags + the growth config the
    /// capture grew with. Snapshot replays make the growth keys pure
    /// provenance (the CHECK loads the stored state); the render light and
    /// lightSoft must be reproduced exactly (they shape the image).
    struct RenderGoldenV21 {
        var growth: String        // uniform | light | rde (capture-time growth mode)
        var food: Float
        var threshold: Float
        var rdeF: Float
        var rdeK: Float
        var rdeDu: Float
        var rdeDv: Float
        var rdeBase: Float
        var rdeGain: Float
        var renderLight: String   // auto | growth | unbiased
        var edges: Float
        var edgeWidth: Float
        var shadows: Float
        var shadowSamples: Int
        var xrayColor: Bool
        var lightSoft: Float
    }

    /// V2.2 golden knobs: tint mode + the heat/pulse tuning. Same provenance
    /// pattern as the v21 block — a snapshot replay carries these values in
    /// the state file's raw SimParams (the state is authoritative); the keys
    /// record what the capture grew/rendered with.
    struct RenderGoldenV22 {
        var tint: String          // auto | none | chemical | food | pulse
        var heatScale: Float      // 0 = auto default
        var pulseDecay: Float
    }

    /// V2.1b golden knobs: the foam look + the render radius scale. The
    /// snapshot's raw SimParams carry radiusScale (belt-and-braces: the check
    /// re-applies the meta value after the load so the radii refresh uses it
    /// even if a state were re-saved with a different scale).
    struct RenderGoldenV21b {
        var foamOpacity: Float    // per-face composite opacity (solid foam + impostor)
        var foamLayers: Int       // max composited faces per ray (ray tracer)
        var foamWall: Float       // foam-xray wall-spike gain
        var radiusScale: Float    // computeRadii scale (0.7 = the foam recommendation)
    }
    /// Default pulse decay (must match SimParams.pulseDecay — legacy metas
    /// without a v22 block load with this).
    static let defaultPulseDecay: Float = 0.95

    /// FNV-1a 64-bit over the canonical config string (the meta's config hash).
    static func configHash(_ s: String) -> String {
        var h: UInt64 = 0xcbf29ce484222325
        for b in s.utf8 { h = (h ^ UInt64(b)) &* 0x100000001b3 }
        return String(format: "%016llx", h)
    }

    private static func goldenMetaPath(_ name: String) -> String { "\(goldenDir)/\(name).meta" }
    private static func goldenPNGPath(_ name: String) -> String { "\(goldenDir)/\(name).png" }
    private static func goldenStatePath(_ name: String) -> String { "\(goldenDir)/\(name).state" }

    // MARK: State snapshots (grown goldens; see RenderGoldenSettings.snapshot)
    //
    // M6: the CGST writer/reader now lives on CellSim (saveState(to:) /
    // loadState(from:)) — one implementation for the CLI, the goldens and the
    // GUI. Golden .state files written from M6 on are v3 (full state); the
    // stored M4 goldens are v1 and still load (CellSim.loadState).

    private static func saveGoldenState(_ path: String, sim: CellSim) throws {
        try sim.saveState(to: path)
    }

    /// Golden .state replay: CellSim.loadState handles v1/v2/v3 (the stored
    /// M4 goldens are v1, M6 captures are v3).
    private static func loadGoldenState(_ path: String, sim: CellSim) throws {
        _ = try sim.loadState(from: path)
    }

    private static func writeGoldenMeta(_ name: String, cfg: GoldenConfig) throws {
        let f = { (v: Float) in String(format: "%.6f", v) }
        var lines = [
            "# CellularGrowth golden image metadata (deterministic config, division OFF)",
            "name=\(name)",
            "rendererVersion=\(cfg.render != nil ? "2" : "1")",
            "configHash=\(configHash(cfg.canonical))",
            "subdiv=\(cfg.subdiv)",
            "jitter=\(f(cfg.jitter))",
            "spring=\(f(cfg.spring))",
            "planar=\(f(cfg.planar))",
            "bulge=\(f(cfg.bulge))",
            "roi=\(f(cfg.roi))",
            "repulsion=\(f(cfg.repulsion))",
            "seed=\(cfg.seed)",
            "steps=\(cfg.steps)",
            "width=\(cfg.width)",
            "height=\(cfg.height)",
            "meanAbsTolerance=\(f(cfg.meanAbsTolerance))",
        ]
        if let r = cfg.render {
            lines.append("style=\(r.style)")
            lines.append("samples=\(r.samples)")
            lines.append("maxCells=\(r.maxCells)")
            lines.append("division=\(r.division ? 1 : 0)")
            lines.append("aoRadius=\(f(r.aoRadius))")
            lines.append("aoStrength=\(f(r.aoStrength))")
            lines.append("aoFloor=\(f(r.aoFloor))")
            lines.append("xrayExposure=\(f(r.xrayExposure))")
            lines.append("xrayGamma=\(f(r.xrayGamma))")
            lines.append("densityScale=\(f(r.densityScale))")
            lines.append("profileSupport=\(f(r.profileSupport))")
            lines.append("snapshot=\(r.snapshot ? 1 : 0)")
            lines.append("aaSeed=\(r.seed)")
            if let v = r.v21 {
                lines.append("v21=1")
                lines.append("growth=\(v.growth)")
                lines.append("food=\(f(v.food))")
                lines.append("threshold=\(f(v.threshold))")
                lines.append("rdeF=\(f(v.rdeF))")
                lines.append("rdeK=\(f(v.rdeK))")
                lines.append("rdeDu=\(f(v.rdeDu))")
                lines.append("rdeDv=\(f(v.rdeDv))")
                lines.append("rdeBase=\(f(v.rdeBase))")
                lines.append("rdeGain=\(f(v.rdeGain))")
                lines.append("renderLight=\(v.renderLight)")
                lines.append("edges=\(f(v.edges))")
                lines.append("edgeWidth=\(f(v.edgeWidth))")
                lines.append("shadows=\(f(v.shadows))")
                lines.append("shadowSamples=\(v.shadowSamples)")
                lines.append("xrayColor=\(v.xrayColor ? 1 : 0)")
                lines.append("lightSoft=\(f(v.lightSoft))")
            }
            if let v = r.v22 {
                lines.append("v22=1")
                lines.append("tint=\(v.tint)")
                lines.append("heatScale=\(f(v.heatScale))")
                lines.append("pulseDecay=\(f(v.pulseDecay))")
            }
            if let b = r.v21b {
                lines.append("v21b=1")
                lines.append("foamOpacity=\(f(b.foamOpacity))")
                lines.append("foamLayers=\(b.foamLayers)")
                lines.append("foamWall=\(f(b.foamWall))")
                lines.append("radiusScale=\(f(b.radiusScale))")
            }
        }
        try lines.joined(separator: "\n").appending("\n").write(toFile: goldenMetaPath(name), atomically: true, encoding: .utf8)
    }

    private static func readGoldenMeta(_ name: String) -> GoldenConfig? {
        guard let text = try? String(contentsOfFile: goldenMetaPath(name), encoding: .utf8) else { return nil }
        var kv: [String: String] = [:]
        for line in text.split(separator: "\n") {
            let l = line.trimmingCharacters(in: .whitespaces)
            if l.hasPrefix("#") || l.isEmpty { continue }
            let parts = l.split(separator: "=", maxSplits: 1)
            if parts.count == 2 { kv[String(parts[0])] = String(parts[1]) }
        }
        func f(_ k: String) -> Float? { kv[k].flatMap { Float($0) } }
        func i(_ k: String) -> Int? { kv[k].flatMap { Int($0) } }
        guard let subdiv = i("subdiv"), let jitter = f("jitter"),
              let spring = f("spring"), let planar = f("planar"), let bulge = f("bulge"),
              let roi = f("roi"), let repulsion = f("repulsion"),
              let seed = i("seed"), let steps = i("steps"),
              let width = i("width"), let height = i("height"),
              let tol = f("meanAbsTolerance") else { return nil }
        // M4 ray-traced goldens carry render keys; legacy metas do not.
        var render: RenderGoldenSettings? = nil
        if let style = kv["style"] {
            guard let samples = i("samples"), let maxCells = i("maxCells"),
                  let division = i("division"),
                  let aoRadius = f("aoRadius"), let aoStrength = f("aoStrength"),
                  let aoFloor = f("aoFloor"), let xrayExposure = f("xrayExposure"),
                  let xrayGamma = f("xrayGamma"), let densityScale = f("densityScale"),
                  let profileSupport = f("profileSupport"), let snapshot = i("snapshot"),
                  let aaSeed = i("aaSeed") else {
                print("golden meta '\(name)': style present but render keys incomplete")
                return nil
            }
            render = RenderGoldenSettings(style: style, samples: samples, maxCells: maxCells,
                                          division: division != 0,
                                          aoRadius: aoRadius, aoStrength: aoStrength,
                                          aoFloor: aoFloor, xrayExposure: xrayExposure,
                                          xrayGamma: xrayGamma, densityScale: densityScale,
                                          profileSupport: profileSupport, snapshot: snapshot != 0,
                                          seed: UInt32(aaSeed))
            // V2.1 block: present only when the meta carries the marker.
            if kv["v21"] == "1" {
                let v21 = RenderGoldenV21(
                    growth: kv["growth"] ?? "uniform",
                    food: f("food") ?? 0.01,
                    threshold: f("threshold") ?? 1.0,
                    rdeF: f("rdeF") ?? 0.037,
                    rdeK: f("rdeK") ?? 0.06,
                    rdeDu: f("rdeDu") ?? 0.35,
                    rdeDv: f("rdeDv") ?? 0.15,
                    rdeBase: f("rdeBase") ?? 0.25,
                    rdeGain: f("rdeGain") ?? 2.5,
                    renderLight: kv["renderLight"] ?? "auto",
                    edges: f("edges") ?? 0,
                    edgeWidth: f("edgeWidth") ?? 0.12,
                    shadows: f("shadows") ?? 1.0,
                    shadowSamples: i("shadowSamples") ?? 4,
                    xrayColor: (i("xrayColor") ?? 0) != 0,
                    lightSoft: f("lightSoft") ?? 0.35)
                render!.v21 = v21
            }
            // V2.2 block: tint mode + heat/pulse tuning.
            if kv["v22"] == "1" {
                render!.v22 = RenderGoldenV22(tint: kv["tint"] ?? "auto",
                                              heatScale: f("heatScale") ?? 0,
                                              pulseDecay: f("pulseDecay") ?? defaultPulseDecay)
            }
            // V2.1b block: foam look + render radius scale.
            if kv["v21b"] == "1" {
                render!.v21b = RenderGoldenV21b(
                    foamOpacity: f("foamOpacity") ?? 0.5,
                    foamLayers: i("foamLayers") ?? 6,
                    foamWall: f("foamWall") ?? 0.25,
                    radiusScale: f("radiusScale") ?? 0.5)
            }
        }
        return GoldenConfig(steps: steps, subdiv: subdiv, jitter: jitter, spring: spring,
                            planar: planar, bulge: bulge, roi: roi, repulsion: repulsion,
                            seed: UInt32(seed), width: width, height: height,
                            meanAbsTolerance: tol, render: render)
    }

    /// Run a golden config and render it offscreen; returns the RGBA bytes.
    /// Legacy configs (render == nil) use the raster impostor path, unchanged
    /// from M3. Ray-traced configs (render != nil, M4): with `loadSnapshot`
    /// the stored state is loaded (deterministic --golden-check replay — the
    /// growth itself is not run-to-run reproducible); otherwise the config is
    /// grown (capture).
    private static func runGoldenRender(_ name: String, _ cfg: GoldenConfig, loadSnapshot: Bool, queue: MTLCommandQueue, sim: CellSim) -> [UInt8] {
        // M5: goldens are uniform-mode by construction (growth-mode keys are
        // not in the meta format; the paper gates for the growth modes use
        // --screenshot/--render instead). Explicitly reset every M5 field so
        // ambient CLI flags cannot leak into a golden capture/check: growth
        // mode off, light/rde coupling defaults, and the light direction at
        // the pre-M5 impostor default (0.55, 0.75, 0.4) — the M3 raster
        // goldens were captured with that shading light, and the unified
        // light now reads it from SimParams. V2.8: setGrowthMode also resets
        // the three drives to the uniform preset so ambient --growth-* flags
        // cannot leak either.
        sim.setGrowthMode(0)
        sim.params.lightDirX = 0.55
        sim.params.lightDirY = 0.75
        sim.params.lightDirZ = 0.4
        sim.params.lightDiffusionRate = 0.02
        sim.params.lightNutrient = 0.05
        sim.params.lightDecay = 0.02
        sim.params.lightBase = 0.15
        sim.params.lightInterval = 8
        sim.params.lightSoft = 0.35
        sim.params.rdeF = 0.037
        sim.params.rdeK = 0.06
        sim.params.rdeDu = 0.35
        sim.params.rdeDv = 0.15
        sim.params.rdeBase = 0.25
        sim.params.rdeGain = 2.5
        // V2.2 tint knobs: reset to the auto defaults so ambient CLI flags
        // cannot leak into a golden (the meta keys / state file carry them).
        sim.params.tintMode = 0
        sim.params.heatScale = 0
        sim.params.pulseDecay = defaultPulseDecay
        // V2.4 differentiation knobs: same leak-guard (typing off by default —
        // the type-0 contract). A fresh SimParams is the source of truth so
        // the defaults here cannot drift from the struct's.
        let v24 = SimParams()
        sim.params.typeMode = v24.typeMode
        sim.params.differentiationPhase = v24.differentiationPhase
        sim.params.typeCapV = v24.typeCapV
        sim.params.typeStemV = v24.typeStemV
        // V2.5 death knobs: same leak-guard — goldens are DEATH-FREE (the
        // death block changes topology, which no golden tolerates).
        sim.params.deathEnabled = v24.deathEnabled
        sim.params.deathThreshold = v24.deathThreshold
        sim.params.deathSteps = v24.deathSteps
        guard let r = cfg.render else {
            sim.params.divisionEnabled = 0
            sim.params.springFactor = cfg.spring
            sim.params.planarFactor = cfg.planar
            sim.params.bulgeFactor = cfg.bulge
            sim.params.roi = cfg.roi
            sim.params.repulsionStrength = cfg.repulsion
            sim.params.rngSeed = cfg.seed
            sim.syncParams()

            let (base, links) = Icosphere.generate(subdivisions: cfg.subdiv, spacing: sim.params.linkRestLength)
            let positions = cfg.jitter > 0 ? seededJitter(base, radius: cfg.jitter, seed: cfg.seed) : base
            sim.reset(positions: positions, links: links)
            runSteps(queue: queue, sim: sim, count: cfg.steps)

            // Fresh radii for every cell before rendering (same as --screenshot).
            if let cb = queue.makeCommandBuffer() {
                sim.refreshRadii(cb)
                cb.commit()
                cb.waitUntilCompleted()
            }
            return renderToBytes(sim: sim, queue: queue, width: cfg.width, height: cfg.height)
        }

        if loadSnapshot && r.snapshot {
            do {
                try loadGoldenState(goldenStatePath(name), sim: sim)
            } catch {
                print("golden render FAILED: \(error)")
                exit(1)
            }
        } else {
            sim.params.divisionEnabled = r.division ? 1 : 0
            sim.params.springFactor = cfg.spring
            sim.params.planarFactor = cfg.planar
            sim.params.bulgeFactor = cfg.bulge
            sim.params.roi = cfg.roi
            sim.params.repulsionStrength = cfg.repulsion
            sim.params.rngSeed = cfg.seed
            sim.params.maxCells = UInt32(r.maxCells)
            // V2.1 capture-time growth config (provenance keys from the meta;
            // a snapshot replay ignores these — the state file carries it).
            if let v = r.v21 {
                sim.setGrowthMode(["uniform", "light", "rde"].firstIndex(of: v.growth).map(UInt32.init) ?? 0)
                sim.params.foodIncrement = v.food
                sim.params.splitThreshold = v.threshold
                sim.params.rdeF = v.rdeF
                sim.params.rdeK = v.rdeK
                sim.params.rdeDu = v.rdeDu
                sim.params.rdeDv = v.rdeDv
                sim.params.rdeBase = v.rdeBase
                sim.params.rdeGain = v.rdeGain
            }
            // V2.2 capture-time tint config (same provenance pattern).
            if let v = r.v22 {
                sim.params.tintMode = tintModeCode(v.tint)
                sim.params.heatScale = max(v.heatScale, 0)
                sim.params.pulseDecay = min(max(v.pulseDecay, 0), 0.9999)
            }
            sim.syncParams()

            let (base, links) = Icosphere.generate(subdivisions: cfg.subdiv, spacing: sim.params.linkRestLength)
            let positions = cfg.jitter > 0 ? seededJitter(base, radius: cfg.jitter, seed: cfg.seed) : base
            sim.reset(positions: positions, links: links)
            runSteps(queue: queue, sim: sim, count: cfg.steps)
        }

        // M4 ray-traced golden: fresh radii + ONE grid rebuild for the final
        // state (the renderer traverses the static hash), then the tile loop.
        // V2.1: the render light for the new goldens (legacy goldens stay
        // unbiased); directional renders refresh the per-cell illumination
        // FRACTIONS in the same prep CB (after the grid rebuild — the march
        // traverses it). lightSoft drives both the fractional illumination
        // and the shadow-ray cone.
        sim.params.lightSoft = r.v21?.lightSoft ?? 0.35
        sim.syncParams()
        // V2.1b: the meta's render radius scale (belt-and-braces over the
        // snapshot's own raw params) — BEFORE the prep CB so refreshRadii
        // computes every radius with it.
        if let b = r.v21b {
            sim.params.radiusScale = min(max(b.radiusScale, 0.05), 1)
            sim.syncParams()
        }
        var directional: SIMD3<Float>? = nil
        if let v = r.v21, v.renderLight == "growth" {
            directional = simd_normalize(SIMD3<Float>(sim.params.lightDirX,
                                                      sim.params.lightDirY,
                                                      sim.params.lightDirZ))
        }
        if let cb = queue.makeCommandBuffer() {
            sim.refreshRadii(cb)
            sim.encodeGridRebuild(cb)
            if directional != nil {
                sim.encodeIlluminationRefresh(cb)
            }
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { print("golden render prep CB ERROR: \(e)") }
        }
        sim.syncCellCountFromDevice()

        // M6: the interactive ADDITIVE X-ray (raster impostor path) can be
        // locked by its own golden — style "xray-impostor" selects it instead
        // of the ray tracer. V2.1b: "foam-impostor" locks the interactive
        // FOAM composite the same way (the sorted premultiplied chambers).
        // The ao*/xray*/density meta keys are ray-tracer knobs (recorded for
        // meta compatibility; unused by the impostor).
        if r.style == "xray-impostor" {
            return renderToBytes(sim: sim, queue: queue, width: cfg.width, height: cfg.height,
                                 style: "xray")
        }
        if r.style == "foam-impostor" {
            return renderToBytes(sim: sim, queue: queue, width: cfg.width, height: cfg.height,
                                 style: "foam",
                                 foamOpacity: r.v21b?.foamOpacity ?? 0.5)
        }

        var settings = RayRenderSettings()
        switch r.style {
        case "xray": settings.style = .xray
        case "foam": settings.style = .foam
        case "foam-xray": settings.style = .foamXray
        default: settings.style = .solid
        }
        settings.directionalLight = directional   // legacy goldens: nil (unbiased look)
        settings.width = cfg.width
        settings.height = cfg.height
        settings.samples = r.samples
        settings.aoRadius = r.aoRadius
        settings.aoStrength = r.aoStrength
        settings.aoFloor = r.aoFloor
        settings.xrayExposure = r.xrayExposure
        settings.xrayGamma = r.xrayGamma
        settings.densityScale = r.densityScale
        settings.profileSupport = r.profileSupport
        settings.seed = r.seed
        // V2.1 golden render knobs (recorded in the meta; all defaults for
        // legacy goldens — byte-identical path). The RETIRED edges/edgeWidth
        // keys are parsed for meta provenance only and are no longer applied
        // (the foam chamber walls are the wall visualization now).
        if let v = r.v21 {
            settings.shadows = v.shadows
            settings.shadowSamples = v.shadowSamples
            settings.xrayColored = v.xrayColor
        }
        // V2.1b foam knobs (only present on foam-era metas).
        if let b = r.v21b {
            settings.foamOpacity = b.foamOpacity
            settings.foamLayers = b.foamLayers
            settings.foamWall = b.foamWall
        }
        // Same framing convention as --render/--screenshot.
        var maxAbs: Float = 0
        for p in sim.readPositions() { maxAbs = max(maxAbs, simd_length(p)) }
        settings.camera.distance = maxAbs * 2.6

        let renderer = RayRenderer(device: sim.device, queue: queue, sim: sim)
        let result = renderer.render(settings: settings)
        return result.bytes
    }

    private static func runGoldenCapture(_ name: String, queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        // Config from the CLI flags (parseOptions applies golden-appropriate
        // defaults for flags that were not given); recorded fully in the meta
        // so --golden-check never depends on the flags being repeated.
        // An explicit --style selects the M4 ray tracer and records all its
        // tuning knobs; without it, the M3 raster goldens stay byte-compatible.
        var render: RenderGoldenSettings? = nil
        if o.styleGiven {
            let d = RayRenderSettings()
            // V2.1: attach the new knob block ONLY when any of it is
            // non-default — legacy metas' canonical strings (and config
            // hashes) must stay byte-identical. The retired edges/edgeWidth
            // flags no longer exist; the foam chamber walls are the wall
            // visualization.
            let shadowSamples = ProcessInfo.processInfo.environment["CG_SHADOW_SAMPLES"]
                .flatMap { Int($0) } ?? d.shadowSamples
            let v21Active = o.shadows != 1.0 || o.xrayColor || o.lightSoft != 0.35
                || o.renderLight != "auto"
                || o.growth != "uniform" || o.food != 0.01 || o.threshold != 1.0
                || o.rdeF != 0.037 || o.rdeK != 0.06 || o.rdeDu != 0.35
                || o.rdeDv != 0.15 || o.rdeBase != 0.25 || o.rdeGain != 2.5
            let v21: RenderGoldenV21? = v21Active
                ? RenderGoldenV21(growth: o.growth, food: o.food, threshold: o.threshold,
                                  rdeF: o.rdeF, rdeK: o.rdeK, rdeDu: o.rdeDu, rdeDv: o.rdeDv,
                                  rdeBase: o.rdeBase, rdeGain: o.rdeGain,
                                  renderLight: o.renderLight, edges: 0,
                                  edgeWidth: 0.12, shadows: o.shadows,
                                  shadowSamples: shadowSamples, xrayColor: o.xrayColor,
                                  lightSoft: o.lightSoft)
                : nil
            // V2.2 knob block, same only-when-non-default rule.
            let v22Active = o.tint != "auto" || o.heatScaleGiven || o.pulseDecayGiven
            let v22: RenderGoldenV22? = v22Active
                ? RenderGoldenV22(tint: o.tint, heatScale: o.heatScale, pulseDecay: o.pulseDecay)
                : nil
            // V2.1b knob block: written when the style is a foam style OR an
            // explicit radius scale was given (either shapes the rendered
            // pixels; the check re-applies the recorded values).
            let foamStyle = o.style == "foam" || o.style == "foam-xray" || o.style == "foam-impostor"
            let v21bActive = o.radiusScaleGiven || foamStyle
            let v21b: RenderGoldenV21b? = v21bActive
                ? RenderGoldenV21b(foamOpacity: max(min(foamOpacityFromEnv(), 1), 0.02),
                                   foamLayers: max(1, min(foamLayersFromEnv(), 8)),
                                   foamWall: max(foamWallFromEnv(), 0),
                                   radiusScale: min(max(o.radiusScale, 0.05), 1))
                : nil
            render = RenderGoldenSettings(style: o.style, samples: o.samples,
                                          maxCells: o.maxCells, division: o.division,
                                          aoRadius: d.aoRadius, aoStrength: d.aoStrength,
                                          aoFloor: d.aoFloor, xrayExposure: d.xrayExposure,
                                          xrayGamma: d.xrayGamma, densityScale: d.densityScale,
                                          profileSupport: d.profileSupport,
                                          snapshot: true, seed: d.seed, v21: v21, v22: v22,
                                          v21b: v21b)
        }
        let cfg = GoldenConfig(steps: o.steps, subdiv: o.subdiv, jitter: o.jitter,
                               spring: o.spring, planar: o.planar, bulge: o.bulge,
                               roi: o.roi, repulsion: o.repulsion, seed: o.seed,
                               width: o.styleGiven ? o.width : 800,
                               height: o.styleGiven ? o.height : 600,
                               meanAbsTolerance: GoldenConfig.defaultTolerance,
                               render: render)
        print("golden capture '\(name)': \(cfg.canonical)")
        print("configHash=\(configHash(cfg.canonical))")
        let bytes = runGoldenRender(name, cfg, loadSnapshot: false, queue: queue, sim: sim)
        do {
            try FileManager.default.createDirectory(atPath: goldenDir, withIntermediateDirectories: true)
            try writePNG(bytes, width: cfg.width, height: cfg.height, to: goldenPNGPath(name))
            if cfg.render?.snapshot == true {
                try saveGoldenState(goldenStatePath(name), sim: sim)
                print("state snapshot written: \(goldenStatePath(name)) (\(sim.cellCount) cells)")
            }
            try writeGoldenMeta(name, cfg: cfg)
            print("golden written: \(goldenPNGPath(name)) + \(goldenMetaPath(name))")
        } catch {
            print("golden capture failed: \(error)")
            exit(1)
        }
        exit(0)
    }

    private static func runGoldenCheck(_ name: String, queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        // Re-render from the META's config (not the CLI flags), so stored
        // goldens stay valid when CLI defaults drift.
        guard let cfg = readGoldenMeta(name) else {
            print("golden check FAILED: no readable meta at \(goldenMetaPath(name)) (capture first)")
            exit(1)
        }
        guard let raw = try? String(contentsOfFile: goldenMetaPath(name), encoding: .utf8),
              let storedHash = raw.split(separator: "\n")
                  .first(where: { $0.hasPrefix("configHash=") })
                  .map({ String($0.dropFirst("configHash=".count)) }),
              storedHash == configHash(cfg.canonical) else {
            print("golden check FAILED: meta configHash mismatch — meta edited or unparsable")
            exit(1)
        }
        print("golden check '\(name)': \(cfg.canonical)")
        let bytes = runGoldenRender(name, cfg, loadSnapshot: true, queue: queue, sim: sim)
        guard let (stored, w, h) = readPNG(goldenPNGPath(name)), w == cfg.width, h == cfg.height else {
            print("golden check FAILED: cannot read \(goldenPNGPath(name)) at \(cfg.width)x\(cfg.height)")
            exit(1)
        }
        var sum = 0
        var maxD = 0
        for i in 0..<min(bytes.count, stored.count) {
            let d = abs(Int(bytes[i]) - Int(stored[i]))
            sum += d
            maxD = max(maxD, d)
        }
        let mean = Float(sum) / Float(max(bytes.count, 1)) / 255.0
        let pass = mean <= cfg.meanAbsTolerance
        print("golden check: mean |Δpixel| = \(String(format: "%.5f", mean * 255))/255 (\(String(format: "%.5f", mean))), max = \(maxD)/255, tolerance = \(String(format: "%.5f", cfg.meanAbsTolerance * 255))/255")
        print(pass ? "GOLDEN PASS" : "GOLDEN FAIL")
        exit(pass ? 0 : 1)
    }

    /// Decode a PNG into the same pixel format the writer used (sRGB, BGRA8,
    /// premultiplied-first little-endian) for byte comparison.
    private static func readPNG(_ path: String) -> (bytes: [UInt8], width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let w = img.width
        let h = img.height
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB),
                                  bitmapInfo: CGBitmapInfo(rawValue:
                                      CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue)) else { return nil }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { return nil }
        let ptr = data.assumingMemoryBound(to: UInt8.self)
        return ([UInt8](UnsafeBufferPointer(start: ptr, count: w * h * 4)), w, h)
    }

    // MARK: - Selftest (invariant harness)

    /// Invariant harness, two phases (AGENTS.md validation gates):
    ///
    /// Phase A (control, forces only): the initial ball with division disabled
    /// and jitter applied must settle — strict gate, mean link length within 2%
    /// of linkRestLength.
    ///
    /// Phase B (growth): division enabled, checkpoints every 500 steps with the
    /// exact topological invariants plus a bounded-mean gate. The strict 2%
    /// gate is NOT applied to a mesh that has divided: (a) the daughter is
    /// necessarily born inside the daughter-arc's territory, so every division
    /// pumps local compression in; (b) a stretched patch on a flat
    /// neighbourhood is an exact equilibrium of the paper's force set (bulge
    /// x_r = dotN + sqrt(max(L^2 - t^2, 0)) vanishes there, and symmetric
    /// spring/planar targets average to P) — the sheet has no global
    /// scale-restoring force, so division stretch is permanent (measured:
    /// x1.5 growth => ~7% mean stretch that never relaxes). This is the
    /// paper's own no-repulsion regime (Fig. 5); repulsion (M3) is what
    /// prevents the fold degeneration. Gate: mean <= 1.5 x rest (runaway guard).
    private static func runSelftest(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        let subdiv = max(o.subdiv, 4)          // growing-run scale: 2 562 cells
        let steps = max(o.steps, 3000)
        var allPass = true

        // --- Phase A: control (forces only, strict settle gate) ---
        sim.params.divisionEnabled = 0
        var (base, links) = Icosphere.generate(subdivisions: subdiv, spacing: sim.params.linkRestLength)
        // Seeded jitter: headless runs are reproducible (M3). Same distribution
        // shape as SimViewModel.randomInBall (rejection sampling in a ball).
        let jittered = seededJitter(base, radius: 0.25, seed: sim.params.rngSeed)
        print("selftest phase A (control): subdiv \(subdiv) (\(base.count) cells), division OFF, jitter 0.25 (seeded), 600 steps")
        sim.reset(positions: jittered, links: links)
        runSteps(queue: queue, sim: sim, count: 600)
        let (passA, linesA) = checkInvariants(sim: sim, settled: true, meanLimit: 1.02)
        print("phase A t=600  cells=\(sim.cellCount)  ->  \(passA ? "PASS" : "FAIL")")
        for l in linesA { print("   \(l)") }
        if !passA { allPass = false }

        // --- Phase B: growth (division on, topological invariants exact) ---
        sim.params.divisionEnabled = 1
        (base, links) = Icosphere.generate(subdivisions: subdiv, spacing: sim.params.linkRestLength)
        print("selftest phase B (growth): subdiv \(subdiv) (\(base.count) cells), division ON, food \(sim.params.foodIncrement), threshold \(sim.params.splitThreshold), seed \(sim.params.rngSeed), cap \(sim.maxCells), \(steps) steps")
        sim.reset(positions: base, links: links)

        func checkpoint(_ step: Int) {
            let (pass, lines) = checkInvariants(sim: sim, settled: false, meanLimit: 1.5)
            print("checkpoint t=\(step)  cells=\(sim.cellCount)  ->  \(pass ? "PASS" : "FAIL")")
            for l in lines { print("   \(l)") }
            if !pass { allPass = false }
        }

        checkpoint(0)
        var done = 0
        while done < steps {
            let batch = min(100, steps - done)
            runSteps(queue: queue, sim: sim, count: batch)
            done += batch
            if done % 500 == 0 || done == steps {
                checkpoint(done)
            }
        }

        // --- Phase C: growth under the light mode (M5) ---
        // Division + light nutrient at the default gate values; the invariants
        // must hold under the new food coupling, and the lit/dark nutrient
        // gradient must actually exist (else the mode is a no-op).
        sim.setGrowthMode(1)
        sim.syncParams()
        (base, links) = Icosphere.generate(subdivisions: subdiv, spacing: sim.params.linkRestLength)
        print("selftest phase C (light growth): growthMode light, diffusion \(sim.params.lightDiffusionRate), nutrient \(sim.params.lightNutrient), decay \(sim.params.lightDecay), base \(sim.params.lightBase), \(min(steps, 1500)) steps")
        sim.reset(positions: base, links: links)
        var doneC = 0
        while doneC < min(steps, 1500) {
            let batch = min(100, min(steps, 1500) - doneC)
            runSteps(queue: queue, sim: sim, count: batch)
            doneC += batch
            if doneC % 500 == 0 || doneC == min(steps, 1500) {
                checkpoint(doneC)
            }
        }
        let lightMetrics = lightGradientStats(sim: sim)
        // V2.1 gate recalibration (measured, documented — NOT lowered): the
        // fractional soft light (lightSoft 0.35) compresses the facing/away
        // nutrient ratio by design — partially shaded cells now receive a
        // partial nutrient flux instead of a hard 0. Measured on this exact
        // config: binary (lightSoft 0) ratio ~10.7, fractional ratio 2.07
        // (selftest) / 2.15 (headless, same config). The old gate (> 1.2)
        // was calibrated for the binary verdict's ~10x ratio; at the new
        // operating point it would pass even a badly degraded gradient.
        // Raised to > 1.5: still well below the measured 2.07 (38% margin),
        // and it keeps asserting that a clear lit/dark gradient EXISTS.
        print("   light gradient: facing=\(String(format: "%.3f", lightMetrics.facing)) away=\(String(format: "%.3f", lightMetrics.away)) ratio=\(String(format: "%.2f", lightMetrics.ratio)) \(lightMetrics.ratio > 1.5 ? "PASS" : "FAIL") (gate > 1.5, V2.1 recalibrated for the fractional light)")
        if !(lightMetrics.ratio > 1.5) { allPass = false }

        // --- Phase D: growth under the RDE mode (M5) ---
        // V2.8: setGrowthMode(2) sets the pure legacy rde drive preset
        // (U=0, L=0, R=1) — the shader's pure-legacy condition keeps this
        // phase's coupling UNGATED and bit-exact pre-V2.8.
        sim.setGrowthMode(2)
        sim.syncParams()
        (base, links) = Icosphere.generate(subdivisions: subdiv, spacing: sim.params.linkRestLength)
        print("selftest phase D (rde growth): growthMode rde, F=\(sim.params.rdeF) k=\(sim.params.rdeK) Du=\(sim.params.rdeDu) Dv=\(sim.params.rdeDv), \(min(steps, 1500)) steps")
        sim.reset(positions: base, links: links)
        var doneD = 0
        while doneD < min(steps, 1500) {
            let batch = min(100, min(steps, 1500) - doneD)
            runSteps(queue: queue, sim: sim, count: batch)
            doneD += batch
            if doneD % 500 == 0 || doneD == min(steps, 1500) {
                checkpoint(doneD)
            }
        }
        sim.setGrowthMode(0)
        sim.syncParams()

        // --- Phase E: typed growth (V2.3) — the mushroom rule matrix ------
        // A small paddenstoel-preset run: phase-0 polar typing (vertical
        // axis, t1 -0.4 / t2 0.9 — the preset's tuned values), phase 1 (cap
        // uniform + stem edge-only, voeg dormant) switching to phase 2
        // (voeg-only) after phaseSwitchX executed divisions. Beyond the usual
        // invariants (checkpoint() — which now includes the type-validity
        // line), the phase E gates are:
        //   - Σ per-type counts == V (every cell carries a valid type; the
        //     invariants check covers the codes, this asserts the partition);
        //   - the rule matrix is ACTIVE: daughters inherit the parent's
        //     type, so a type's count can only change when that TYPE
        //     divides — phase 1 must freeze lamel+dal, phase 2 must freeze
        //     stem+cap (checked between consecutive same-phase checkpoints);
        //   - the phase transition is observed EXACTLY ONCE (the device
        //     divCounter is monotonic; it flips at the first step boundary
        //     after reaching phaseSwitchX) and growth actually reaches X.
        let phaseX = 2700
        sim.params.lamelRate = 0.5
        sim.params.dalRate = 0.25
        sim.params.interiorDivider = 10.0
        sim.params.phaseSwitchX = UInt32(phaseX)
        sim.params.divisionEnabled = 1
        sim.syncParams()
        (base, links) = Icosphere.generate(subdivisions: max(o.subdiv, 3), spacing: sim.params.linkRestLength)
        let eSteps = min(steps, 2000)
        print("selftest phase E (typed growth): paddenstoel rule matrix, \(base.count) cells, t1 -0.4 t2 0.9, phaseX \(phaseX), \(eSteps) steps")
        sim.reset(positions: base, links: links)
        sim.applyPolarTypes(axis: SIMD3<Float>(0, 1, 0), t1: -0.4, t2: 0.9)
        var lastP1: [Int]? = nil   // last phase-1 checkpoint's type counts
        var lastP2: [Int]? = nil   // last phase-2 checkpoint's type counts
        var phaseFlips = 0
        var lastPhase = 1
        func typeCheckpoint(_ step: Int) {
            checkpoint(step)   // the usual invariants (incl. type validity)
            let types = sim.readTypes()
            var counts = [Int](repeating: 0, count: 5)
            for t in types where Int(t) < 5 { counts[Int(t)] += 1 }
            let sum = counts.reduce(0, +)
            let sumOK = sum == sim.cellCount
            let div = sim.divisionCounter
            let phase = div >= phaseX ? 2 : 1
            if phase != lastPhase { phaseFlips += 1; lastPhase = phase }
            var matrixOK = true
            if phase == 1 {
                if let lc = lastP1, counts[2] != lc[2] || counts[3] != lc[3] {
                    print("   FAIL: voeg counts changed in phase 1 (lamel \(lc[2])->\(counts[2]), dal \(lc[3])->\(counts[3]))")
                    matrixOK = false
                }
                lastP1 = counts
            } else {
                if let lc = lastP2, counts[1] != lc[1] || counts[4] != lc[4] {
                    print("   FAIL: stem/cap counts changed in phase 2 (stem \(lc[1])->\(counts[1]), cap \(lc[4])->\(counts[4]))")
                    matrixOK = false
                }
                lastP2 = counts
            }
            print("   types: none=\(counts[0]) stem=\(counts[1]) lamel=\(counts[2]) dal=\(counts[3]) cap=\(counts[4]) Σ=\(sum) \(sumOK ? "PASS" : "FAIL")  divisions=\(div) phase=\(phase) flips=\(phaseFlips) \(matrixOK ? "matrix PASS" : "matrix FAIL")")
            if !sumOK || !matrixOK { allPass = false }
        }
        typeCheckpoint(0)
        var doneE = 0
        while doneE < eSteps {
            let batch = min(100, eSteps - doneE)
            runSteps(queue: queue, sim: sim, count: batch)
            doneE += batch
            if doneE % 500 == 0 || doneE == eSteps {
                typeCheckpoint(doneE)
            }
        }
        let finalDiv = sim.divisionCounter
        let transitionOK = phaseFlips == 1 && finalDiv >= phaseX
        print("   phase transition: flips=\(phaseFlips) (gate == 1), final divisions=\(finalDiv) (gate >= \(phaseX)) \(transitionOK ? "PASS" : "FAIL")")
        if !transitionOK { allPass = false }
        // Restore the untyped defaults (the shared sim object runs phases
        // sequentially; nothing after phase E reads the typed state).
        sim.params.phaseSwitchX = 0
        sim.syncParams()

        // --- Phase F: RDE differentiation (V2.4) ---------------------------
        // The DIFFERENTIATION PHASE: division FROZEN (the whole division
        // block is skipped — the cell count and the division counter must not
        // move), the RDE keeps stepping, and typeTransition converts type-0
        // cells (v >= typeCapV -> cap, v <= typeStemV -> stem) with
        // TYPE-MEMORY hysteresis — a typed cell ignores the RDE forever, so
        // the flip stream (new transitions per checkpoint window) must settle
        // to 0. Then growth resumes under the REGION rule matrix (every type
        // divides uniformly within its region; the frozen v field sets the
        // regional rates via the food coupling) — cap regions must grow
        // faster per capita than stem regions, that is the patterned-growth
        // point. Thresholds are the shipped defaults (hardcoded so the
        // selftest contract does not drift with CLI flags).
        // Gates (beyond the usual invariants at every checkpoint):
        //   1. frozen phase: cellCount and divisionCounter constant;
        //   2. the type partition goes 0/V -> mixed (stem AND cap present);
        //   3. the flip stream settles to 0 (type memory — no flicker);
        //   4. type memory holds: only 0 -> 1 / 0 -> 4 transitions ever;
        //   5. regional growth: cellCount grows AND cap per-capita growth >
        //      stem per-capita growth.
        sim.setGrowthMode(2)
        sim.params.typeMode = 1
        sim.params.typeCapV = 0.30
        sim.params.typeStemV = 0.10
        sim.params.phaseSwitchX = 0      // region mode ignores it anyway
        sim.params.divisionEnabled = 0   // the settle stage is a STATIC ball;
                                         // division is enabled exactly when
                                         // the phase starts (below), so the
                                         // frozen gate proves the PHASE (not
                                         // divisionEnabled) freezes division
        sim.syncParams()
        (base, links) = Icosphere.generate(subdivisions: max(o.subdiv, 4), spacing: sim.params.linkRestLength)
        let settleF = 500
        let phaseF = 1000
        let growF = 1000
        print("selftest phase F (RDE differentiation): \(base.count) cells, thresholds v>=\(sim.params.typeCapV) cap / v<=\(sim.params.typeStemV) stem, settle \(settleF) + differentiate \(phaseF) + regional growth \(growF) steps")
        sim.reset(positions: base, links: links)
        runSteps(queue: queue, sim: sim, count: settleF)
        let frozenCount = sim.cellCount
        let frozenDiv = sim.divisionCounter
        sim.params.divisionEnabled = 1
        sim.params.differentiationPhase = 1
        sim.syncParams()
        print("   t=\(settleF): differentiation phase STARTED (frozen: \(frozenCount) cells, \(frozenDiv) divisions)")
        var fPrevTypes: [UInt32]? = nil
        var fCounts: [Int] = []
        var fLastFlips = -1
        var fSawFlips = false
        var fMemoryOK = true
        var fFrozenOK = true
        var doneF = 0
        while doneF < phaseF {
            let batch = min(100, phaseF - doneF)
            runSteps(queue: queue, sim: sim, count: batch)
            doneF += batch
            let t = settleF + doneF
            checkpoint(t)   // the usual invariants (topology static in the phase)
            let types = sim.readTypes()
            var counts = [Int](repeating: 0, count: 5)
            var flips = 0
            if let prev = fPrevTypes {
                for i in 0..<min(prev.count, types.count) {
                    let a = prev[i], b = types[i]
                    if a != b {
                        flips += 1
                        if !(a == 0 && (b == 1 || b == 4)) { fMemoryOK = false }
                    }
                }
                if flips > 0 { fSawFlips = true }
            }
            fPrevTypes = types
            for x in types where Int(x) < 5 { counts[Int(x)] += 1 }
            if sim.cellCount != frozenCount || sim.divisionCounter != frozenDiv {
                fFrozenOK = false
            }
            fCounts = counts
            fLastFlips = flips
            print("   t=\(t) types: none=\(counts[0]) stem=\(counts[1]) cap=\(counts[4])  flips=\(flips)  cells=\(sim.cellCount) divisions=\(sim.divisionCounter) \(sim.cellCount == frozenCount && sim.divisionCounter == frozenDiv ? "frozen PASS" : "FROZEN FAIL")")
        }
        let mixedF = fCounts[1] > 0 && fCounts[4] > 0
        // "Flips settle to ~0": the same gate as the diff-probe — the
        // residual trickle of marginal hoverers (the RDE keeps stepping in
        // the phase) must stay a small fraction of V (measured 0-4 cells per
        // 100-step window at 2562 cells; a flickering pattern would churn
        // O(V) per window).
        let flipGateF = max(1, frozenCount / 200)
        let settledF = fLastFlips <= flipGateF && fSawFlips
        print("   differentiation: mixed \(mixedF ? "PASS" : "FAIL") (stem=\(fCounts[1]) cap=\(fCounts[4])), flips settled \(settledF ? "PASS" : "FAIL") (last window \(fLastFlips), gate <= \(flipGateF)), type memory \(fMemoryOK ? "PASS" : "FAIL"), frozen \(fFrozenOK ? "PASS" : "FAIL")")
        if !(mixedF && settledF && fMemoryOK && fFrozenOK) { allPass = false }

        // Stage 3: regional growth resumes.
        sim.params.differentiationPhase = 0
        sim.syncParams()
        print("   t=\(settleF + phaseF): differentiation phase ENDED — regional growth resumes (typeMode region: every type divides uniformly; the frozen v field sets the regional rates)")
        let startCountsF = fCounts
        var endCountsF = startCountsF
        var doneG = 0
        while doneG < growF {
            let batch = min(100, growF - doneG)
            runSteps(queue: queue, sim: sim, count: batch)
            doneG += batch
            let t = settleF + phaseF + doneG
            if doneG % 500 == 0 || doneG == growF {
                checkpoint(t)
                let types = sim.readTypes()
                var counts = [Int](repeating: 0, count: 5)
                for x in types where Int(x) < 5 { counts[Int(x)] += 1 }
                endCountsF = counts
                let stemF = startCountsF[1] > 0 ? Float(counts[1]) / Float(startCountsF[1]) : 0
                let capF = startCountsF[4] > 0 ? Float(counts[4]) / Float(startCountsF[4]) : 0
                print("   t=\(t) types: none=\(counts[0]) stem=\(counts[1]) cap=\(counts[4]) cells=\(sim.cellCount)  per-capita growth: stem x\(String(format: "%.2f", stemF)) cap x\(String(format: "%.2f", capF))")
            }
        }
        let grewF = sim.cellCount > frozenCount
        let stemFactorF = startCountsF[1] > 0 ? Float(endCountsF[1]) / Float(startCountsF[1]) : 0
        let capFactorF = startCountsF[4] > 0 ? Float(endCountsF[4]) / Float(startCountsF[4]) : 0
        let regionalOK = grewF && capFactorF > stemFactorF
        print("   regional growth: cells \(frozenCount) -> \(sim.cellCount) \(grewF ? "PASS" : "FAIL")  per-capita stem x\(String(format: "%.2f", stemFactorF)) cap x\(String(format: "%.2f", capFactorF)) \(regionalOK ? "PASS (cap > stem)" : "FAIL")")
        if !regionalOK { allPass = false }
        // Restore the defaults (the shared sim object; nothing after phase F).
        sim.setGrowthMode(0)
        sim.params.typeMode = 0
        let v24d = SimParams()
        sim.params.typeCapV = v24d.typeCapV
        sim.params.typeStemV = v24d.typeStemV
        sim.params.differentiationPhase = 0
        sim.syncParams()

        // --- Phase G: starvation death (V2.5) — V rises then falls ---------
        // Light-mode growth + death (the prune dynamics, paper §8: cell death
        // "with appropriate topological changes to the surface"): the lit
        // side grows (V rises) while the shaded side starves — its food
        // ledger climbs too slowly to re-cross deathThreshold within the
        // window — and dies off (V falls). The death splice is the inverse of
        // division and Euler-exact: sum valence = 6(V-2) with V = LIVE cells
        // must hold EXACTLY at every checkpoint (the invariants), and the
        // mesh must stay ONE connected component (the flood-fill invariant —
        // a death splice must never disconnect). Gates beyond the invariants:
        //   1. V strictly rises then strictly falls across the checkpoints
        //      (the pruning overtakes the growth);
        //   2. deaths actually happened (deathCounter > 0).
        sim.setGrowthMode(1)
        sim.params.deathEnabled = 1
        sim.params.deathThreshold = sim.params.splitThreshold
        sim.params.deathSteps = 250
        sim.syncParams()
        (base, links) = Icosphere.generate(subdivisions: max(o.subdiv, 4), spacing: sim.params.linkRestLength)
        let gSteps = min(steps, 4000)
        print("selftest phase G (starvation death): light growth + death, \(base.count) cells, threshold \(sim.params.deathThreshold), window \(sim.params.deathSteps) steps, \(gSteps) steps")
        sim.reset(positions: base, links: links)
        var gPrev: Int? = nil          // previous checkpoint's LIVE count
        var roseG = false, fellG = false
        var gDeaths = 0
        func deathCheckpoint(_ step: Int) {
            checkpoint(step)   // the full invariant set (dead-aware + connectivity)
            let live = sim.liveCount
            if let p = gPrev {
                if live > p { roseG = true }
                if live < p { fellG = true }
            }
            gPrev = live
            gDeaths = sim.deathCounter
            let starve = sim.readStarving()
            let dead = sim.readDeadFlags()
            var mx = 0
            for i in 0..<sim.cellCount where dead[i] == 0 { mx = max(mx, Int(starve[i])) }
            print("   t=\(step) live=\(live) deadSlots=\(sim.deathCount) deaths=\(gDeaths) starving max \(mx)/\(sim.params.deathSteps)")
        }
        deathCheckpoint(0)
        var doneG2 = 0
        while doneG2 < gSteps {
            let batch = min(100, gSteps - doneG2)
            runSteps(queue: queue, sim: sim, count: batch)
            doneG2 += batch
            if doneG2 % 500 == 0 || doneG2 == gSteps { deathCheckpoint(doneG2) }
        }
        let vRiseFallOK = roseG && fellG
        let deathsOK = gDeaths > 0
        print("   death dynamics: V rose \(roseG) then fell \(fellG) (gate: both) \(vRiseFallOK ? "PASS" : "FAIL"), deaths \(gDeaths) (gate > 0) \(deathsOK ? "PASS" : "FAIL")")
        if !vRiseFallOK || !deathsOK { allPass = false }
        // Restore the death-off defaults (the shared sim object).
        sim.params.deathEnabled = 0
        sim.params.deathSteps = 250
        sim.syncParams()

        print(allPass ? "SELFTEST PASS" : "SELFTEST FAIL")
        exit(allPass ? 0 : 1)
    }

    /// Mean nutrient of light-facing vs light-averted cells (N·L > 0.3 / < -0.3).
    /// The light mode's reason to exist: the gradient must be present.
    private static func lightGradientStats(sim: CellSim) -> (facing: Float, away: Float, ratio: Float) {
        let nut = UnsafeBufferPointer(start: sim.nutrientPointer, count: sim.cellCount)
        let nor = UnsafeBufferPointer(start: sim.normalsPointer, count: sim.cellCount)
        let L = simd_normalize(SIMD3<Float>(sim.params.lightDirX, sim.params.lightDirY, sim.params.lightDirZ))
        var fs: Float = 0, fn = 0, as_: Float = 0, an = 0
        for i in 0..<sim.cellCount {
            let d = simd_dot(simd_normalize(nor[i]), L)
            if d > 0.3 { fs += nut[i]; fn += 1 }
            else if d < -0.3 { as_ += nut[i]; an += 1 }
        }
        let f = fn > 0 ? fs / Float(fn) : 0
        let a = an > 0 ? as_ / Float(an) : 0
        return (f, a, a > 1e-6 ? f / a : (f > 0 ? .infinity : 1))
    }

    /// Run `count` steps in batches of 100, syncing the device cell counter
    /// after each completed batch. Headless paths may grow the spatial-hash
    /// table here (no command buffer is in flight — see
    /// CellSim.syncCellCountFromDevice).
    private static func runSteps(queue: MTLCommandQueue, sim: CellSim, count: Int, gridGrowth: Bool = true) {
        var done = 0
        while done < count {
            let batch = min(100, count - done)
            guard let cb = queue.makeCommandBuffer() else { fatalError("no command buffer") }
            for _ in 0..<batch { sim.step(cb) }
            cb.commit()
            cb.waitUntilCompleted()
            done += batch
            sim.syncCellCountFromDevice(allowGridGrowth: gridGrowth)
        }
    }

    /// CPU-side invariant check on the live GPU buffers (GPU must be idle).
    /// Returns per-invariant PASS/FAIL lines. `settled` selects the strict
    /// two-sided settle gate (control phase); otherwise `meanLimit` bounds the
    /// mean link length from above (growing phase; see runSelftest docs).
    /// V2.5: dead-aware — every invariant is computed over the LIVE cells
    /// (dead slots stay allocated with stale data; their linkCounts were
    /// zeroed at death), so Euler's sum valence = 6(V−2) uses V = liveCount,
    /// and a NEW invariant was added: the mesh must be ONE connected
    /// component (CPU flood-fill over the live links — a death splice must
    /// never disconnect the surface).
    private static func checkInvariants(sim: CellSim, settled: Bool, meanLimit: Float) -> (pass: Bool, lines: [String]) {
        let v = sim.cellCount
        guard v > 2 else { return (false, ["FAIL: no cells"]) }
        let maxValence = CellSim.maxValence
        let counts = UnsafeBufferPointer(start: sim.linkCountsPointer, count: v)
        let links = UnsafeBufferPointer(start: sim.linksPointer, count: v * maxValence)
        let pos = UnsafeBufferPointer(start: sim.positionsPointer, count: v)
        let food = UnsafeBufferPointer(start: sim.foodPointer, count: v)
        let dead = UnsafeBufferPointer(start: sim.deadFlagsPointer, count: v)   // V2.5

        var liveN = 0
        var deadWithLinks = 0       // dead slots that kept links (must be 0)
        var sumValence = 0
        var minValence = Int.max
        var maxVal = 0
        var nanCount = 0
        var badFood = 0
        var maxAbs: Float = 0
        var lenSum: Float = 0
        var lenCount: Float = 0

        // Pass 1 (LIVE cells only): ring bounds + duplicate entries, valence
        // range, finite positions/food, link lengths, max |P|. Dead slots:
        // their linkCounts must be 0 (deathExecute zeroes them).
        var seen = [UInt32](repeating: UInt32.max, count: v)
        var ringsOK = true
        for i in 0..<v {
            if dead[i] != 0 {
                if counts[i] != 0 { deadWithLinks += 1 }
                continue
            }
            liveN += 1
            let n = Int(counts[i])
            if n < 3 || n > maxValence { ringsOK = false }
            if n < minValence { minValence = n }
            if n > maxVal { maxVal = n }
            sumValence += n
            let gen = UInt32(i) &+ 1
            let base = i * maxValence
            for k in 0..<n {
                let j = Int(links[base + k])
                if j >= v {
                    ringsOK = false
                    continue
                }
                if dead[j] != 0 { ringsOK = false }   // live ring -> dead slot
                if seen[j] == gen { ringsOK = false }   // duplicate ring entry
                seen[j] = gen
                let d = simd_distance(pos[i], pos[j])
                lenSum += d
                lenCount += 1
            }
            let p = pos[i]
            if p.x.isNaN || p.y.isNaN || p.z.isNaN ||
               p.x.isInfinite || p.y.isInfinite || p.z.isInfinite { nanCount += 1 }
            maxAbs = max(maxAbs, simd_length(p))
            if !food[i].isFinite { badFood += 1 }
        }

        // M5: chemical state stays finite in light/rde modes (uniform leaves
        // the chemical buffers untouched; still checked — harmless).
        var badChem = 0
        if sim.params.growthMode == 1 {
            for i in 0..<v where dead[i] == 0
                && !sim.nutrientPointer[i].isFinite { badChem += 1 }
        }
        if sim.params.growthMode == 2 {
            for i in 0..<v where dead[i] == 0
                && !sim.chemUPointer[i].isFinite { badChem += 1 }
            for i in 0..<v where dead[i] == 0
                && !sim.chemVPointer[i].isFinite { badChem += 1 }
        }
        // V2.2: the pulse buffer is maintained every step (decay pass); a NaN
        // there would leak into the tint (pulse mode blends by it).
        for i in 0..<v where dead[i] == 0 && !sim.pulsePointer[i].isFinite { badChem += 1 }
        // V2.3: every LIVE cell's type must be a valid type code (0-4); the
        // per-type counts then partition the LIVE population by construction.
        // (Dead slots keep stale types — excluded, they are past the rule
        // matrix and the render.)
        var badType = 0
        for i in 0..<v where dead[i] == 0 && sim.cellTypePointer[i] > 4 { badType += 1 }

        // Pass 2 (LIVE cells): bidirectionality — every directed link (i -> j)
        // must have its reverse (j -> i). O(links * max valence) with early
        // exit. This also catches a live ring entry pointing at a dead slot
        // (a dead slot's ring was zeroed, so the reverse cannot exist).
        var bidirOK = ringsOK
        var badLinks: [String] = []
        if bidirOK {
            for i in 0..<v {
                if dead[i] != 0 { continue }
                let n = Int(counts[i])
                let base = i * maxValence
                for k in 0..<n {
                    let j = Int(links[base + k])
                    let jn = Int(counts[j])
                    let jbase = j * maxValence
                    var found = false
                    for k2 in 0..<jn {
                        if links[jbase + k2] == UInt32(i) { found = true; break }
                    }
                    if !found {
                        bidirOK = false
                        if badLinks.count < 4 {
                            // Diagnostic: the asymmetric pair + both rings
                            // (topology-bisect aid; see the V2.5 report).
                            let ri = (0..<n).map { Int(links[base + $0]) }
                            let rj = (0..<jn).map { Int(links[jbase + $0]) }
                            badLinks.append("i=\(i)->j=\(j) missing reverse | ring(i)=\(ri) ring(j)=\(rj)")
                        }
                        break
                    }
                }
                if !bidirOK { break }
            }
        }

        // Closed triangulated sphere: sum valence == 6 * (V - 2) — V = LIVE
        // cells (V2.5: a death splice is Euler-exact — dV −1, dE −3, dF −2 —
        // so the identity holds EXACTLY across deaths; dead slots are out).
        let eulerTarget = 6 * (liveN - 2)
        let eulerOK = sumValence == eulerTarget
        let valenceOK = minValence >= 3 && maxVal <= maxValence
        let finiteOK = nanCount == 0 && badFood == 0 && badChem == 0
        let typesOK = badType == 0
        let deadOK = deadWithLinks == 0
        let meanLen = lenCount > 0 ? lenSum / lenCount : 0
        let rest = sim.params.linkRestLength
        let lenOK = settled
            ? abs(meanLen - rest) <= 0.02 * rest
            : meanLen <= meanLimit * rest
        let boundOK = maxAbs.isFinite && maxAbs <= 10_000

        // V2.5: connectivity — ONE component (CPU flood-fill over the live
        // links, O(V + E)). Death closes the hole behind every removal, so
        // the mesh must never disconnect; a failed splice or a stale ring
        // shows up here immediately.
        var connOK = true
        var connDetail = "n/a"
        if liveN > 0 {
            var visited = [Bool](repeating: false, count: v)
            var stack: [Int] = []
            for i in 0..<v where dead[i] == 0 {
                visited[i] = true
                stack.append(i)
                break
            }
            var reached = 0
            while let i = stack.popLast() {
                reached += 1
                let base = i * maxValence
                for k in 0..<Int(counts[i]) {
                    let j = Int(links[base + k])
                    if j >= 0 && j < v && !visited[j] && dead[j] == 0 {
                        visited[j] = true
                        stack.append(j)
                    }
                }
            }
            connOK = reached == liveN
            connDetail = "reached \(reached)/\(liveN)"
        }

        var lines: [String] = []
        func add(_ name: String, _ ok: Bool, _ detail: String) {
            lines.append("\(ok ? "PASS" : "FAIL"): \(name)  (\(detail))")
        }
        add("links bidirectional", bidirOK, badLinks.isEmpty ? "every ring entry has its reverse" : badLinks.joined(separator: " || "))
        add("closed triangulation (sum valence == 6(V-2), V = live)", eulerOK, "sum=\(sumValence) target=\(eulerTarget) live=\(liveN) dead=\(v - liveN)")
        add("valence in [3, maxValence]", valenceOK, "min=\(minValence == Int.max ? 0 : minValence) max=\(maxVal)")
        add("no NaN/Inf (positions, food, chemicals, pulse)", finiteOK, "bad positions=\(nanCount) food=\(badFood) chem=\(badChem)")
        add("cell types valid (0-4, Σ per-type = live V)", typesOK, "bad type codes=\(badType)")
        add("dead slots clean (linkCounts 0, no live ring references)", deadOK, "dead slots with links=\(deadWithLinks)")
        add("one connected component (flood-fill)", connOK, connDetail)
        add(settled ? "mean link length ~ rest (settle)" : "mean link length bounded (growing)",
            lenOK, "mean=\(String(format: "%.4f", meanLen)) rest=\(rest) \(settled ? "(tolerance 2%)" : "(limit \(meanLimit)x)")")
        add("max |P| bounded", boundOK, "max=\(String(format: "%.2f", maxAbs)) limit=10000")

        // Structure diagnostics: link length distribution + valence histogram
        // (LIVE cells only).
        var lens = [Float]()
        lens.reserveCapacity(Int(lenCount))
        var valenceHist = [Int](repeating: 0, count: maxValence + 1)
        for i in 0..<v where dead[i] == 0 {
            valenceHist[Int(counts[i])] += 1
            let base = i * maxValence
            for k in 0..<Int(counts[i]) {
                lens.append(simd_distance(pos[i], pos[Int(links[base + k])]))
            }
        }
        lens.sort()
        func pct(_ p: Double) -> Float {
            guard !lens.isEmpty else { return 0 }
            let idx = min(Int(p * Double(lens.count)), lens.count - 1)
            return lens[idx]
        }
        let hist = (3...min(9, maxValence)).map { "v\($0)=\(valenceHist[$0])" }.joined(separator: " ")
        lines.append("diag: len p10=\(String(format: "%.2f", pct(0.10))) p50=\(String(format: "%.2f", pct(0.50))) p90=\(String(format: "%.2f", pct(0.90))) p99=\(String(format: "%.2f", pct(0.99))) max=\(String(format: "%.2f", pct(1.0)))  valences: \(hist)")

        let pass = bidirOK && eulerOK && valenceOK && finiteOK && typesOK && deadOK && connOK && lenOK && boundOK
        return (pass, lines)
    }

    // MARK: - Offscreen render + PNG

    /// Render the current state offscreen and return the raw RGBA bytes
    /// (shared by --screenshot and the golden capture/check). style: "solid"
    /// (the historical depth-tested impostors), "xray" (the M6 additive
    /// density-sprite impostors — the interactive X-ray, NOT the ray tracer)
    /// or "foam" (V2.1b: the premultiplied chamber composite, back-to-front
    /// sorted — see Renderer.encodeScene). Goldens always pass "solid" or an
    /// explicit style key in their meta, so their pixels are unchanged.
    /// V2.1b: the painted edge layer is retired; foamOpacity drives the foam
    /// composite (goldens record the value in the v21b meta block).
    private static func renderToBytes(sim: CellSim, queue: MTLCommandQueue, width: Int, height: Int,
                                      style: String = "solid",
                                      foamOpacity: Float = 0.5) -> [UInt8] {
        let device = sim.device

        let renderer = Renderer(device: device, queue: queue, sim: sim)
        renderer.isPlaying = false
        renderer.renderStyle = style == "xray" ? .xray : (style == "foam" ? .foam : .solid)
        renderer.foamOpacity = foamOpacity
        let ballRadius = sim.readPositions().map { simd_length($0) }.max() ?? 1
        renderer.camera.distance = ballRadius * 2.6

        guard let color = device.makeTexture(descriptor: textureDesc(width: width, height: height, pixelFormat: .bgra8Unorm, usage: [.renderTarget, .shaderRead])),
              let depth = device.makeTexture(descriptor: textureDesc(width: width, height: height, pixelFormat: .depth32Float, usage: [.renderTarget])) else {
            fatalError("screenshot: texture alloc failed")
        }

        // X-ray accumulates density on a BLACK background (offline-style look);
        // foam composites over the studio background like solid.
        let clear = style == "xray"
            ? MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            : MTLClearColor(red: 0.065, green: 0.075, blue: 0.095, alpha: 1)
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = color
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = clear
        rpd.depthAttachment.texture = depth
        rpd.depthAttachment.loadAction = .clear
        rpd.depthAttachment.storeAction = .dontCare
        rpd.depthAttachment.clearDepth = 1.0

        guard let cb = queue.makeCommandBuffer() else { return [] }
        renderer.encodeScene(into: rpd, commandBuffer: cb, aspect: Float(width) / Float(height))
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            print("Screenshot render error: \(error)")
        }

        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        color.getBytes(&bytes, bytesPerRow: width * 4,
                       from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                       size: MTLSize(width: width, height: height, depth: 1)),
                       mipmapLevel: 0)
        return bytes
    }

    private static func renderScreenshot(sim: CellSim, queue: MTLCommandQueue, to path: String,
                                         width: Int = 1600, height: Int = 1200,
                                         style: String = "solid",
                                         foamOpacity: Float = 0.5) {
        let bytes = renderToBytes(sim: sim, queue: queue, width: width, height: height, style: style,
                                  foamOpacity: foamOpacity)
        do {
            try writePNG(bytes, width: width, height: height, to: path)
            let styleName = style == "xray" ? "impostor X-ray" : (style == "foam" ? "impostor foam" : "solid")
            print("Screenshot written: \(path) (\(styleName))")
        } catch {
            print("Screenshot failed: \(error)")
        }
    }

    private static func textureDesc(width: Int, height: Int, pixelFormat: MTLPixelFormat, usage: MTLTextureUsage) -> MTLTextureDescriptor {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat,
                                                         width: width, height: height,
                                                         mipmapped: false)
        d.usage = usage
        d.storageMode = .shared
        return d
    }

    private static func writePNG(_ rgba: [UInt8], width: Int, height: Int, to path: String) throws {
        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB),
                                  bitmapInfo: CGBitmapInfo(rawValue:
                                      CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue)) else {
            throw NSError(domain: "png", code: 1)
        }
        rgba.withUnsafeBytes { src in
            ctx.data?.copyMemory(from: src.baseAddress!, byteCount: width * height * 4)
        }
        guard let img = ctx.makeImage() else {
            throw NSError(domain: "png", code: 1)
        }
        let url = URL(fileURLWithPath: path) as CFURL
        guard let dest = CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil) else {
            throw NSError(domain: "png", code: 2)
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(domain: "png", code: 3)
        }
    }
}

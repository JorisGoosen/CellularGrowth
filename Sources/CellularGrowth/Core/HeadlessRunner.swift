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
import CoreGraphics
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
            || args.contains("--rde-F") || args.contains("--rde-k")
            || args.contains("--rde-du") || args.contains("--rde-dv")
            || args.contains("--rde-base") || args.contains("--rde-gain")
            || args.contains("--rde-probe") || args.contains("--rde-dump")
            || args.contains("--render-light")
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
        var maxCells = 2_000_000
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
        var rdeF: Float = 0.037
        var rdeK: Float = 0.06
        var rdeDu: Float = 0.35
        var rdeDv: Float = 0.15
        var rdeBase: Float = 0.25
        var rdeGain: Float = 2.5
        var rdeProbe = false       // static-ball RDE pattern validation mode
        var rdeDump: String?       // chemical scatter dump directory (probe + runs)
        var renderLight = "auto"   // auto | growth | unbiased (M5 unified light)
    }

    private static func parseOptions() -> Options {
        var o = Options()
        let args = CommandLine.arguments
        func value(_ flag: String) -> String? {
            guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
            return args[idx + 1]
        }
        func given(_ flag: String) -> Bool { args.contains(flag) }
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
            case "--rde-F": o.rdeF = Float(value("--rde-F") ?? "") ?? o.rdeF
            case "--rde-k": o.rdeK = Float(value("--rde-k") ?? "") ?? o.rdeK
            case "--rde-du": o.rdeDu = Float(value("--rde-du") ?? "") ?? o.rdeDu
            case "--rde-dv": o.rdeDv = Float(value("--rde-dv") ?? "") ?? o.rdeDv
            case "--rde-base": o.rdeBase = Float(value("--rde-base") ?? "") ?? o.rdeBase
            case "--rde-gain": o.rdeGain = Float(value("--rde-gain") ?? "") ?? o.rdeGain
            case "--rde-probe": o.rdeProbe = true
            case "--rde-dump": o.rdeDump = value("--rde-dump")
            case "--render-light": o.renderLight = (value("--render-light") ?? "auto").lowercased()
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
        sim.params.growthMode = growthModeCode(o.growth)
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
        sim.params.rdeF = o.rdeF
        sim.params.rdeK = o.rdeK
        sim.params.rdeDu = min(max(o.rdeDu, 0), 1)
        sim.params.rdeDv = min(max(o.rdeDv, 0), 1)
        sim.params.rdeBase = max(o.rdeBase, 0)
        sim.params.rdeGain = o.rdeGain

        if o.rdeProbe {
            runRdeProbe(queue: queue, sim: sim, o: o)   // exits
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
        while done < o.steps {
            let batch = min(100, o.steps - done)
            guard let cb = queue.makeCommandBuffer() else { fatalError("no command buffer") }
            for _ in 0..<batch { sim.step(cb) }
            cb.commit()
            cb.waitUntilCompleted()
            done += batch
            sim.syncCellCountFromDevice(allowGridGrowth: true)
            if done >= o.divStop && sim.params.divisionEnabled != 0 {
                sim.params.divisionEnabled = 0
                sim.syncParams()
                print("t=\(done): division disabled (diagnostic --div-stop)")
            }
            if done % 500 == 0 || done == o.steps {
                print("t=\(done)   cells=\(sim.cellCount)  meanLinkLength=\(meanLinkLength())")
                printGrowthCheckpoint(sim: sim)
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
        sim.params.growthMode = 2
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

    private static func runHeadless(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        growAndReport(queue: queue, sim: sim, o: o)

        // M5: --state-save now works in the headless path too (it was only
        // handled by --render before; the flag silently did nothing here).
        if let savePath = o.stateSave {
            do {
                try saveStateSnapshot(savePath, sim: sim)
                print("state saved: \(savePath) (\(sim.cellCount) cells)")
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
            renderScreenshot(sim: sim, queue: queue, to: path)
        }

        exit(0)
    }

    // MARK: - M4 offline ray-traced render

    /// --render PATH: grow the configured state (same seeded pipeline as
    /// --headless), then ray-trace it offscreen in the chosen style. The grid
    /// for the final state is built ONCE here (the sim rebuilds it per step;
    /// rendering traverses a static snapshot of it).
    private static func runRender(queue: MTLCommandQueue, sim: CellSim, o: Options) -> Never {
        if let loadPath = o.stateLoad {
            // Render a STORED state (--state-load): skips the growth entirely
            // (paper-gate workflow: same state rendered in both styles).
            do {
                try loadStateSnapshot(loadPath, sim: sim)
            } catch {
                print("state load FAILED: \(error)")
                exit(1)
            }
            print("state loaded: \(loadPath) (\(sim.cellCount) cells)")
        } else {
            growAndReport(queue: queue, sim: sim, o: o, useSeededJitter: true)
        }

        // Fresh radii (daughters from the final step) + one grid rebuild for
        // the final state, in a single waited command buffer.
        if let cb = queue.makeCommandBuffer() {
            sim.refreshRadii(cb)
            sim.encodeGridRebuild(cb)
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { print("render prep CB ERROR: \(e)") }
        }
        sim.syncCellCountFromDevice()   // host mirror consistent (GPU idle)

        var settings = RayRenderSettings()
        settings.style = o.style == "xray" ? .xray : .solid
        if o.style != "solid" && o.style != "xray" {
            print("--style '\(o.style)' unknown; using solid")
        }
        // M5 unified light: the solid style shades with the SAME light the
        // growth illumination uses when directionality is requested. auto
        // selects directional in light-growth mode (so the render shows which
        // cells receive light) and the paper's unbiased omnidirectional look
        // otherwise; "growth"/"unbiased" override explicitly. X-ray has no
        // lighting (density accumulation) — unaffected.
        let sharedLight = SIMD3<Float>(sim.params.lightDirX, sim.params.lightDirY, sim.params.lightDirZ)
        switch o.renderLight {
        case "growth":
            settings.directionalLight = sharedLight
            print("render light: directional (shared growth light)")
        case "unbiased":
            settings.directionalLight = nil
            print("render light: unbiased omnidirectional (paper default)")
        case "auto":
            settings.directionalLight = sim.params.growthMode == 1 ? sharedLight : nil
            print("render light: auto -> \(settings.directionalLight != nil ? "directional (light-growth mode)" : "unbiased omnidirectional")")
        default:
            print("--render-light '\(o.renderLight)' unknown (auto | growth | unbiased)")
            exit(1)
        }
        settings.width = o.width
        settings.height = o.height
        settings.samples = o.samples
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

        // Same framing convention as --screenshot: camera distance from the
        // structure's max |P| (OrbitCamera yaw/pitch/fov defaults shared).
        var maxAbs: Float = 0
        for p_ in sim.readPositions() { maxAbs = max(maxAbs, simd_length(p_)) }
        settings.camera.distance = maxAbs * 2.6

        let renderer = RayRenderer(device: sim.device, queue: queue, sim: sim)
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = renderer.render(settings: settings)
        let wall = CFAbsoluteTimeGetCurrent() - t0
        print("Render: \(settings.style.rawValue == 0 ? "solid" : "xray") \(o.width)x\(o.height) samples=\(settings.samples) cells=\(sim.cellCount)  wall \(String(format: "%.2f", wall))s  gpu \(String(format: "%.2f", result.gpuSeconds))s")
        if let savePath = o.stateSave {
            do {
                try saveStateSnapshot(savePath, sim: sim)
                print("state saved: \(savePath) (\(sim.cellCount) cells)")
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
        print("cells            | ms/step cpu | ms/step gpu | steps/s")
        for level in levels {
            let (positions, links) = Icosphere.generate(subdivisions: level, spacing: sim.params.linkRestLength)
            sim.reset(positions: positions, links: links)
            let initial = sim.cellCount

            // warmup
            if let cb = queue.makeCommandBuffer() {
                for _ in 0..<10 { sim.step(cb) }
                cb.commit()
                cb.waitUntilCompleted()
            }

            let t0 = CFAbsoluteTimeGetCurrent()
            var gpuTime = 0.0
            var done = 0
            while done < steps {
                let batch = min(10, steps - done)
                guard let cb = queue.makeCommandBuffer() else { break }
                for _ in 0..<batch { sim.step(cb) }
                cb.commit()
                cb.waitUntilCompleted()
                gpuTime += cb.gpuEndTime - cb.gpuStartTime
                done += batch
            }
            let dt = CFAbsoluteTimeGetCurrent() - t0
            sim.syncCellCountFromDevice(allowGridGrowth: true)
            let cells = o.benchGrowth ? "\(initial) -> \(sim.cellCount)" : "\(initial)"
            print("\(cells) | \(String(format: "%.3f", dt / Double(steps) * 1000)) | \(String(format: "%.3f", gpuTime / Double(steps) * 1000)) | \(String(format: "%.0f", Double(steps) / dt))")
            print("   \(sim.gridMemoryDescription())")
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
        sim.params.growthMode = 0
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
    }

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

    /// Binary snapshot of the live state: magic "CGST", version, cellCount,
    /// maxValence, then positions (SIMD3 stride 16), links (strided per cell),
    /// linkCounts. GPU must be idle.
    private static func saveStateSnapshot(_ path: String, sim: CellSim) throws {
        let count = sim.cellCount
        var bytes = [UInt8]()
        bytes.reserveCapacity(16 + count * (MemoryLayout<SIMD3<Float>>.stride
                                            + CellSim.maxValence * MemoryLayout<UInt32>.stride
                                            + MemoryLayout<UInt32>.stride))
        func append(_ v: UInt32) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { bytes.append(contentsOf: $0) }
        }
        bytes.append(contentsOf: Array("CGST".utf8))
        append(1)                               // version
        append(UInt32(count))
        append(UInt32(CellSim.maxValence))
        let pos = UnsafeBufferPointer(start: sim.positionsPointer, count: count)
        for p in pos {
            for c in [p.x, p.y, p.z] {
                var f = c.bitPattern.littleEndian
                withUnsafeBytes(of: &f) { bytes.append(contentsOf: $0) }
            }
            var pad: UInt32 = 0
            withUnsafeBytes(of: &pad) { bytes.append(contentsOf: $0) }
        }
        let links = UnsafeBufferPointer(start: sim.linksPointer, count: count * CellSim.maxValence)
        for l in links { append(l) }
        let counts = UnsafeBufferPointer(start: sim.linkCountsPointer, count: count)
        for c in counts { append(c) }
        try Data(bytes).write(to: URL(fileURLWithPath: path))
    }

    /// Load a state snapshot and install it via CellSim.reset (recomputes
    /// normals, radii, food/RNG placeholders deterministically).
    private static func loadStateSnapshot(_ path: String, sim: CellSim) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        func u32(_ off: Int) -> UInt32 {
            UInt32(data[off]) | (UInt32(data[off+1]) << 8) | (UInt32(data[off+2]) << 16) | (UInt32(data[off+3]) << 24)
        }
        guard data.count >= 16, String(bytes: data[0..<4], encoding: .ascii) == "CGST",
              u32(4) == 1 else {
            throw NSError(domain: "golden", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "\(path): not a CGST v1 state snapshot"])
        }
        let count = Int(u32(8))
        let maxValence = Int(u32(12))
        guard maxValence == CellSim.maxValence else {
            throw NSError(domain: "golden", code: 11,
                          userInfo: [NSLocalizedDescriptionKey: "\(path): maxValence \(maxValence) != \(CellSim.maxValence)"])
        }
        let expected = 16 + count * (MemoryLayout<SIMD3<Float>>.stride + maxValence * 4 + 4)
        guard data.count == expected else {
            throw NSError(domain: "golden", code: 12,
                          userInfo: [NSLocalizedDescriptionKey: "\(path): size \(data.count) != expected \(expected) for \(count) cells"])
        }
        var positions = [SIMD3<Float>](repeating: .zero, count: count)
        data.withUnsafeBytes { src in
            positions.withUnsafeMutableBytes { dst in
                dst.copyBytes(from: UnsafeRawBufferPointer(rebasing: src.dropFirst(16).prefix(count * MemoryLayout<SIMD3<Float>>.stride)))
            }
        }
        var flatLinks = [UInt32](repeating: 0, count: count * maxValence)
        data.withUnsafeBytes { src in
            flatLinks.withUnsafeMutableBytes { dst in
                dst.copyBytes(from: UnsafeRawBufferPointer(rebasing: src.dropFirst(16 + count * MemoryLayout<SIMD3<Float>>.stride).prefix(count * maxValence * 4)))
            }
        }
        // linkCounts block follows the links block.
        var counts = [UInt32](repeating: 0, count: count)
        data.withUnsafeBytes { src in
            counts.withUnsafeMutableBytes { dst in
                dst.copyBytes(from: UnsafeRawBufferPointer(rebasing: src.dropFirst(16 + count * (MemoryLayout<SIMD3<Float>>.stride + maxValence * 4)).prefix(count * 4)))
            }
        }
        var links = [[UInt32]](repeating: [], count: count)
        for i in 0..<count {
            let n = Int(counts[i])
            links[i] = Array(flatLinks[i * maxValence..<(i * maxValence + n)])
        }
        sim.reset(positions: positions, links: links)
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
        // light now reads it from SimParams.
        sim.params.growthMode = 0
        sim.params.lightDirX = 0.55
        sim.params.lightDirY = 0.75
        sim.params.lightDirZ = 0.4
        sim.params.lightDiffusionRate = 0.02
        sim.params.lightNutrient = 0.05
        sim.params.lightDecay = 0.02
        sim.params.lightBase = 0.15
        sim.params.rdeF = 0.037
        sim.params.rdeK = 0.06
        sim.params.rdeDu = 0.35
        sim.params.rdeDv = 0.15
        sim.params.rdeBase = 0.25
        sim.params.rdeGain = 2.5
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
                try loadStateSnapshot(goldenStatePath(name), sim: sim)
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
            sim.syncParams()

            let (base, links) = Icosphere.generate(subdivisions: cfg.subdiv, spacing: sim.params.linkRestLength)
            let positions = cfg.jitter > 0 ? seededJitter(base, radius: cfg.jitter, seed: cfg.seed) : base
            sim.reset(positions: positions, links: links)
            runSteps(queue: queue, sim: sim, count: cfg.steps)
        }

        // M4 ray-traced golden: fresh radii + ONE grid rebuild for the final
        // state (the renderer traverses the static hash), then the tile loop.
        if let cb = queue.makeCommandBuffer() {
            sim.refreshRadii(cb)
            sim.encodeGridRebuild(cb)
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { print("golden render prep CB ERROR: \(e)") }
        }
        sim.syncCellCountFromDevice()

        var settings = RayRenderSettings()
        settings.style = r.style == "xray" ? .xray : .solid
        settings.directionalLight = nil   // goldens keep the unbiased look (M5 unified light)
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
            render = RenderGoldenSettings(style: o.style, samples: o.samples,
                                          maxCells: o.maxCells, division: o.division,
                                          aoRadius: d.aoRadius, aoStrength: d.aoStrength,
                                          aoFloor: d.aoFloor, xrayExposure: d.xrayExposure,
                                          xrayGamma: d.xrayGamma, densityScale: d.densityScale,
                                          profileSupport: d.profileSupport,
                                          snapshot: true, seed: d.seed)
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
                try saveStateSnapshot(goldenStatePath(name), sim: sim)
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
        sim.params.growthMode = 1
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
        print("   light gradient: facing=\(String(format: "%.3f", lightMetrics.facing)) away=\(String(format: "%.3f", lightMetrics.away)) ratio=\(String(format: "%.2f", lightMetrics.ratio)) \(lightMetrics.ratio > 1.2 ? "PASS" : "FAIL") (gate > 1.2)")
        if !(lightMetrics.ratio > 1.2) { allPass = false }

        // --- Phase D: growth under the RDE mode (M5) ---
        sim.params.growthMode = 2
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
        sim.params.growthMode = 0
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
    private static func checkInvariants(sim: CellSim, settled: Bool, meanLimit: Float) -> (pass: Bool, lines: [String]) {
        let v = sim.cellCount
        guard v > 2 else { return (false, ["FAIL: no cells"]) }
        let maxValence = CellSim.maxValence
        let counts = UnsafeBufferPointer(start: sim.linkCountsPointer, count: v)
        let links = UnsafeBufferPointer(start: sim.linksPointer, count: v * maxValence)
        let pos = UnsafeBufferPointer(start: sim.positionsPointer, count: v)
        let food = UnsafeBufferPointer(start: sim.foodPointer, count: v)

        var sumValence = 0
        var minValence = Int.max
        var maxVal = 0
        var nanCount = 0
        var badFood = 0
        var maxAbs: Float = 0
        var lenSum: Float = 0
        var lenCount: Float = 0

        // Pass 1: ring bounds + duplicate entries, valence range, finite
        // positions/food, link lengths, max |P|.
        var seen = [UInt32](repeating: UInt32.max, count: v)
        var ringsOK = true
        for i in 0..<v {
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
            for x in UnsafeBufferPointer(start: sim.nutrientPointer, count: v) where !x.isFinite { badChem += 1 }
        }
        if sim.params.growthMode == 2 {
            for x in UnsafeBufferPointer(start: sim.chemUPointer, count: v) where !x.isFinite { badChem += 1 }
            for x in UnsafeBufferPointer(start: sim.chemVPointer, count: v) where !x.isFinite { badChem += 1 }
        }

        // Pass 2: bidirectionality — every directed link (i -> j) must have
        // its reverse (j -> i). O(links * max valence) with early exit.
        var bidirOK = ringsOK
        if bidirOK {
            for i in 0..<v {
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
                    if !found { bidirOK = false; break }
                }
                if !bidirOK { break }
            }
        }

        // Closed triangulated sphere: sum valence == 6 * (V - 2)
        // (Euler: V - E + F = 2 with E = sum val / 2, F = sum val / 3; the
        // equality forces every face to be a triangle).
        let eulerTarget = 6 * (v - 2)
        let eulerOK = sumValence == eulerTarget
        let valenceOK = minValence >= 3 && maxVal <= maxValence
        let finiteOK = nanCount == 0 && badFood == 0 && badChem == 0
        let meanLen = lenCount > 0 ? lenSum / lenCount : 0
        let rest = sim.params.linkRestLength
        let lenOK = settled
            ? abs(meanLen - rest) <= 0.02 * rest
            : meanLen <= meanLimit * rest
        let boundOK = maxAbs.isFinite && maxAbs <= 10_000

        var lines: [String] = []
        func add(_ name: String, _ ok: Bool, _ detail: String) {
            lines.append("\(ok ? "PASS" : "FAIL"): \(name)  (\(detail))")
        }
        add("links bidirectional", bidirOK, "every ring entry has its reverse")
        add("closed triangulation (sum valence == 6(V-2))", eulerOK, "sum=\(sumValence) target=\(eulerTarget)")
        add("valence in [3, maxValence]", valenceOK, "min=\(minValence == Int.max ? 0 : minValence) max=\(maxVal)")
        add("no NaN/Inf (positions, food, M5 chemicals)", finiteOK, "bad positions=\(nanCount) food=\(badFood) chem=\(badChem)")
        add(settled ? "mean link length ~ rest (settle)" : "mean link length bounded (growing)",
            lenOK, "mean=\(String(format: "%.4f", meanLen)) rest=\(rest) \(settled ? "(tolerance 2%)" : "(limit \(meanLimit)x)")")
        add("max |P| bounded", boundOK, "max=\(String(format: "%.2f", maxAbs)) limit=10000")

        // Structure diagnostics: link length distribution + valence histogram.
        var lens = [Float]()
        lens.reserveCapacity(Int(lenCount))
        var valenceHist = [Int](repeating: 0, count: maxValence + 1)
        for i in 0..<v {
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

        let pass = bidirOK && eulerOK && valenceOK && finiteOK && lenOK && boundOK
        return (pass, lines)
    }

    // MARK: - Offscreen render + PNG

    /// Render the current state offscreen and return the raw RGBA bytes
    /// (shared by --screenshot and the golden capture/check).
    private static func renderToBytes(sim: CellSim, queue: MTLCommandQueue, width: Int, height: Int) -> [UInt8] {
        let device = sim.device

        let renderer = Renderer(device: device, queue: queue, sim: sim)
        renderer.isPlaying = false
        let ballRadius = sim.readPositions().map { simd_length($0) }.max() ?? 1
        renderer.camera.distance = ballRadius * 2.6

        guard let color = device.makeTexture(descriptor: textureDesc(width: width, height: height, pixelFormat: .bgra8Unorm, usage: [.renderTarget, .shaderRead])),
              let depth = device.makeTexture(descriptor: textureDesc(width: width, height: height, pixelFormat: .depth32Float, usage: [.renderTarget])) else {
            fatalError("screenshot: texture alloc failed")
        }

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = color
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.065, green: 0.075, blue: 0.095, alpha: 1)
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
                                         width: Int = 1600, height: Int = 1200) {
        let bytes = renderToBytes(sim: sim, queue: queue, width: width, height: height)
        do {
            try writePNG(bytes, width: width, height: height, to: path)
            print("Screenshot written: \(path)")
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

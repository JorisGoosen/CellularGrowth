//  Presets.swift
//  M6: named parameter presets — ONE source of truth shared by the CLI
//  (--preset NAME) and the GUI (preset dropdown). A preset is a full param
//  bundle (growth mode, force factors, repulsion, food/threshold, light/rde
//  coupling) plus the reset-time shape knobs (subdivision, jitter) that
//  co-determine the form.
//
//  Values are documented next to each preset; where the paper gives numbers
//  (Table 1 factor ranges, §7 diffusion gate values) they are cited. Presets
//  only override what they declare — explicit CLI flags still win (--preset
//  is applied before flag parsing, and the GUI applies it to the published
//  slider state so the panel always shows what is running).

import Foundation

struct SimPreset {
    let name: String
    let summary: String
    /// Reset-time shape knobs (SimViewModel.jitter / .subdivision equivalents).
    let subdivision: Int
    let jitter: Float
    /// Full SimParams bundle.
    let params: SimParams
    /// V2.3: reset-time polar typing knobs (nil = no typing — all cells stay
    /// type 0, the historical behaviour). Types are per-cell STATE assigned
    /// at reset by CellSim.applyPolarTypes, NOT SimParams — the axis and the
    /// stem/cap thresholds are the reset-time knobs (like subdivision/jitter).
    var typeAxis: SIMD3<Float>? = nil
    /// s < typeT1 -> stem (type 1).
    var typeT1: Float = -0.35
    /// s > typeT2 -> cap (type 4); the lamel/dal band sits between.
    var typeT2: Float = 0.30

    static let all: [SimPreset] = [ball, brain, plant, coral, degenerateDemo, paddenstoel]

    static func named(_ name: String) -> SimPreset? {
        all.first { $0.name == name }
    }

    // MARK: - The presets

    /// M1 rest state: division off, no food, no repulsion — the paper's Fig. 3
    /// relaxation target (spring/planar/bulge all 1/3, Table 1 mid-range).
    static let ball = SimPreset(
        name: "ball",
        summary: "M1 rest state — forces only, no growth (paper Fig. 3)",
        subdivision: 3,
        jitter: 0.25,
        params: SimParams(
            springFactor: 1.0 / 3.0, planarFactor: 1.0 / 3.0, bulgeFactor: 1.0 / 3.0,
            foodIncrement: 0, splitThreshold: 1.0, divisionEnabled: 0,
            roi: 1.6, repulsionStrength: 0,
            growthMode: 0))

    /// Uniform growth + repulsion (paper §7: "uniform cell growth rates ...
    /// results that look more like internal organs, such as brains, with
    /// complex folded shapes"). Defaults that produced coherent folded forms
    /// in M3–M5 validation runs (benchlog M3/M5).
    static let brain = SimPreset(
        name: "brain",
        summary: "Uniform growth + repulsion — organ/brain-like folds (paper Fig. 8)",
        subdivision: 3,
        jitter: 0.25,
        params: SimParams(
            springFactor: 1.0 / 3.0, planarFactor: 1.0 / 3.0, bulgeFactor: 1.0 / 3.0,
            foodIncrement: 0.01, splitThreshold: 1.0, divisionEnabled: 1,
            roi: 1.6, repulsionStrength: 0.1,
            growthMode: 0))

    /// Light-created nutrient with LOW diffusion (paper §7: low inter-cellular
    /// diffusion + single-direction light → "remarkably plant-like forms",
    /// Fig. 11). Light straight up (0,1,0) — the paper's single-direction
    /// illumination. Growth drive ∝ (lightBase + nutrient).
    static let plant = SimPreset(
        name: "plant",
        summary: "Light nutrient, low diffusion — plant-like growth toward the light (paper Fig. 11)",
        subdivision: 3,
        jitter: 0.25,
        params: SimParams(
            springFactor: 1.0 / 3.0, planarFactor: 1.0 / 3.0, bulgeFactor: 1.0 / 3.0,
            foodIncrement: 0.01, splitThreshold: 1.0, divisionEnabled: 1,
            roi: 1.6, repulsionStrength: 0.1,
            growthMode: 1,
            lightDirX: 0, lightDirY: 1, lightDirZ: 0,
            lightDiffusionRate: 0.02, lightNutrient: 0.05,
            lightDecay: 0.02, lightBase: 0.15, lightInterval: 8))

    /// RDE-patterned growth (paper §4 method 2): Gray-Scott spot pattern
    /// (F 0.037 / k 0.06 — the "coral/mitosis" regime, M5 rde-probe) drives
    /// the food increment ∝ (rdeBase + rdeGain·v) — localised bursts of
    /// growth where v is high.
    static let coral = SimPreset(
        name: "coral",
        summary: "RDE (Gray-Scott) patterned growth — coral-like localised growth",
        subdivision: 3,
        jitter: 0.25,
        params: SimParams(
            springFactor: 1.0 / 3.0, planarFactor: 1.0 / 3.0, bulgeFactor: 1.0 / 3.0,
            foodIncrement: 0.01, splitThreshold: 1.0, divisionEnabled: 1,
            roi: 1.6, repulsionStrength: 0.1,
            growthMode: 2,
            rdeF: 0.037, rdeK: 0.06, rdeDu: 0.35, rdeDv: 0.15,
            rdeBase: 0.25, rdeGain: 2.5))

    /// Paper §7/Fig. 5: WITHOUT repulsion the forms degenerate into "a
    /// meaningless unstructured mess" — the surface self-intersects as it
    /// folds. Same growth as `brain` but repulsion 0.
    static let degenerateDemo = SimPreset(
        name: "degenerate-demo",
        summary: "No repulsion — surface degeneration (paper Fig. 5 warning case)",
        subdivision: 3,
        jitter: 0.25,
        params: SimParams(
            springFactor: 1.0 / 3.0, planarFactor: 1.0 / 3.0, bulgeFactor: 1.0 / 3.0,
            foodIncrement: 0.01, splitThreshold: 1.0, divisionEnabled: 1,
            roi: 1.6, repulsionStrength: 0,
            growthMode: 0))

    /// V2.3 "paddenstoel" (mushroom): the two-phase typed-growth showcase.
    /// Vertical type axis; stem = s < -0.4 (the bottom), cap = s > 0.9 (a
    /// small top patch that grows into the dome), the lamel/dal band between
    /// them (the future gill collar). Phase 1: the cap grows uniformly
    /// (exponential dome inflation from the small patch) and the stem only
    /// at its band edge (stalk elongation); the band itself is dormant.
    /// Phase 2 (after phaseSwitchX = 2700 executed divisions, ~t=1050 at
    /// subdiv 3): ONLY the band grows — lamel at lamelRate 0.5 at its edge
    /// links (neighbours in the stem or cap), dal at half that (0.25),
    /// lamel<->dal interior links at lamelRate/10 (0.05) — the gill collar
    /// ruffles out at the stem/cap junction while the cap and stem freeze.
    /// Tuned by screenshot sweep (probe_A..probe_E, V2.3 report): the cap
    /// patch must start SMALL (t2 high) or its exponential phase-1 growth
    /// swamps the form (probe_t400); the phase-2 rates must sit below the
    /// phase-1 cap rate or the collar swamps the stem (probe_A/probe_E).
    /// Foam rendering + --tint type (or type-pulse) recommended
    /// (--radius-scale 0.7). Every value documented here is a tuned choice
    /// where the paper is silent (the paper has no cell types — this is the
    /// user's head idea).
    static let paddenstoel = SimPreset(
        name: "paddenstoel",
        summary: "Two-phase typed growth — stem + cap + lamel/dal gill band (mushroom; foam + tint type recommended)",
        subdivision: 3,
        jitter: 0.25,
        params: SimParams(
            springFactor: 1.0 / 3.0, planarFactor: 1.0 / 3.0, bulgeFactor: 1.0 / 3.0,
            foodIncrement: 0.01, splitThreshold: 1.0, divisionEnabled: 1,
            roi: 1.6, repulsionStrength: 0.1,
            growthMode: 0,
            lamelRate: 0.5, dalRate: 0.25, interiorDivider: 10.0,
            phaseSwitchX: 2700),
        typeAxis: SIMD3<Float>(0, 1, 0),
        typeT1: -0.4,
        typeT2: 0.9)
}

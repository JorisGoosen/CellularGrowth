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

    static let all: [SimPreset] = [ball, brain, plant, coral, degenerateDemo]

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
}

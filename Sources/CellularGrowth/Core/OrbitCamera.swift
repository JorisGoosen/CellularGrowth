//  OrbitCamera.swift
//  Minimal orbit camera plus column-major matrix helpers matching Metal's
//  column-vector convention (MSL: matrix * vector).

import simd

/// V2.6 camera choreography: follow-centroid + circle mode. Purely host-side
/// VIEWER state — the no-momentum rule (paper §4, AGENTS.md hard rule 5)
/// governs the integrator, not the observer; the exponential smoothing here
/// is a display-side filter on the camera TARGET, not a simulated mass, and
/// no simulation buffer is touched.
struct CameraChoreography {
    /// The camera target tracks the LIVE cells' geometric centroid (uniform
    /// cell mass — the paper assigns no per-cell masses; documented design
    /// choice). Defaults match the HEADLESS contract (OFF = the legacy
    /// fixed-target framing the goldens cover); the GUI switches it ON in
    /// SimViewModel.init — the asymmetry is deliberate and documented.
    var followCentroid = false
    /// Per-frame exponential smoothing of the followed target:
    /// smoothed += (centroid − smoothed) · α. α 0 = snap, 1 = frozen. The
    /// state snaps to the centroid when (re)initialized (nil) — a follow
    /// that starts centered, then smooths the growth drift.
    var followSmoothing: Float = 0.15
    /// Circle mode: yaw advances by `orbitRate` radians EVERY rendered frame
    /// around the (possibly following) target. Per-FRAME — not per-second —
    /// because export frames are equally spaced in STEPS (the M6 convention),
    /// so a per-frame rate keeps exports deterministic; the GUI renders at
    /// ~60 fps (0.005 rad/frame ≈ one full turn per 21 s).
    var orbiting = false
    var orbitRate: Float = 0.005
    /// Smoothed target state (nil = not initialized — the next advance
    /// snaps to the current centroid).
    private(set) var smoothedTarget: SIMD3<Float>? = nil

    /// One rendered frame of follow-centroid smoothing. The RAW centroid of
    /// the live cells comes from CellSim.liveCentroidAndRadius() (host-side
    /// CPU pass — no GPU pass added).
    mutating func advance(centroid: SIMD3<Float>) {
        let a = min(max(followSmoothing, 0), 1)
        if let s = smoothedTarget {
            smoothedTarget = s + (centroid - s) * a
        } else {
            smoothedTarget = centroid   // snap on (re)init
        }
    }

    /// Clear the smoothing state (reset / state load / follow switched off):
    /// the next advance snaps to the current centroid.
    mutating func reset() { smoothedTarget = nil }
}

struct OrbitCamera {
    var target = SIMD3<Float>.zero
    var distance: Float = 20
    var yaw: Float = 0.7      // radians, around world +Y
    var pitch: Float = 0.35   // radians
    var fovY: Float = .pi / 4

    var eye: SIMD3<Float> {
        target + distance * SIMD3<Float>(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
    }

    func viewMatrix() -> float4x4 {
        lookAt(eye: eye, target: target, up: SIMD3<Float>(0, 1, 0))
    }

    func projectionMatrix(aspect: Float, near: Float = 0.1, far: Float = 2000) -> float4x4 {
        perspective(fovY: fovY, aspect: aspect, near: near, far: far)
    }
}

/// Right-handed view matrix; in view space the camera looks down -Z.
func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
    let z = simd_normalize(eye - target)       // backward
    let x = simd_normalize(simd_cross(up, z))  // right
    let y = simd_cross(z, x)                   // up
    return float4x4(rows: [
        SIMD4(x.x, x.y, x.z, -simd_dot(x, eye)),
        SIMD4(y.x, y.y, y.z, -simd_dot(y, eye)),
        SIMD4(z.x, z.y, z.z, -simd_dot(z, eye)),
        SIMD4(0, 0, 0, 1),
    ])
}

/// Metal-style perspective (clip z in [0,1]).
func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
    let f = 1 / tan(fovY * 0.5)
    let nf = 1 / (near - far)
    return float4x4(columns: (
        SIMD4(f / aspect, 0, 0, 0),
        SIMD4(0, f, 0, 0),
        SIMD4(0, 0, far * nf, -1),
        SIMD4(0, 0, far * near * nf, 0)
    ))
}

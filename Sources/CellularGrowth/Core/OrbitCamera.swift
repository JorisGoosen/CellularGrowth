//  OrbitCamera.swift
//  Minimal orbit camera plus column-major matrix helpers matching Metal's
//  column-vector convention (MSL: matrix * vector).

import simd

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

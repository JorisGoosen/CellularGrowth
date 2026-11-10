//  CPUReference.swift
//  Pure-Swift golden reference of the exact GPU step semantics (--lockstep).
//
//  Reproduces, in Float precision and brute-force O(N^2) form, the paper §4
//  step the GPU executes:
//    1. normals — area-weighted, ring-order cross-product walk (computeNormals)
//    2. collisionOffset — brute force over ALL cells in cell-INDEX order
//       (repulsionAccumulate sums the same contributions in spatial-hash bin
//       order; float addition is not associative, so bitwise equality with the
//       GPU is impossible — the comparison runs on tolerances, calibrated in
//       the M3 report)
//    3. integrate — over-damped spring/planar/bulge + collisionOffset, same
//       clamps as the shader (bulge radicand >= 0 with the sign kept, the M2
//       revision; d > 1e-9 spring guards; collisionOffset added only when
//       repulsionStrength > 0)
//  Division is not modelled: --lockstep runs with division OFF (daughter slot
//  allocation is GPU-schedule-dependent, see the M2 report — comparing
//  index-labelled topology across implementations is meaningless anyway).

import Foundation
import simd

struct CPUReference {

    let linkRestLength: Float
    let springFactor: Float
    let planarFactor: Float
    let bulgeFactor: Float
    let roi: Float
    let repulsionStrength: Float

    private(set) var positions: [SIMD3<Float>]
    private let links: [[UInt32]]        // ring order, as generated
    /// O(1) linked test for the brute force (also kept for the CELLDBG
    /// fingerprint cross-checks).
    private let linkSets: [Set<UInt32>]

    init(positions: [SIMD3<Float>], links: [[UInt32]],
         linkRestLength: Float, springFactor: Float, planarFactor: Float,
         bulgeFactor: Float, roi: Float, repulsionStrength: Float) {
        self.positions = positions
        self.links = links
        self.linkRestLength = linkRestLength
        self.springFactor = springFactor
        self.planarFactor = planarFactor
        self.bulgeFactor = bulgeFactor
        self.roi = roi
        self.repulsionStrength = repulsionStrength
        self.linkSets = links.map(Set.init)
    }

    /// Area-weighted vertex normal, same ring walk as computeNormals.
    func computeNormals() -> [SIMD3<Float>] {
        var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
        for i in positions.indices {
            let P = positions[i]
            let ring = links[i]
            var acc = SIMD3<Float>.zero
            if ring.count >= 3 {
                var prev = positions[Int(ring[ring.count - 1])] - P
                for k in 0..<ring.count {
                    let cur = positions[Int(ring[k])] - P
                    acc += simd_cross(prev, cur)
                    prev = cur
                }
            }
            let len2 = simd_dot(acc, acc)
            normals[i] = len2 > 1e-20 ? acc * (1.0 / sqrt(len2)) : SIMD3<Float>(0, 1, 0)
        }
        return normals
    }

    /// Paper §4 collisionOffset, brute force: every cell pair within roi that
    /// is not directly linked, accumulated in cell-index order.
    func collisionOffsets(normals: [SIMD3<Float>]? = nil) -> [SIMD3<Float>] {
        let N = normals ?? computeNormals()
        let n = positions.count
        var out = [SIMD3<Float>](repeating: .zero, count: n)
        if repulsionStrength <= 0 { return out }
        let roi2 = roi * roi
        let invRoi2 = 1.0 / roi2
        for i in 0..<n {
            let P = positions[i]
            let ring = linkSets[i]
            var off = SIMD3<Float>.zero
            for r in 0..<n where r != i {
                let D = P - positions[r]
                let d2 = simd_dot(D, D)
                if d2 > roi2 { continue }
                if ring.contains(UInt32(r)) { continue }
                let d = sqrt(d2)
                let dir = d > 1e-6 ? D / d : N[i]
                off += ((roi2 - d2) * invRoi2) * dir
            }
            out[i] = repulsionStrength * off
        }
        return out
    }

    /// One GPU-equivalent step (no food, no division).
    mutating func step() {
        let N = computeNormals()
        let collision = collisionOffsets(normals: N)
        let n = positions.count
        var next = positions
        let L = linkRestLength
        let L2 = L * L
        for i in 0..<n {
            let P = positions[i]
            let normal = N[i]
            let ring = links[i]
            let cnt = ring.count
            var springTarget = SIMD3<Float>.zero
            var planarTarget = SIMD3<Float>.zero
            var bulgeSum: Float = 0
            for k in 0..<cnt {
                let Lp = positions[Int(ring[k])]
                let toP = P - Lp
                let d = simd_length(toP)
                // (1) spring: push to rest length along the link direction
                springTarget += Lp + (d > 1e-9 ? toP * (L / d) : SIMD3<Float>.zero)
                // (2) planar: pull toward the neighbour centroid
                planarTarget += Lp
                // (3) bulge: outward push along N for compressed links; the
                // sqrt is domain-clamped but the sum term keeps its sign
                // (M2 revision of the M1 clamp, see integrateForces).
                let lv = Lp - P
                let dotN = simd_dot(lv, normal)
                let t2 = simd_dot(lv, lv) - dotN * dotN
                bulgeSum += sqrt(max(L2 - t2, 0)) + dotN
            }
            let invN: Float = cnt > 0 ? 1.0 / Float(cnt) : 0
            springTarget *= invN
            planarTarget *= invN
            let bulgeDist = bulgeSum * invN
            var Pn = P
                + springFactor * (springTarget - P)
                + planarFactor * (planarTarget - P)
                + bulgeFactor * (bulgeDist * normal)
            if repulsionStrength > 0 { Pn += collision[i] }
            next[i] = Pn
        }
        positions = next
    }
}

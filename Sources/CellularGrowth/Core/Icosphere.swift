//  Icosphere.swift
//  Initial "ball of cells": particles uniformly spaced on the surface of a sphere
//  (paper §4, Figure 3). Links are emitted in CYCLIC RING ORDER (CCW seen from
//  outside) — the invariant the normal kernel relies on, and which cell division
//  (M3) must preserve.

import Foundation
import simd

enum Icosphere {

    /// Returns cell positions scaled so the mean neighbour distance equals `spacing`,
    /// plus ring-ordered neighbour lists per cell.
    static func generate(subdivisions: Int, spacing: Float)
        -> (positions: [SIMD3<Float>], links: [[UInt32]])
    {
        let t: Float = (1 + sqrt(5)) / 2

        // All base vertices normalized to the unit sphere (the raw table has
        // circumradius sqrt(1 + t²) ≈ 1.902, which would leave 12 spikes).
        var verts: [SIMD3<Float>] = [
            SIMD3(-1,  t,  0), SIMD3( 1,  t,  0), SIMD3(-1, -t,  0), SIMD3( 1, -t,  0),
            SIMD3( 0, -1,  t), SIMD3( 0,  1,  t), SIMD3( 0, -1, -t), SIMD3( 0,  1, -t),
            SIMD3( t,  0, -1), SIMD3( t,  0,  1), SIMD3(-t,  0, -1), SIMD3(-t,  0,  1),
        ].map { simd_normalize($0) }
        var faces: [(UInt32, UInt32, UInt32)] = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        // Loop-subdivide each triangle into 4, projecting midpoints onto the sphere.
        for _ in 0..<max(0, subdivisions) {
            var midCache: [UInt64: UInt32] = [:]
            var newFaces: [(UInt32, UInt32, UInt32)] = []
            newFaces.reserveCapacity(faces.count * 4)
            func midpoint(_ a: UInt32, _ b: UInt32) -> UInt32 {
                let key = a < b ? (UInt64(a) << 32 | UInt64(b)) : (UInt64(b) << 32 | UInt64(a))
                if let cached = midCache[key] { return cached }
                let m = UInt32(verts.count)
                let mv = simd_normalize((verts[Int(a)] + verts[Int(b)]) * 0.5)
                verts.append(mv)
                midCache[key] = m
                return m
            }
            for (a, b, c) in faces {
                let ab = midpoint(a, b)
                let bc = midpoint(b, c)
                let ca = midpoint(c, a)
                newFaces.append(contentsOf: [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
            }
            faces = newFaces
        }

        // Adjacency sets from face edges.
        var adjacency: [Set<UInt32>] = Array(repeating: [], count: verts.count)
        for (a, b, c) in faces {
            adjacency[Int(a)].insert(b); adjacency[Int(a)].insert(c)
            adjacency[Int(b)].insert(a); adjacency[Int(b)].insert(c)
            adjacency[Int(c)].insert(a); adjacency[Int(c)].insert(b)
        }

        // Order each neighbour ring CCW around the outward vertex normal
        // (angle measured from the first neighbour; cyclic, so wrap-around is valid).
        var links: [[UInt32]] = []
        links.reserveCapacity(verts.count)
        for i in 0..<verts.count {
            let v = verts[i]
            let n = simd_normalize(v)
            let nbrs = Array(adjacency[i])
            precondition(nbrs.count >= 3, "Icosphere vertex with degenerate valence")
            let ref = verts[Int(nbrs[0])] - v
            let sorted = nbrs.sorted { u, w in
                let au = verts[Int(u)] - v
                let aw = verts[Int(w)] - v
                let angU = atan2(simd_dot(simd_cross(ref, au), n), simd_dot(ref, au))
                let angW = atan2(simd_dot(simd_cross(ref, aw), n), simd_dot(ref, aw))
                return angU < angW
            }
            links.append(sorted)
        }

        // Rescale so the mean edge length equals `spacing` (=> mean link length
        // starts at linkRestLength, the spring equilibrium).
        var edgeSum: Float = 0
        for (a, b, c) in faces {
            edgeSum += simd_distance(verts[Int(a)], verts[Int(b)])
            edgeSum += simd_distance(verts[Int(b)], verts[Int(c)])
            edgeSum += simd_distance(verts[Int(c)], verts[Int(a)])
        }
        let meanEdge = edgeSum / Float(faces.count * 3)
        let scale = spacing / meanEdge
        let positions = verts.map { $0 * scale }
        return (positions, links)
    }
}

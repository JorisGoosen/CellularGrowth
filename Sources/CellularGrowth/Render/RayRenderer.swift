//  RayRenderer.swift
//  M4: offline ray-traced renderer (compute kernels, NOT the raster impostor
//  path — the raster pipeline stays the interactive/GUI renderer and the M3
//  golden reference). Traverses the simulation's own spatial hash (CellSim
//  grid, bin size = roi) with an Amanatides–Woo 3D-DDA per ray; the grid is
//  built ONCE for the final state (CellSim.encodeGridRebuild) by the caller.
//
//  Two styles (paper §5/§6):
//    solid — nearest sphere hit + Lambertian shading under a uniform
//            omnidirectional diffuse light, implemented as analytic
//            solid-angle ambient occlusion (paper Fig. 8 look);
//    xray  — per-ray accumulated density through the whole structure,
//            film-curve tone mapped to grayscale (paper Fig. 9 look).
//
//  The image is produced in tileSize × tileSize tiles to bound intermediate
//  memory: per tile ONE bgra8Unorm texture (tileSize² × 4 B = 4 MB at 1024)
//  plus the full-image byte buffer on the host (8600² × 4 B ≈ 296 MB). AA
//  samples accumulate per-thread in registers inside the kernel — no float
//  intermediate buffers at all. Tiling is automatic for any image dimension
//  larger than tileSize (the --tiled CLI flag is an explicit marker; it does
//  not change the tile size).

import Foundation
import Metal
import simd

/// Tuning settings for one ray-traced render. Defaults are the M4-tuned look
/// values (see the M4 report); goldens record them in their meta file so a
/// later default change cannot silently break a stored golden.
struct RayRenderSettings {
    enum Style: UInt32 {
        case solid = 0
        case xray = 1
        case foam = 2          // V2.1b: power-diagram foam chambers, translucent composite
        case foamXray = 3      // V2.1b: X-ray over the foam chambers + wall-crossing membranes
    }

    var style: Style = .solid
    var width: Int = 1600
    var height: Int = 1200
    /// Anti-aliasing samples per pixel (deterministic hash jitter).
    var samples: Int = 1
    // --- solid style ---
    /// Occlusion query radius (world units; linkRestLength = 1). 6 rest
    /// lengths — tuned against paper Fig. 8 (M4 report): smooth large-scale
    /// sculptural shading; smaller radii give a per-bead look at 100k cells.
    var aoRadius: Float = 9.6
    /// Scale on the summed solid-angle occlusion (overlapping cones
    /// over-count; tuned so exposed lobes sit near albedo like Fig. 8).
    var aoStrength: Float = 0.35
    /// Brightness floor inside crevices (albedo × floor at full occlusion).
    var aoFloor: Float = 0.18
    var albedo = SIMD3<Float>(0.87, 0.85, 0.82)
    var backgroundSolid = SIMD3<Float>(1, 1, 1)   // paper-like neutral white
    // --- xray style ---
    /// Film curve exposure: L = 1 - exp(-density · exposure). Tuned against
    /// paper Fig. 9 (M4 report): 0.1 with the skirt-inclusive density
    /// (profileSupport 2) gives the bright-knot/wispy-envelope microscopy
    /// look at ~100k cells.
    var xrayExposure: Float = 0.1
    /// Display gamma applied after the film curve (L^(1/gamma); 1 = linear).
    var xrayGamma: Float = 1.0
    /// Extra scale on the accumulated density before the film curve.
    var densityScale: Float = 1.0
    /// X-ray per-sphere density profile support, in cell radii (2 = blobs
    /// twice the cell radius, so neighbours blend into the paper's cloudy
    /// look; 1 = hard chord integral).
    var profileSupport: Float = 2.0
    var backgroundXray = SIMD3<Float>(0, 0, 0)
    // --- V2.1b foam (the painted V2.1 edge layer is RETIRED — the foam walls
    // are the wall visualization; see the chamber geometry in Shaders.swift) ---
    /// Per-face opacity of the front-to-back chamber composite (solid foam).
    /// 0.5 also makes the interactive impostor path's two-layer blending
    /// order-insensitive for the front pair. The soap-froth depth: with 0.5
    /// the first wall behind a cap weighs 25%, the next layer 12.5%.
    var foamOpacity: Float = 0.5
    /// Max composited chamber faces per ray (the kernel clamps to 8). With
    /// opacity 0.5 the transmittance after 6 faces is 1.6% — the cut is
    /// invisible; dense grown foams can hold 10+ chambers per ray.
    var foamLayers: Int = 6
    /// Foam-xray wall-crossing spike gain (obliquity-weighted thin-film term;
    /// see traverseGrid): the internal membrane lines of the microscopy look.
    /// Tuned on the 84k grown state: 1.5 saturates the film curve, 0.15 is
    /// faint; 0.25 keeps the membranes readable inside the density cloud.
    var foamWall: Float = 0.25
    // --- shared ---
    /// Seed for the deterministic AA jitter.
    var seed: UInt32 = 991
    var camera = OrbitCamera()
    // --- M5 unified light ---
    /// nil = the paper's directionally unbiased omnidirectional solid look
    /// (default; M4 goldens byte-identical). Non-nil = shade the solid style
    /// with this direction — the SAME light vector the growth illumination
    /// uses (auto-selected in light-growth mode; see HeadlessRunner).
    var directionalLight: SIMD3<Float>? = nil
    // --- M6 stereo pairs (paper §6) ---
    /// When true, render() produces TWO images with the eye shifted along the
    /// camera RIGHT axis by ±interocular/2 (parallel-axis stereo, no toe-in)
    /// and composites them side by side (left eye on the left): output is
    /// (2×width)×height. Left = eye − right·io/2, right = eye + right·io/2.
    var stereo: Bool = false
    /// Eye separation as a FRACTION of the scene scale (the bounding-sphere
    /// radius), applied by the caller — default 0.03 ("~0.03× scene scale").
    var interocular: Float = 0.03
    /// Lateral eye offset in world units (set internally per eye; do not set
    /// by hand).
    var eyeShift: SIMD3<Float> = .zero
    // --- V2.1 ---
    /// Per-pixel shadow strength in DIRECTIONAL solid mode (0 = shade with
    /// the per-cell illumination fraction only, no shadow rays; 1 = full
    /// per-pixel penumbra). Shadows exist ONLY under the directional light —
    /// unbiased renders never trace shadow rays (M4 goldens byte-identical).
    var shadows: Float = 1.0
    /// Jittered shadow rays per hit pixel (4–8; cost ~ linear).
    var shadowSamples: Int = 4
    /// Coloured X-ray: per-channel tinted density (contribution = the cell's
    /// tint × its density term, tone mapped per channel). false = the legacy
    /// grayscale accumulation (bit-identical to M4).
    var xrayColored: Bool = false
}

/// Must mirror MSL `struct RenderParams` (Shaders.raytrace) EXACTLY.
/// float4 members on both sides so there is no float3-in-struct alignment
/// ambiguity (Swift SIMD3<Float> aligns to 16; MSL float3 struct layout is a
/// documented trap). 20 scalar fields (80 B) + 8 float4 (128 B) + 2 uint (8 B)
/// = 216 bytes, + the V2.1 tail (4 x 4 B) = 232 bytes, + the V2.1b tail
/// (2 x 4 B floats + 2 x 4 B uints) = 248 bytes. The retired V2.1
/// edges/edgeWidth fields were removed in V2.1b (the foam walls replaced the
/// painted edge layer).
/// Render-only uniform block — deliberately NOT appended to SimParams (that
/// struct is the simulation contract; render tuning changes freely here).
struct RayRenderParams {
    var imageW: UInt32 = 0
    var imageH: UInt32 = 0
    var tileX: UInt32 = 0
    var tileY: UInt32 = 0
    var tileW: UInt32 = 0
    var tileH: UInt32 = 0
    var samples: UInt32 = 1
    var style: UInt32 = 0
    var aoRadius: Float = 4.8
    var aoStrength: Float = 0.85
    var aoFloor: Float = 0.15
    var xrayExposure: Float = 0.1
    var xrayGamma: Float = 1.0
    var tanHalfFov: Float = 0
    var boundingRadius: Float = 0
    var renderSeed: UInt32 = 0
    var densityScale: Float = 1
    var profileSupport: Float = 2
    var bigCount: UInt32 = 0
    var pad2: UInt32 = 0
    var background = SIMD4<Float>(1, 1, 1, 1)
    var camEye = SIMD4<Float>(0, 0, 0, 1)
    var camRight = SIMD4<Float>(1, 0, 0, 0)
    var camUp = SIMD4<Float>(0, 1, 0, 0)
    var camFwd = SIMD4<Float>(0, 0, 1, 0)
    var boundingCentre = SIMD4<Float>(0, 0, 0, 1)
    var albedo = SIMD4<Float>(0.87, 0.85, 0.82, 0)   // RETIRED M5 (per-cell tint buffer); kept for layout compatibility
    var lightDir = SIMD4<Float>(0, 1, 0, 0)          // M5 unified light (unit, toward the light)
    var directionalLight: UInt32 = 0                 // M5: 0 unbiased (default), 1 shared directional
    var pad3: UInt32 = 0
    // --- V2.1 (appended) ---
    var shadows: Float = 1.0        // per-pixel shadow strength (directional only)
    var shadowSamples: UInt32 = 4   // jittered shadow rays per hit pixel
    var xrayColored: UInt32 = 0     // 0 grayscale (legacy), 1 per-channel tinted
    var pad4: UInt32 = 0
    // --- V2.1b (appended; the retired edges/edgeWidth fields were removed) ---
    var foamOpacity: Float = 0.5    // per-face opacity of the foam composite
    var foamLayers: UInt32 = 6      // max composited chamber faces per ray (<= 8)
    var foamWall: Float = 0.25       // foam-xray wall-crossing spike gain
    var pad5: UInt32 = 0
}

// MSL's .xyz swizzle for the float4 members of RayRenderParams.
private extension SIMD4<Float> {
    var xyz: SIMD3<Float> { SIMD3(x, y, z) }
}

final class RayRenderer {

    /// Tile edge in pixels. Tile texture memory = tileSize² × 4 B = 4 MB;
    /// the only intermediate memory besides the host-side output buffer.
    static let tileSize = 1024

    let device: MTLDevice
    let queue: MTLCommandQueue
    let sim: CellSim

    private var pRaytrace: MTLComputePipelineState!
    private var pProbe: MTLComputePipelineState!
    private let paramsBuffer: MTLBuffer
    private var tileTexture: MTLTexture!
    /// Big-cell supplement indices (rad >= roi): grown structures contain
    /// stretched cells outside the grid traversal's correctness domain. Built
    /// per render (not per step); empty for pristine balls.
    private var bigCellsBuffer: MTLBuffer!

    /// Expected byte size of RayRenderParams (guards layout drift).
    private static let paramsSize = 248

    init(device: MTLDevice, queue: MTLCommandQueue, sim: CellSim) {
        self.device = device
        self.queue = queue
        self.sim = sim

        precondition(MemoryLayout<RayRenderParams>.size == Self.paramsSize,
                     "RayRenderParams layout drift: \(MemoryLayout<RayRenderParams>.size) != \(Self.paramsSize)")

        guard let buf = device.makeBuffer(length: Self.paramsSize, options: .storageModeShared) else {
            fatalError("RayRenderer: params buffer alloc failed")
        }
        buf.label = "rayRenderParams"
        paramsBuffer = buf

        do {
            let lib = try device.makeLibrary(source: Shaders.raytrace, options: nil)
            guard let fn = lib.makeFunction(name: "raytraceTile") else {
                fatalError("RayRenderer: missing kernel raytraceTile")
            }
            pRaytrace = try device.makeComputePipelineState(function: fn)
            guard let pfn = lib.makeFunction(name: "raytraceProbe") else {
                fatalError("RayRenderer: missing kernel raytraceProbe")
            }
            pProbe = try device.makeComputePipelineState(function: pfn)
        } catch {
            fatalError("RayRenderer: failed to build pipeline: \(error)")
        }
    }

    /// Compute the uniform block for a render (bounding sphere from the live
    /// buffers + camera basis). GPU must be idle (reads positions/radii).
    func makeParams(settings: RayRenderSettings) -> RayRenderParams {
        // Scene bounding sphere: inflated by max(|P-c| + rad) PLUS one extra
        // max cell radius. The extra radius guarantees every cell sphere lies
        // strictly inside with margin >= maxRadius, so every sphere's ray
        // interval [ta, tb] ends strictly inside [t0, t1] — without it, the
        // cell that DEFINES the bounding sphere has margin exactly 0 and its
        // inclusion flips on 1-ulp differences between the CPU's and the
        // GPU's dot products (measured: 0.26 density flicker on grown states).
        var maxRad: Float = 0
        var lo = SIMD3<Float>(Float.greatestFiniteMagnitude, 0, 0)
        var hi = SIMD3<Float>(-Float.greatestFiniteMagnitude, 0, 0)
        let pos = sim.positionsBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
        let rad = sim.radiiBuffer.contents().assumingMemoryBound(to: Float.self)
        let deadFlags = sim.deadFlagsPointer   // V2.5: dead slots are not scene content
        for i in 0..<sim.cellCount where deadFlags[i] == 0 {
            let p = pos[i]
            maxRad = max(maxRad, rad[i])
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        let centre = (lo + hi) * 0.5
        var boundingRadius: Float = 0
        for i in 0..<sim.cellCount where deadFlags[i] == 0 {
            boundingRadius = max(boundingRadius, simd_distance(pos[i], centre) + rad[i])
        }
        boundingRadius += maxRad + 1e-3

        // Camera basis: EXACTLY the same convention as Renderer.updateCameraBuffer.
        // M6 stereo: the eye is shifted along the camera RIGHT axis
        // (settings.eyeShift, set by renderStereo; zero for monocular).
        let cam = settings.camera
        let eye = cam.eye + settings.eyeShift
        let fwd = simd_normalize(cam.target - eye)
        let worldUp = SIMD3<Float>(0, 1, 0)
        let right = simd_normalize(simd_cross(fwd, worldUp))
        let up = simd_cross(right, fwd)

        var p = RayRenderParams()
        p.imageW = UInt32(settings.width)
        p.imageH = UInt32(settings.height)
        p.samples = UInt32(max(1, settings.samples))
        p.style = settings.style.rawValue
        p.aoRadius = settings.aoRadius
        p.aoStrength = settings.aoStrength
        p.aoFloor = settings.aoFloor
        p.xrayExposure = settings.xrayExposure
        p.xrayGamma = settings.xrayGamma
        p.tanHalfFov = tanf(cam.fovY * 0.5)
        p.boundingRadius = boundingRadius
        p.renderSeed = settings.seed
        p.densityScale = settings.densityScale
        p.profileSupport = settings.profileSupport
        p.background = SIMD4(settings.style == .xray || settings.style == .foamXray
                             ? settings.backgroundXray : settings.backgroundSolid, 1)
        p.camEye = SIMD4(eye, 1)
        p.camRight = SIMD4(right, 0)
        p.camUp = SIMD4(up, 0)
        p.camFwd = SIMD4(fwd, 0)
        p.boundingCentre = SIMD4(centre, 1)
        p.albedo = SIMD4(settings.albedo, 0)
        // V2.1 render knobs.
        p.shadows = settings.shadows
        p.shadowSamples = UInt32(max(0, settings.shadowSamples))
        p.xrayColored = settings.xrayColored ? 1 : 0
        // V2.1b foam knobs.
        p.foamOpacity = max(min(settings.foamOpacity, 1), 0.02)
        p.foamLayers = UInt32(max(1, min(settings.foamLayers, 8)))
        p.foamWall = max(settings.foamWall, 0)
        // M5 unified light: nil = unbiased (paper default); otherwise shade
        // with the shared growth light direction.
        if let dl = settings.directionalLight {
            let l = simd_normalize(dl)
            p.lightDir = SIMD4(l, 0)
            p.directionalLight = 1
        }
        // Big-cell supplement: radii >= roi live outside the grid traversal's
        // correctness domain (crossing point > 1 bin from the centre bin).
        // V2.5: dead slots are excluded (their zeroed radius fails the filter
        // anyway — the explicit flag check keeps the two passes consistent).
        let roi = sim.params.roi
        var bigList: [UInt32] = []
        for i in 0..<sim.cellCount where rad[i] >= roi && deadFlags[i] == 0 {
            bigList.append(UInt32(i))
        }
        // Buffer always exists (the kernel binds it unconditionally; a nil
        // binding is legal but unnecessary here).
        if bigCellsBuffer == nil || bigCellsBuffer.length < max(64, bigList.count) * MemoryLayout<UInt32>.stride {
            guard let buf = device.makeBuffer(length: max(64, bigList.count) * MemoryLayout<UInt32>.stride,
                                              options: .storageModeShared) else {
                fatalError("RayRenderer: bigCells buffer alloc failed")
            }
            buf.label = "rayBigCells"
            bigCellsBuffer = buf
        }
        if bigList.count > 0 {
            bigList.withUnsafeBytes { src in
                bigCellsBuffer.contents().copyMemory(from: src.baseAddress!,
                                                     byteCount: bigList.count * MemoryLayout<UInt32>.stride)
            }
        }
        p.bigCount = UInt32(bigList.count)
        if bigList.count > 0 {
            print("RayRenderer: \(bigList.count) big cells (rad >= roi \(roi)) handled by the supplement pass")
        }
        return p
    }

    /// Render the CURRENT state (host must be in sync: GPU idle, radii fresh,
    /// grid rebuilt via CellSim.encodeGridRebuild). With settings.stereo the
    /// output is a side-by-side composite (left | right), width doubled.
    /// Returns BGRA8 bytes (same memory order the raster offscreen path uses)
    /// + GPU seconds.
    func render(settings: RayRenderSettings) -> (bytes: [UInt8], width: Int, height: Int, gpuSeconds: Double) {
        guard !settings.stereo else { return renderStereo(settings: settings) }
        return renderSingle(settings: settings)
    }

    /// Stereo: two parallel-axis renders (±interocular/2 along camera right),
    /// composited side by side. Each eye renders the FULL image; the scene
    /// bounding sphere is shared (it depends only on the cells).
    private func renderStereo(settings: RayRenderSettings) -> (bytes: [UInt8], width: Int, height: Int, gpuSeconds: Double) {
        // Camera basis (same convention as makeParams).
        let cam = settings.camera
        let fwd = simd_normalize(cam.target - cam.eye)
        let right = simd_normalize(simd_cross(fwd, SIMD3<Float>(0, 1, 0)))
        let io = settings.interocular * sceneScale(settings: settings) * 0.5

        var left = settings
        left.eyeShift = -right * io
        var rightSettings = settings
        rightSettings.eyeShift = right * io

        let l = renderSingle(settings: left)
        let r = renderSingle(settings: rightSettings)
        precondition(l.width == r.width && l.height == r.height, "stereo eye sizes diverged")
        let w = l.width, h = l.height
        var out = [UInt8](repeating: 0, count: w * 2 * h * 4)
        for row in 0..<h {
            // Left eye into columns [0, w), right eye into [w, 2w).
            out.withUnsafeMutableBytes { dst in
                l.bytes.withUnsafeBytes { src in
                    memcpy(dst.baseAddress! + row * 2 * w * 4,
                           src.baseAddress! + row * w * 4, w * 4)
                }
                r.bytes.withUnsafeBytes { src in
                    memcpy(dst.baseAddress! + (row * 2 * w + w) * 4,
                           src.baseAddress! + row * w * 4, w * 4)
                }
            }
        }
        return (out, w * 2, h, l.gpuSeconds + r.gpuSeconds)
    }

    /// Scene scale for the interocular fraction: the bounding-sphere radius
    /// (the same quantity makeParams computes; duplicated host-side here).
    private func sceneScale(settings: RayRenderSettings) -> Float {
        let pos = sim.positionsBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
        let rad = sim.radiiBuffer.contents().assumingMemoryBound(to: Float.self)
        var lo = SIMD3<Float>(Float.greatestFiniteMagnitude, 0, 0)
        var hi = SIMD3<Float>(-Float.greatestFiniteMagnitude, 0, 0)
        for i in 0..<sim.cellCount {
            let p = pos[i]
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        let centre = (lo + hi) * 0.5
        var boundingRadius: Float = 0
        for i in 0..<sim.cellCount {
            boundingRadius = max(boundingRadius, simd_distance(pos[i], centre) + rad[i])
        }
        return boundingRadius
    }

    /// Single-eye tile loop (the M4 renderer, unchanged semantics).
    private func renderSingle(settings: RayRenderSettings) -> (bytes: [UInt8], width: Int, height: Int, gpuSeconds: Double) {
        precondition(sim.cellCount > 0, "RayRenderer: no cells")
        let (w, h) = (settings.width, settings.height)

        var p = makeParams(settings: settings)
        withUnsafeBytes(of: &p) { src in
            paramsBuffer.contents().copyMemory(from: src.baseAddress!,
                                               byteCount: MemoryLayout<RayRenderParams>.size)
        }

        // One tile texture, reused across tiles.
        let tw0 = min(Self.tileSize, w)
        let th0 = min(Self.tileSize, h)
        let tdesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                             width: tw0, height: th0,
                                                             mipmapped: false)
        tdesc.usage = [.shaderWrite, .shaderRead]
        tdesc.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: tdesc) else {
            fatalError("RayRenderer: tile texture alloc failed")
        }
        tileTexture = tex

        var out = [UInt8](repeating: 0, count: w * h * 4)
        var tileBytes = [UInt8](repeating: 0, count: tw0 * th0 * 4)
        var gpuSeconds = 0.0
        var tiles = 0

        var ty = 0
        while ty < h {
            var tx = 0
            while tx < w {
                let tw = min(tw0, w - tx)
                let th = min(th0, h - ty)
                p.tileX = UInt32(tx)
                p.tileY = UInt32(ty)
                p.tileW = UInt32(tw)
                p.tileH = UInt32(th)
                withUnsafeBytes(of: &p) { src in
                    paramsBuffer.contents().copyMemory(from: src.baseAddress!,
                                                       byteCount: MemoryLayout<RayRenderParams>.size)
                }

                guard let cb = queue.makeCommandBuffer() else { fatalError("RayRenderer: no command buffer") }
                guard let enc = cb.makeComputeCommandEncoder() else { fatalError("RayRenderer: no encoder") }
                enc.setComputePipelineState(pRaytrace)
                enc.setTexture(tex, index: 0)
                enc.setBuffer(sim.positionsBuffer, offset: 0, index: 0)
                enc.setBuffer(sim.radiiBuffer, offset: 0, index: 1)
                enc.setBuffer(sim.binStartsBuffer, offset: 0, index: 2)
                enc.setBuffer(sim.particleIndicesBuffer, offset: 0, index: 3)
                enc.setBuffer(sim.counterBuffer, offset: 0, index: 4)
                enc.setBuffer(paramsBuffer, offset: 0, index: 5)
                enc.setBuffer(sim.paramsBuffer, offset: 0, index: 6)
                enc.setBuffer(bigCellsBuffer, offset: 0, index: 7)
                enc.setBuffer(sim.tintBuffer, offset: 0, index: 8)   // M5 per-cell colour
                enc.setBuffer(sim.litFractionBuffer, offset: 0, index: 9)   // V2.1 illumination fractions
                enc.setBuffer(sim.occupancyBuffer, offset: 0, index: 10)   // M6 coarse skip (shadow marches)
                enc.setBuffer(sim.linksBuffer, offset: 0, index: 11)        // V2.1b foam CSG (rings)
                enc.setBuffer(sim.linkCountsBuffer, offset: 0, index: 12)   // V2.1b
                enc.dispatchThreads(MTLSize(width: tw, height: th, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                if let e = cb.error {
                    fatalError("RayRenderer tile (\(tx),\(ty)) CB ERROR: \(e)")
                }
                gpuSeconds += cb.gpuEndTime - cb.gpuStartTime
                tiles += 1

                tileBytes.withUnsafeMutableBytes { dst in
                    tex.getBytes(dst.baseAddress!, bytesPerRow: tw * 4,
                                 from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                                 size: MTLSize(width: tw, height: th, depth: 1)),
                                 mipmapLevel: 0)
                }
                // Blit the tile rows into the full-image buffer.
                out.withUnsafeMutableBytes { dst in
                    tileBytes.withUnsafeBytes { src in
                        let d = dst.baseAddress!.advanced(by: (ty * w + tx) * 4)
                        let s = src.baseAddress!
                        for row in 0..<th {
                            memcpy(d + row * w * 4, s + row * tw * 4, tw * 4)
                        }
                    }
                }
                tx += tw0
            }
            ty += th0
        }
        if tiles > 1 {
            print("RayRenderer: \(tiles) tiles of \(Self.tileSize)²")
        }
        return (out, w, h, gpuSeconds)
    }

    // MARK: - CGDBG CPU verification

    /// GPU one-pixel probe (raytraceProbe kernel): returns the 64-float dump.
    /// GPU idle; settings must match the render being investigated.
    func debugProbePixel(px: Int, py: Int, settings: RayRenderSettings,
                         probeCells: [UInt32] = []) -> [Float] {
        var p = makeParams(settings: settings)
        withUnsafeBytes(of: &p) { src in
            paramsBuffer.contents().copyMemory(from: src.baseAddress!,
                                               byteCount: MemoryLayout<RayRenderParams>.size)
        }
        guard let outBuf = device.makeBuffer(length: 96 * MemoryLayout<Float>.stride,
                                             options: .storageModeShared),
              let pxBuf = device.makeBuffer(length: 10 * MemoryLayout<UInt32>.stride,
                                            options: .storageModeShared),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            fatalError("RayRenderer: probe alloc failed")
        }
        let pxPtr = pxBuf.contents().assumingMemoryBound(to: UInt32.self)
        pxPtr[0] = UInt32(px)
        pxPtr[1] = UInt32(py)
        for k in 0..<8 {
            pxPtr[2 + k] = k < probeCells.count ? probeCells[k] : 0xFFFFFFFF
        }
        enc.setComputePipelineState(pProbe)
        enc.setBuffer(outBuf, offset: 0, index: 0)
        enc.setBuffer(sim.positionsBuffer, offset: 0, index: 1)
        enc.setBuffer(sim.radiiBuffer, offset: 0, index: 2)
        enc.setBuffer(sim.binStartsBuffer, offset: 0, index: 3)
        enc.setBuffer(sim.particleIndicesBuffer, offset: 0, index: 4)
        enc.setBuffer(sim.counterBuffer, offset: 0, index: 5)
        enc.setBuffer(paramsBuffer, offset: 0, index: 6)
        enc.setBuffer(sim.paramsBuffer, offset: 0, index: 7)
        enc.setBuffer(bigCellsBuffer, offset: 0, index: 8)
        enc.setBuffer(pxBuf, offset: 0, index: 9)
        enc.setBuffer(sim.tintBuffer, offset: 0, index: 10)   // V2.1: traverseGrid signature
        enc.setBuffer(sim.linksBuffer, offset: 0, index: 11)   // V2.1b: traverseGrid signature
        enc.setBuffer(sim.linkCountsBuffer, offset: 0, index: 12)   // V2.1b
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { print("CGDBG probe CB ERROR: \(e)") }
        let o = outBuf.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: o, count: 96))
    }

    // PCG-style integer hash — MUST match the MSL jitterHash bit-for-bit.
    static func jitterHash(_ x: UInt32) -> UInt32 {
        var state = x &* 747796405 &+ 2891336453
        let word = ((state >> ((state >> 28) + 4)) ^ state) &* 277803737
        return (word >> 22) ^ word
    }
    static func sampleJitter(px: UInt32, py: UInt32, s: UInt32, seed: UInt32) -> SIMD2<Float> {
        let h = jitterHash(px &* 1973 ^ py &* 9277 ^ s &* 26699 ^ seed &* 7919)
        return SIMD2(Float(h & 0xFFFF) / 65535.0 - 0.5,
                     Float((h >> 16) & 0xFFFF) / 65535.0 - 0.5)
    }

    /// CGDBG diagnostics: recompute a sample of pixels on the CPU (brute
    /// force over ALL cells — no grid, each cell enumerated exactly once, so
    /// no dedup machinery exists to match) and compare against the rendered
    /// bytes. Validates the DDA traversal, the X-ray entry-anchor dedup
    /// (GPU total must equal the CPU total), and the AO sum. GPU idle.
    /// V2.1b: solid/xray only — the foam styles' chamber CSG has no CPU
    /// reference here (their validation is the golden + cross-renderer
    /// consistency tests); a CGDBG run on foam would produce bogus diffs.
    func debugVerify(bytes: [UInt8], settings: RayRenderSettings, pixelCount: Int) {
        guard settings.style == .solid || settings.style == .xray else {
            print("CGDBG verify: skipped (style \(settings.style.rawValue) foam has no CPU reference here)")
            return
        }
        let p = makeParams(settings: settings)
        let w = Int(p.imageW)
        let h = Int(p.imageH)
        let pos = sim.positionsBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
        let radf = sim.radiiBuffer.contents().assumingMemoryBound(to: Float.self)
        let n = sim.cellCount
        let solid = p.style == 0
        let ro = p.camEye.xyz
        let right = p.camRight.xyz
        let up = p.camUp.xyz
        let fwd = p.camFwd.xyz

        var maxDiff: Float = 0
        var sumDiff: Float = 0
        var checked = 0
        var worstPx = 0
        var worstPy = 0
        var cpuNearest: Float = -1
        var cpuHitIdx = -1
        var cpuOcc: Float = -1
        var cpuVisited = 0
        var cpuRo = SIMD3<Float>(0, 0, 0)
        var cpuRd = SIMD3<Float>(0, 0, 0)
        var wNearest: Float = -1
        var wHitIdx = -1
        var wOcc: Float = -1
        var wVisited = 0
        var wDens: Float = -1
        var wTop: [(Int, Float)] = []
        for k in 0..<pixelCount {
            let px = (UInt32(k) &* 7919) % p.imageW
            let py = (UInt32(k) &* 104729) % p.imageH
            let aspect = Float(w) / Float(h)
            var accumColor = SIMD3<Float>(0, 0, 0)
            var densAccum: Float = 0
            var lastRd = SIMD3<Float>(0, 0, 0)
            var lastDens: Float = 0
            var topLocal: [(Int, Float)] = []
            for s in 0..<Int(p.samples) {
                let j = Self.sampleJitter(px: px, py: py, s: UInt32(s), seed: p.renderSeed)
                let ndcx = (2.0 * (Float(px) + j.x) / Float(w) - 1.0) * p.tanHalfFov * aspect
                let ndcy = (1.0 - 2.0 * (Float(py) + j.y) / Float(h)) * p.tanHalfFov
                let dirV = simd_normalize(SIMD3<Float>(ndcx, ndcy, -1))
                let rd = simd_normalize(dirV.x * right + dirV.y * up - dirV.z * fwd)
                lastRd = rd

                // Clip to the bounding sphere (same as the kernel).
                let b = p.boundingCentre.xyz - ro
                let tc = simd_dot(b, rd)
                let d2 = simd_dot(b, b) - tc * tc
                if d2 > p.boundingRadius * p.boundingRadius {
                    if solid { accumColor += p.background.xyz }
                    continue
                }
                let thc = sqrt(p.boundingRadius * p.boundingRadius - d2)
                let t0 = max(tc - thc, 0)
                let t1 = tc + thc

                // Brute force over ALL cells.
                var nearest = Float.infinity
                var nearestIdx = -1
                var dens: Float = 0
                for i in 0..<n {
                    let rad = radf[i]
                    if rad <= 0 { continue }
                    let support = rad * p.profileSupport
                    let C = pos[i]
                    let toC = C - ro
                    let tcl = simd_dot(toC, rd)
                    let dLine2 = simd_dot(toC, toC) - tcl * tcl
                    // Density filters at the profile support (C1 edge, matches
                    // the GPU paths); the nearest hit at the sphere test.
                    let r2test = solid ? rad * rad : support * support
                    if dLine2 > r2test { continue }
                    if solid {
                        let th = sqrt(rad * rad - dLine2)
                        let ta = tcl - th
                        let tb = tcl + th
                        if tb > t0 && ta < t1 && ta > t0 && ta < nearest {
                            nearest = ta; nearestIdx = i
                        }
                    }
                    if !solid {
                        // CPU enumerates each cell once: the GPU anchor-dedup
                        // total must equal this (the actual dedup check).
                        let tf = min(max(tcl, t0), t1)
                        let dv = C - (ro + rd * tf)
                        let dd2 = simd_dot(dv, dv)
                        if dd2 < support * support {
                            let q = 1.0 - dd2 / (support * support)
                            let c = rad * q * q
                            dens += c
                            if c > 0.03 {
                                topLocal.append((i, c))
                                if topLocal.count > 8 {
                                    topLocal.sort { $0.1 > $1.1 }
                                    topLocal.removeLast()
                                }
                            }
                        }
                    }
                }

                if solid {
                    if nearest < .infinity, nearestIdx >= 0 {
                        let C = pos[nearestIdx]
                        let rad = radf[nearestIdx]
                        let Ph = ro + rd * nearest
                        let N = (Ph - C) / max(rad, 1e-9)
                        var occ: Float = 0
                        var visited: Int = 0
                        for i in 0..<n {
                            let rad2 = radf[i]
                            if rad2 <= 0 { continue }
                            let dv = pos[i] - Ph
                            let dd2 = simd_dot(dv, dv)
                            if dd2 < 1e-12 || dd2 > p.aoRadius * p.aoRadius { continue }
                            visited += 1
                            let invD = 1.0 / sqrt(dd2)
                            let cosT = simd_dot(N, dv) * invD
                            if cosT <= 0 { continue }
                            occ += (rad2 * rad2 * invD * invD) * cosT
                        }
                        let ao = min(max(1.0 - p.aoStrength * occ, 0), 1)
                        accumColor += p.albedo.xyz * (p.aoFloor + (1.0 - p.aoFloor) * ao)
                        cpuNearest = nearest
                        cpuHitIdx = nearestIdx
                        cpuOcc = occ
                        cpuVisited = visited
                    } else {
                        accumColor += p.background.xyz
                        cpuNearest = -1
                        cpuHitIdx = -1
                        cpuOcc = -1
                        cpuVisited = 0
                    }
                } else {
                    densAccum += dens
                    lastDens = dens
                }
            }

            // Expected pixel value (same tone mapping as the kernel).
            var expected: SIMD3<Float>
            if solid {
                expected = accumColor / Float(max(Int(p.samples), 1))
            } else {
                let dens = densAccum / Float(max(Int(p.samples), 1)) * p.densityScale
                var L = 1.0 - exp(-dens * p.xrayExposure)
                L = pow(L, 1.0 / max(p.xrayGamma, 1e-3))
                expected = SIMD3(L, L, L)
            }
            expected = simd_clamp(expected, SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 1, 1))

            // Actual bytes (BGRA order).
            let off = (Int(py) * w + Int(px)) * 4
            let actual = SIMD3<Float>(Float(bytes[off + 2]) / 255.0,
                                      Float(bytes[off + 1]) / 255.0,
                                      Float(bytes[off + 0]) / 255.0)
            let diff = simd_length(abs(actual - expected))
            if diff > maxDiff {
                maxDiff = diff
                worstPx = Int(px)
                worstPy = Int(py)
                cpuRo = ro
                cpuRd = lastRd
                wNearest = solid ? cpuNearest : -1
                wHitIdx = solid ? cpuHitIdx : -1
                wOcc = cpuOcc
                wVisited = cpuVisited
                wDens = solid ? -1 : lastDens
                wTop = topLocal
            }
            sumDiff += diff
            checked += 1
            if k < 8 {
                print("CGDBG px (\(px),\(py)): expected (\(String(format: "%.3f", expected.x)), \(String(format: "%.3f", expected.y)), \(String(format: "%.3f", expected.z)))  actual (\(String(format: "%.3f", actual.x)), \(String(format: "%.3f", actual.y)), \(String(format: "%.3f", actual.z)))  |Δ|=\(String(format: "%.3f", diff))")
            }
        }
        print("CGDBG verify (\(solid ? "solid" : "xray"), \(checked) px vs CPU brute force): mean |Δ| = \(String(format: "%.2e", checked > 0 ? sumDiff / Float(checked) : 0)), max |Δ| = \(String(format: "%.2e", maxDiff)) (greyscale-length units)")

        // Full instrumentation on the worst pixel: GPU probe dump vs CPU values.
        if maxDiff > 1e-3 {
            let d = debugProbePixel(px: worstPx, py: worstPy, settings: settings,
                                    probeCells: wTop.map { UInt32($0.0) })
            print("CGDBG worst px (\(worstPx),\(worstPy)) GPU probe: t0=\(String(format: "%.3f", d[0])) t1=\(String(format: "%.3f", d[1])) nearest=\(String(format: "%.3f", d[2])) hitIdx=\(Int(d[3])) xrayDens=\(String(format: "%.4f", d[4]))")
            print("   Ph=(\(String(format: "%.3f", d[5])), \(String(format: "%.3f", d[6])), \(String(format: "%.3f", d[7])))  N=(\(String(format: "%.3f", d[8])), \(String(format: "%.3f", d[9])), \(String(format: "%.3f", d[10])))")
            print("   AO cube lo=(\(Int(d[11])),\(Int(d[12])),\(Int(d[13]))) hi=(\(Int(d[14])),\(Int(d[15])),\(Int(d[16])))  visited=\(Int(d[49])) kept=\(Int(d[50])) occ=\(String(format: "%.4f", d[51]))")
            for c in 0..<8 {
                let idx = Int(d[17 + c * 4])
                if idx < 0 { break }
                print("   occluder \(c): cell \(idx) d=\(String(format: "%.3f", d[17 + c * 4 + 1])) cosT=\(String(format: "%.3f", d[17 + c * 4 + 2])) rad=\(String(format: "%.3f", d[17 + c * 4 + 3]))")
            }
            print("   ro=(\(String(format: "%.4f", d[52])), \(String(format: "%.4f", d[53])), \(String(format: "%.4f", d[54])))  rd=(\(String(format: "%.6f", d[55])), \(String(format: "%.6f", d[56])), \(String(format: "%.6f", d[57])))  jitter=(\(String(format: "%.4f", d[58])), \(String(format: "%.4f", d[59])))")
            print("   CPU ro=(\(String(format: "%.4f", cpuRo.x)), \(String(format: "%.4f", cpuRo.y)), \(String(format: "%.4f", cpuRo.z)))  rd=(\(String(format: "%.6f", cpuRd.x)), \(String(format: "%.6f", cpuRd.y)), \(String(format: "%.6f", cpuRd.z)))")
            // CPU evaluation of the GPU's hit cell for the worst pixel.
            if Int(d[3]) > 0 && Int(d[3]) < n {
                let C = pos[Int(d[3])]
                let rad = radf[Int(d[3])]
                let toC = C - cpuRo
                let tcl = simd_dot(toC, cpuRd)
                let dLine2 = simd_dot(toC, toC) - tcl * tcl
                print("   CPU eval of GPU hit cell \(Int(d[3])): dLine2=\(String(format: "%.6f", dLine2)) rad2=\(String(format: "%.6f", rad * rad)) ta=\(String(format: "%.3f", tcl - sqrt(max(rad * rad - dLine2, 0)))) tb=\(String(format: "%.3f", tcl + sqrt(max(rad * rad - dLine2, 0))))")
                let posGPU = SIMD3<Float>(d[52 + 0], d[52 + 1], d[52 + 2])
                let rdGPU = SIMD3<Float>(d[55], d[56], d[57])
                let toG = C - posGPU
                let tcG = simd_dot(toG, rdGPU)
                let dG = simd_dot(toG, toG) - tcG * tcG
                print("   CPU eval vs GPU ray   : dLine2=\(String(format: "%.6f", dG)) (GPU said hit ⇒ dLine2 <= rad2)")
            }
            print("   GPU rp echo: px=\(Int(d[60])) py=\(Int(d[61])) renderSeed=\(d[62].bitPattern) tanHalfFov=\(String(format: "%.6f", d[63]))")
            print("   GPU hash truth: combined=\(d[64].bitPattern) hash=\(d[65].bitPattern) helper j=(\(String(format: "%.4f", d[66])), \(String(format: "%.4f", d[67])))  densBF(GPU, no grid)=\(String(format: "%.4f", d[68]))")
            for (k, cell) in wTop.enumerated() {
                let rad = d[72 + k * 3], dl = d[72 + k * 3 + 1], gc = d[72 + k * 3 + 2]
                print("   cell \(cell.0): CPU rad=\(String(format: "%.4f", radf[cell.0])) contrib=\(String(format: "%.4f", cell.1))  | GPU rad=\(String(format: "%.4f", rad)) dLine2=\(String(format: "%.4f", dl)) contrib=\(String(format: "%.4f", gc))")
            }
            print("   CPU: nearest=\(String(format: "%.3f", wNearest)) hitIdx=\(wHitIdx) occ=\(String(format: "%.4f", wOcc)) visitedInRadius=\(wVisited) dens=\(String(format: "%.4f", wDens))")
        }
    }
}

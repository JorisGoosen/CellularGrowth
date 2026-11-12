//  Renderer.swift
//  Instanced sphere-impostor renderer. Each cell draws one camera-facing quad;
//  the fragment shader shades it as a sphere with per-pixel normals and depth.
//  Usable both through an MTKView (interactive) and into an offscreen texture
//  (headless validation screenshots).

import Foundation
import Metal
import MetalKit
import simd

/// Must mirror MSL `struct Camera` (240 bytes: 208 + the V2.1 xrayPad float4
/// + the V2.1b foamPad/eyePad float4s).
struct CameraUniforms {
    var view: simd_float4x4
    var proj: simd_float4x4
    var right: SIMD4<Float>
    var up: SIMD4<Float>
    var negFwd: SIMD4<Float>
    var lightDir: SIMD4<Float>
    var xrayPad: SIMD4<Float>   // M6: .x = interactive X-ray exposure gain
    var foamPad: SIMD4<Float>   // V2.1b: .x = foam opacity (0 = the historical solid path; the fragment branches on it)
    var eyePad: SIMD4<Float>    // V2.1b: .xyz = world eye (the foam CSG's ray origin)
}

final class Renderer: NSObject, MTKViewDelegate {

    /// M6 render style: Solid = the original depth-tested impostor path
    /// (unchanged — the M3 goldens compare its offscreen output), X-ray = the
    /// additive density-sprite accumulation (interactive approximation of the
    /// offline X-ray; see cellFragmentXray in Shaders.swift). V2.1b Foam = the
    /// power-diagram CHAMBER impostors: each cell draws its chamber's first
    /// face (curved cap or flat power-bisector wall) premultiplied by the foam
    /// opacity, instances sorted back-to-front — the interactive foam-depth
    /// look (walls behind walls).
    enum RenderStyle {
        case solid
        case xray
        case foam
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let sim: CellSim

    // Plain vars, written from the main thread, read on the display-link thread.
    var isPlaying = true
    var stepsPerFrame = 4
    var renderStyle: RenderStyle = .solid
    /// Interactive X-ray exposure gain (per-sprite premultiplier; the 8-bit
    /// additive clamp is the tone map — see cellFragmentXray). Tuned against
    /// the offline X-ray of the same grown state (M6 report): 0.3 keeps dense
    /// knots from saturating to flat white while dim regions still read.
    var xrayGain: Float = 0.3
    // --- V2.1b foam (replaces the retired V2.1 edge layer) ---
    /// Per-face opacity of the foam composite. 0.5 makes the two-layer
    /// blend order-insensitive; the back-to-front instance sort handles the
    /// rest. 0 would select the historical solid path in the fragment.
    var foamOpacity: Float = 0.5

    var camera = OrbitCamera()

    private var pCellRender: MTLRenderPipelineState!
    private var pCellRenderXray: MTLRenderPipelineState!
    private var pCellRenderFoam: MTLRenderPipelineState!
    private var depthState: MTLDepthStencilState!
    private var depthStateXray: MTLDepthStencilState!   // test off, write off (shared by foam)
    private let cameraBuffer: MTLBuffer
    /// V2.1b foam: instance -> cell indirection, sorted back-to-front (the
    /// fixed-function over-blend needs far-to-near order). Preallocated for
    /// maxCells; refilled when the view/cells change (throttled on large states).
    private let foamOrderBuffer: MTLBuffer
    private var foamOrderCount = -1                     // last sorted cell count
    private var foamOrderFrame: UInt64 = 0              // last sorted frame
    private var foamOrderEye = SIMD3<Float>(.nan, 0, 0) // last sorted camera state
    private var foamOrderFwd = SIMD3<Float>(0, 0, 0)

    // Stats plumbing
    private var frameIndex: UInt64 = 0
    private var statsDue = false
    var onStats: ((Int, Float, Double) -> Void)?   // (cellCount, meanLinkLength, stepsPerSecond)
    private var lastStatsTime = CFAbsoluteTimeGetCurrent()
    private var stepsSinceLastStats = 0

    init(device: MTLDevice, queue: MTLCommandQueue, sim: CellSim) {
        self.device = device
        self.queue = queue
        self.sim = sim
        cameraBuffer = device.makeBuffer(length: MemoryLayout<CameraUniforms>.size,
                                         options: .storageModeShared)!
        cameraBuffer.label = "camera"
        foamOrderBuffer = device.makeBuffer(length: sim.maxCells * MemoryLayout<UInt32>.stride,
                                            options: .storageModeShared)!
        foamOrderBuffer.label = "foamDrawOrder"

        super.init()

        do {
            let lib = try device.makeLibrary(source: Shaders.render, options: nil)
            let vfn = lib.makeFunction(name: "cellVertex")!
            let ffn = lib.makeFunction(name: "cellFragment")!
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vfn
            desc.fragmentFunction = ffn
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            desc.depthAttachmentPixelFormat = .depth32Float
            pCellRender = try device.makeRenderPipelineState(descriptor: desc)

            // M6 X-ray: same vertex layout, density fragment, ADDITIVE
            // blending (one/one — premultiplied accumulation), no depth
            // test/write so every sprite accumulates regardless of position.
            let xvfn = lib.makeFunction(name: "cellVertexXray")!
            let xffn = lib.makeFunction(name: "cellFragmentXray")!
            let xdesc = MTLRenderPipelineDescriptor()
            xdesc.vertexFunction = xvfn
            xdesc.fragmentFunction = xffn
            xdesc.colorAttachments[0].pixelFormat = .bgra8Unorm
            xdesc.depthAttachmentPixelFormat = .depth32Float
            xdesc.colorAttachments[0].isBlendingEnabled = true
            xdesc.colorAttachments[0].sourceRGBBlendFactor = .one
            xdesc.colorAttachments[0].destinationRGBBlendFactor = .one
            xdesc.colorAttachments[0].sourceAlphaBlendFactor = .one
            xdesc.colorAttachments[0].destinationAlphaBlendFactor = .one
            xdesc.colorAttachments[0].rgbBlendOperation = .add
            xdesc.colorAttachments[0].alphaBlendOperation = .add
            pCellRenderXray = try device.makeRenderPipelineState(descriptor: xdesc)

            // V2.1b foam: the SAME fragment (it branches on foamPad.x), a
            // drawOrder-indirected vertex, PREMULTIPLIED over-blending
            // (one / one-minus-source-alpha) and NO depth test — the CPU
            // keeps the instances sorted back-to-front, which makes the
            // fixed-function over-blend the correct front-to-back chamber
            // composite.
            let fdesc = MTLRenderPipelineDescriptor()
            fdesc.vertexFunction = lib.makeFunction(name: "cellVertexFoam")!
            fdesc.fragmentFunction = ffn
            fdesc.colorAttachments[0].pixelFormat = .bgra8Unorm
            fdesc.depthAttachmentPixelFormat = .depth32Float
            fdesc.colorAttachments[0].isBlendingEnabled = true
            fdesc.colorAttachments[0].sourceRGBBlendFactor = .one
            fdesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            fdesc.colorAttachments[0].sourceAlphaBlendFactor = .one
            fdesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            fdesc.colorAttachments[0].rgbBlendOperation = .add
            fdesc.colorAttachments[0].alphaBlendOperation = .add
            pCellRenderFoam = try device.makeRenderPipelineState(descriptor: fdesc)

            let ds = MTLDepthStencilDescriptor()
            ds.depthCompareFunction = .less
            ds.isDepthWriteEnabled = true
            depthState = device.makeDepthStencilState(descriptor: ds)
            let xds = MTLDepthStencilDescriptor()
            xds.depthCompareFunction = .always
            xds.isDepthWriteEnabled = false
            depthStateXray = device.makeDepthStencilState(descriptor: xds)
        } catch {
            fatalError("Renderer: failed to build pipeline: \(error)")
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let cb = queue.makeCommandBuffer() else { return }
        // The X-ray look needs a BLACK background (density accumulation on
        // dark, like the offline style); solid keeps the studio blue-grey.
        view.clearColor = renderStyle == .xray
            ? MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            : MTLClearColor(red: 0.065, green: 0.075, blue: 0.095, alpha: 1)
        guard let rpd = view.currentRenderPassDescriptor else { return }

        runFrame(into: rpd, commandBuffer: cb, aspect: aspect(of: view.drawableSize))
        cb.present(drawable)
        cb.commit()
    }

    // MARK: - Frame assembly

    private func runFrame(into rpd: MTLRenderPassDescriptor, commandBuffer cb: MTLCommandBuffer, aspect: Float) {
        if isPlaying {
            for _ in 0..<max(1, stepsPerFrame) {
                sim.step(cb)
            }
            stepsSinceLastStats += max(1, stepsPerFrame)
        }
        frameIndex &+= 1

        if frameIndex % 30 == 0 {
            sim.encodeStats(cb)
            statsDue = true
        }

        encodeScene(into: rpd, commandBuffer: cb, aspect: aspect)

        let now = CFAbsoluteTimeGetCurrent()
        cb.addCompletedHandler { [weak self] _ in
            guard let self else { return }
            // GPU idle here: safe to read the device cell counter (division
            // may have grown the structure this frame).
            self.sim.syncCellCountFromDevice()
            if self.statsDue {
                self.statsDue = false
                let mean = self.sim.readMeanLinkLength()
                let dt = max(now - self.lastStatsTime, 1e-6)
                let sps = Double(self.stepsSinceLastStats) / dt
                self.lastStatsTime = now
                self.stepsSinceLastStats = 0
                self.onStats?(self.sim.cellCount, mean, sps)
            }
        }
    }

    private func aspect(of size: CGSize) -> Float {
        Float(size.width / max(size.height, 1))
    }

    private func updateCameraBuffer(aspect: Float) {
        let eye = camera.eye
        let fwd = simd_normalize(camera.target - eye)
        let worldUp = SIMD3<Float>(0, 1, 0)
        let right = simd_normalize(simd_cross(fwd, worldUp))
        let up = simd_cross(right, fwd)

        // M5 unified light: the shading light IS the growth-mode light
        // (SimParams.lightDirX/Y/Z, normalized here). The default value equals
        // the pre-M5 hardcoded direction, so the M3 raster goldens are
        // byte-identical.
        let light = simd_normalize(SIMD3<Float>(sim.params.lightDirX,
                                                sim.params.lightDirY,
                                                sim.params.lightDirZ))

        var u = CameraUniforms(
            view: camera.viewMatrix(),
            proj: camera.projectionMatrix(aspect: aspect),
            right: SIMD4(right, 0),
            up: SIMD4(up, 0),
            negFwd: SIMD4(-fwd, 0),
            lightDir: SIMD4(light, 0),
            xrayPad: SIMD4(xrayGain, 0, 0, 0),
            // V2.1b: .x = foam opacity — 0 keeps the historical solid path
            // (the fragment branches on it); the retired edge layer is gone.
            foamPad: SIMD4(renderStyle == .foam ? max(min(foamOpacity, 1), 0.02) : 0, 0, 0, 0),
            eyePad: SIMD4(eye, 1)
        )
        withUnsafeBytes(of: &u) { src in
            cameraBuffer.contents().copyMemory(from: src.baseAddress!,
                                               byteCount: MemoryLayout<CameraUniforms>.size)
        }
    }

    // MARK: - V2.1b foam draw order

    /// Re-sort the foam draw order back-to-front when needed. The composite
    /// blend needs FAR-TO-NEAR instance order; cell indices are arbitrary
    /// with respect to depth, so the order must be recomputed as the view and
    /// the cells move. Throttled: every frame up to 20k cells, every 30th
    /// frame above (a 650k-cell Swift sort costs ~50–100 ms; the depth order
    /// of a slowly evolving foam changes on ~dozens-of-frames scales).
    /// Called from encodeScene before the foam draw — the position buffer is
    /// read on the encoding thread while the GPU has NOT started this frame's
    /// command buffer (steps + draw encode into the same, still-uncommitted
    /// CB), so it holds the last completed state — the same contract as
    /// updateCameraBuffer's params read.
    private func refreshFoamOrder() {
        let n = sim.cellCount
        guard n > 0, n <= sim.maxCells else { return }
        let eye = camera.eye
        let fwd = simd_normalize(camera.target - eye)
        let interval: UInt64 = n <= 20_000 ? 1 : 30
        let cameraMoved = simd_distance_squared(eye, foamOrderEye) > 1e-12
            || simd_distance_squared(fwd, foamOrderFwd) > 1e-12
        guard n != foamOrderCount || cameraMoved
                || frameIndex - foamOrderFrame >= interval else { return }

        let pos = sim.positionsBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
        var order = [UInt32](0..<UInt32(n))
        // Depth key = distance along the view axis (larger = farther).
        var keys = [Float](repeating: 0, count: n)
        for i in 0..<n { keys[i] = simd_dot(pos[i] - eye, fwd) }
        // Farthest FIRST (the over-blend composites back-to-front).
        order.sort { keys[Int($0)] > keys[Int($1)] }
        order.withUnsafeBytes { src in
            foamOrderBuffer.contents().copyMemory(from: src.baseAddress!,
                                                  byteCount: n * MemoryLayout<UInt32>.stride)
        }
        foamOrderCount = n
        foamOrderFrame = frameIndex
        foamOrderEye = eye
        foamOrderFwd = fwd
    }

    /// Shared scene encoding for the MTKView path and the offscreen path.
    /// The style selects the pipeline: solid = depth-tested opaque impostors
    /// (the historical path, byte-identical); xray = additive density sprites
    /// (depth test/write off); foam (V2.1b) = premultiplied chamber faces,
    /// back-to-front sorted, no depth test. The CALLER owns the clear colour
    /// (the MTKView path sets black for xray; the offscreen path's render-pass
    /// descriptor must do the same — see renderToBytes).
    func encodeScene(into rpd: MTLRenderPassDescriptor, commandBuffer cb: MTLCommandBuffer, aspect: Float) {
        guard sim.cellCount > 0, let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else { return }
        updateCameraBuffer(aspect: aspect)
        if renderStyle == .foam {
            refreshFoamOrder()
        }
        switch renderStyle {
        case .solid:
            enc.setRenderPipelineState(pCellRender)
            enc.setDepthStencilState(depthState)
        case .xray:
            enc.setRenderPipelineState(pCellRenderXray)
            enc.setDepthStencilState(depthStateXray)
        case .foam:
            enc.setRenderPipelineState(pCellRenderFoam)
            enc.setDepthStencilState(depthStateXray)   // always / no write — the sort is the order
        }
        enc.setVertexBuffer(sim.positionsBuffer, offset: 0, index: 0)
        enc.setVertexBuffer(sim.radiiBuffer, offset: 0, index: 1)
        enc.setVertexBuffer(cameraBuffer, offset: 0, index: 2)
        enc.setVertexBuffer(sim.tintBuffer, offset: 0, index: 3)   // M5 per-cell colour
        if renderStyle == .foam {
            enc.setVertexBuffer(foamOrderBuffer, offset: 0, index: 4)   // V2.1b draw order
        }
        enc.setFragmentBuffer(cameraBuffer, offset: 0, index: 0)
        // V2.1b foam fragment: positions/links/linkCounts (the chamber CSG),
        // litFraction (illuminate), params (maxValence stride), tint (the
        // shared wall's two-cell average colour), radii (the neighbours'
        // radii in the CSG).
        enc.setFragmentBuffer(sim.positionsBuffer, offset: 0, index: 1)
        enc.setFragmentBuffer(sim.linksBuffer, offset: 0, index: 2)
        enc.setFragmentBuffer(sim.linkCountsBuffer, offset: 0, index: 3)
        enc.setFragmentBuffer(sim.litFractionBuffer, offset: 0, index: 4)
        enc.setFragmentBuffer(sim.paramsBuffer, offset: 0, index: 5)
        enc.setFragmentBuffer(sim.tintBuffer, offset: 0, index: 6)
        enc.setFragmentBuffer(sim.radiiBuffer, offset: 0, index: 7)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: sim.cellCount)
        enc.endEncoding()
    }
}

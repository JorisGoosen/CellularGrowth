//  Renderer.swift
//  Instanced sphere-impostor renderer. Each cell draws one camera-facing quad;
//  the fragment shader shades it as a sphere with per-pixel normals and depth.
//  Usable both through an MTKView (interactive) and into an offscreen texture
//  (headless validation screenshots).

import Foundation
import Metal
import MetalKit
import simd

/// Must mirror MSL `struct Camera` (192 bytes).
struct CameraUniforms {
    var view: simd_float4x4
    var proj: simd_float4x4
    var right: SIMD4<Float>
    var up: SIMD4<Float>
    var negFwd: SIMD4<Float>
    var lightDir: SIMD4<Float>
}

final class Renderer: NSObject, MTKViewDelegate {

    let device: MTLDevice
    let queue: MTLCommandQueue
    let sim: CellSim

    // Plain vars, written from the main thread, read on the display-link thread.
    var isPlaying = true
    var stepsPerFrame = 4

    var camera = OrbitCamera()

    private var pCellRender: MTLRenderPipelineState!
    private var depthState: MTLDepthStencilState!
    private let cameraBuffer: MTLBuffer

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

            let ds = MTLDepthStencilDescriptor()
            ds.depthCompareFunction = .less
            ds.isDepthWriteEnabled = true
            depthState = device.makeDepthStencilState(descriptor: ds)
        } catch {
            fatalError("Renderer: failed to build pipeline: \(error)")
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cb = queue.makeCommandBuffer() else { return }

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
            lightDir: SIMD4(light, 0)
        )
        withUnsafeBytes(of: &u) { src in
            cameraBuffer.contents().copyMemory(from: src.baseAddress!,
                                               byteCount: MemoryLayout<CameraUniforms>.size)
        }
    }

    /// Shared scene encoding for the MTKView path and the offscreen path.
    func encodeScene(into rpd: MTLRenderPassDescriptor, commandBuffer cb: MTLCommandBuffer, aspect: Float) {
        guard sim.cellCount > 0, let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else { return }
        updateCameraBuffer(aspect: aspect)
        enc.setRenderPipelineState(pCellRender)
        enc.setDepthStencilState(depthState)
        enc.setVertexBuffer(sim.positionsBuffer, offset: 0, index: 0)
        enc.setVertexBuffer(sim.radiiBuffer, offset: 0, index: 1)
        enc.setVertexBuffer(cameraBuffer, offset: 0, index: 2)
        enc.setVertexBuffer(sim.tintBuffer, offset: 0, index: 3)   // M5 per-cell colour
        enc.setFragmentBuffer(cameraBuffer, offset: 0, index: 0)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: sim.cellCount)
        enc.endEncoding()
    }
}

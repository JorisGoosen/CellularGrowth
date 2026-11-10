//  CellSim.swift
//  GPU orchestration of the cellular simulation (paper §4/§5).
//
//  M5 scope: M3/M4 + growth modes (paper §4 food methods): uniform random
//  (0, default and unchanged), light-created nutrient (1), Gray-Scott RDE (2).
//  The per-step kernel pipeline is now:
//    normals -> [grid rebuild: count, scanLocal, scanBlocks, scanOffset,
//    scatter] -> [repulsionAccumulate when repulsionStrength > 0]
//    -> [M5 chemicals: boundMax + illuminate + clearBound + nutrientDiffuse
//    (light mode) | rdeStep (rde mode)] -> integrateForces (adds
//    collisionOffset) -> food -> division (decide, execute, finalize)
//    -> cellTint (AFTER division so this step's daughters are tinted — M6)
//    -> radii
//  The chemical passes read the same frozen pre-step state as repulsion and
//  write ONLY the chemical buffers, so the force/repulsion semantics are
//  untouched (--lockstep forces growthMode 0 + division off, as before).
//  In light mode the grid is rebuilt even with repulsion disabled (the light
//  march traverses it).
//  The grid is a hashed sparse uniform grid (bin size = roi, bit-packed
//  coordinate hash) rebuilt from scratch every step; repulsionStrength == 0
//  skips the whole grid/repulsion block and reproduces the M2 behaviour
//  exactly (integrateForces guards the collisionOffset read).
//  Everything lives in preallocated Metal buffers (storageModeShared — on
//  Apple Silicon this is device-local unified memory). Positions are
//  double-buffered: all kernels of a step read the "current" buffer and write
//  the "next" one; division additionally writes daughter cells into the next
//  buffer at slots >= the step-start count (disjoint from integrateForces'
//  writes), so the buffers swap cleanly after the step.
//
//  Growth / concurrency design (details in Shaders.swift kernel comments):
//  - The live cell count is cellCounter[0] on the device. Kernels bound on it,
//    never on params.cellCount (host mirror only). The host reads the counter
//    back only while the GPU is idle (syncCellCountFromDevice).
//  - Division = decide (claim, on frozen data) -> execute (mechanics) ->
//    finalizeFood (parent food reset). Claims form a deterministic independent
//    set at ring-distance 2 (min cell index wins), so split write sets are
//    provably disjoint; no CAS loop is needed. decide and execute MUST be
//    dispatched with the same width, so a stale claim beyond the width is
//    never consumed.
//  - Step dispatch width = min(maxCells, 2 * host cellCount): an over-estimate
//    under normal growth. If growth outgrows it within a batch, affected cells
//    simply stall until the next host sync — no state can be corrupted.
//  - Grid table size T grows with the live cell count, but ONLY from paths
//    that guarantee no command buffer is in flight (headless batch loops after
//    waitUntilCompleted): a mid-flight params update could otherwise be seen
//    by the count and scatter kernels of an already-enqueued step with
//    different hash bits, corrupting that step's bin bookkeeping. The GUI
//    completion handler therefore keeps the reset-time T (correct, just less
//    efficient once a GUI run grows far past its initial count).

import Foundation
import Metal
import simd

/// Must mirror MSL `struct SimParams` exactly (31 x 4 bytes = 124).
/// Fields are only ever APPENDED at the end, on both sides.
struct SimParams {
    var linkRestLength: Float = 1.0
    var springFactor: Float = 0.33
    var planarFactor: Float = 0.33
    var bulgeFactor: Float = 0.33
    var radiusScale: Float = 0.5
    var cellCount: UInt32 = 0        // host mirror; kernels use the device counter
    var maxValence: UInt32 = UInt32(CellSim.maxValence)
    // --- M2 (appended) ---
    var foodIncrement: Float = 0.01  // mean per-step food = foodIncrement * U[0,1)
    var splitThreshold: Float = 1.0
    var divisionEnabled: UInt32 = 1
    var rngSeed: UInt32 = 12345      // consumed host-side at reset
    var maxCells: UInt32 = 2_000_000 // hard capacity enforced on the device
    // --- M3 (appended) ---
    /// Radius of influence for repulsion. Bin size = roi (h >= roi rule).
    /// Chosen 1.6 x linkRestLength: below the icosphere's second-neighbour
    /// distance (~1.7 x rest) so the pristine ball feels no repulsion at rest
    /// (settle gate unchanged), above ~1.4 x rest so folded contacts are caught.
    var roi: Float = 1.6
    /// Repulsion strength (paper §4 collisionOffset scale). 0 disables.
    /// Tuned over 0.05–0.3 (see benchlog M3 / M3 report): 0.1 gives coherent
    /// folded growth at food 0.01; higher values freeze the form.
    var repulsionStrength: Float = 0.1
    /// Packed hash bits per axis; T = 2^(bx+by+bz). Host-derived from the live
    /// cell count, clamped to [2^8, 2^22] (buffers sized for the 2^22 max).
    var gridBitsX: UInt32 = 4
    var gridBitsY: UInt32 = 4
    var gridBitsZ: UInt32 = 3
    // --- M5 (appended) ---
    /// Growth mode (paper §4 food methods): 0 uniform random increment,
    /// 1 light-created nutrient (method 3), 2 Gray-Scott RDE (method 2).
    var growthMode: UInt32 = 0
    /// Unit vector TOWARD the light (three floats — no float3-in-struct
    /// alignment traps). THE shared light: consumed by the growth illumination
    /// (illuminate kernel) AND the renderers (impostor shading, directional
    /// solid style). Stored UNNORMALIZED (normalized at every use site) — the
    /// default is the pre-M5 impostor light (0.55, 0.75, 0.4) so the M3 raster
    /// goldens are byte-identical under the unified light.
    var lightDirX: Float = 0.55
    var lightDirY: Float = 0.75
    var lightDirZ: Float = 0.4
    /// Jacobi nutrient diffusion along links per step, clamped to [0, 1]
    /// (convex combination). Paper gate values: 0.02 plant-like, 0.3 organ-like.
    var lightDiffusionRate: Float = 0.02
    /// Nutrient produced per step in a lit cell.
    var lightNutrient: Float = 0.05
    /// First-order nutrient decay per step: the steady lit/dark gradient is
    /// controlled by lightDecay vs lightDiffusionRate (see illuminate comment
    /// in Shaders.swift for why pure accumulation / division-consumption fail).
    var lightDecay: Float = 0.02
    /// Food coupling floor in light mode: increment ∝ (lightBase + nutrient).
    /// > 0 so fully shaded regions keep growing slowly (no frozen flaps).
    var lightBase: Float = 0.15
    /// Illumination-march interval in steps. The march is the expensive light
    /// pass (empty-interior traversal dominates at scale — see benchlog M5);
    /// the shadow pattern evolves on ~100-step scales (structure moves
    /// ~0.5%/step, nutrient time constant ~1/lightDecay = 50 steps), so
    /// marching every 8th step with 8x pulsed production leaves the nutrient
    /// steady state unchanged. 1 = every step (original semantics).
    var lightInterval: UInt32 = 8
    /// Gray-Scott feed rate (default 0.037: "coral/mitosis" spot regime).
    var rdeF: Float = 0.037
    /// Gray-Scott kill rate (default 0.06).
    var rdeK: Float = 0.06
    /// u diffusion, graph-normalized Laplacian (avg(ring) − x), [0, 1] stable.
    /// 0.35 halves the classic Gray-Scott mixing weight (Karl Sims' 3x3
    /// convolution: Σw = 1, center −1): the pattern wavelength shrinks so
    /// spots localize even on the 642-cell validation ball — at 1.0/0.5 the
    /// u depletion homogenizes over that small domain and the field goes
    /// extinct after nucleation (measured, rde-probe).
    var rdeDu: Float = 0.35
    /// v diffusion (u > v is what pattern formation needs).
    var rdeDv: Float = 0.15
    /// Food coupling in rde mode: increment ∝ (rdeBase + rdeGain · v).
    /// v sits ~0.4 in spots, ~0 elsewhere → ~5x differential at the defaults.
    var rdeBase: Float = 0.25
    var rdeGain: Float = 2.5
    // --- M6 (appended) ---
    /// Coarse occupancy hash bits per axis (the light-march skip map: a bit
    /// per 8³-fine-bin coordinate block, hashed like the fine grid). Derived
    /// host-side in setGridBits as max(bits − 3, 2) per axis; collisions are
    /// harmless (a spuriously SET bit only disables the skip — conservative).
    var gridCoarseBitsX: UInt32 = 2
    var gridCoarseBitsY: UInt32 = 2
    var gridCoarseBitsZ: UInt32 = 2
    /// M6 light-march coarse skip enable (1 = on). Kill switch for A/B
    /// validation (CG_NO_COARSE_SKIP=1): the skip is provably verdict-identical
    /// (only ever taken over regions proven free of cells), and this flag is
    /// how that proof is checked in practice.
    var lightCoarseSkip: UInt32 = 1
}

enum ShaderError: Error {
    case missingFunction(String)
}

final class CellSim {

    static let maxValence = 16

    /// Spatial-hash table size bounds: T = 2^(bx+by+bz), clamped so buffers
    /// are preallocated once for the maximum (gridMaxBits = 22 -> T_max =
    /// 4,194,304 bins; a fixed function of maxCells, never reallocated).
    static let gridMinBits: UInt32 = 8
    static let gridMaxBits: UInt32 = 22

    let device: MTLDevice
    let queue: MTLCommandQueue
    let maxCells: Int

    // Buffers (SoA; strided layout documented per kernel in Shaders.swift).
    private let positionsA: MTLBuffer
    private let positionsB: MTLBuffer
    private let normalsBuffer: MTLBuffer
    private let linksBuffer: MTLBuffer      // UInt32, [cell][maxValence]
    private let linkCountsBuffer: MTLBuffer // UInt32 per cell
    private let foodBuffer: MTLBuffer       // Float per cell
    private let rngBuffer: MTLBuffer        // UInt64 PCG32-style state per cell
    private let splitFlagsBuffer: MTLBuffer // UInt32 per cell (division claims)
    let counterBuffer: MTLBuffer            // UInt32: authoritative live cell count
    let radiiBuffer: MTLBuffer              // Float per cell (read by the renderer)
    let paramsBuffer: MTLBuffer             // SimParams mirror (read by the M4 ray renderer for roi + hash bits)
    private let statsPartialsBuffer: MTLBuffer // SIMD2<Float> per threadgroup
    // --- M3 spatial hash + repulsion ---
    // binCounts/binStarts/particleIndices are internal (not private): the M4
    // ray renderer (Render/RayRenderer.swift) traverses the SAME grid for its
    // DDA ray walk — one shared spatial index, built once for the final state.
    let binCountsBuffer: MTLBuffer               // UInt32 per bin (T_max): counts, then fill cursor (self-cleans to 0)
    let binStartsBuffer: MTLBuffer               // UInt32 per bin + 1: global exclusive prefix
    private let blockSumsBuffer: MTLBuffer       // UInt32 per scan threadgroup
    let particleIndicesBuffer: MTLBuffer         // UInt32 per cell: counting-sort output
    private let collisionOffsetBuffer: MTLBuffer // SIMD3<Float> per cell: repulsion field
    // --- M5 growth-mode chemicals ---
    // Nutrient (light mode) and u/v (rde mode) are double-buffered like
    // positions: each step's chemical pass reads the current buffer and writes
    // the next one; the host swaps after the step. Division reads/writes the
    // CURRENT chemical buffers (they are finalized before the division block
    // within a step). boundBuffer holds max |P|² (uint bit-cast) for the light
    // march's ray bound; cleared by clearBound within the same step.
    private let nutrientA: MTLBuffer        // Float per cell
    private let nutrientB: MTLBuffer
    private let chemUA: MTLBuffer
    private let chemUB: MTLBuffer
    private let chemVA: MTLBuffer
    private let chemVB: MTLBuffer
    private let boundBuffer: MTLBuffer      // UInt32: atomic max |P|² accumulator
    /// M5 per-cell render colour (cellTint kernel; neutral albedo unless rde).
    let tintBuffer: MTLBuffer
    /// M5 per-cell illumination verdict (illuminate kernel; 1 = lit). The ray
    /// tracer's directional solid style shades with it — the render shows
    /// exactly the cells the growth illuminated.
    let litFlagsBuffer: MTLBuffer
    // --- M6 light-march occupancy ---
    /// Coarse occupancy BITFIELD for the light march's empty-interior skip:
    /// one bit per hashed 8³-fine-bin coordinate block (FIXED 7-bit packed
    /// hash per axis → T_c = 2^21 slots → 2^18 words = 256 KB, L2-resident).
    /// Set by gridCount (atomic OR), cleared by a blit fill at the START of
    /// every grid rebuild (the map must persist through the post-scatter
    /// march — a "self-cleaning" count mirrored in gridScatter is WRONG: the
    /// scatter would zero it before illuminate ever reads it; found by the
    /// M6 A/B). A CLEAR bit proves the 3×3×3 coarse neighbourhood holds no
    /// cell (safe to skip); hash collisions only over-report (conservative).
    /// Sized for block coordinates ±63 (structures up to ~800 world units
    /// across at roi 1.6); beyond that the skip degrades to a no-op, never
    /// wrong.
    let occupancyBuffer: MTLBuffer
    private var usingNutA = true
    private var usingChemA = true

    var params = SimParams()
    private(set) var cellCount = 0          // host mirror of counterBuffer[0]
    private var usingA = true
    /// Step counter for the light-march scheduler (lightInterval); reset with
    /// the state.
    private var stepIndex: UInt64 = 0

    /// Current positions (post-step).
    var positionsBuffer: MTLBuffer { usingA ? positionsA : positionsB }
    private var nextPositionsBuffer: MTLBuffer { usingA ? positionsB : positionsA }
    /// Current nutrient level (light mode; post-step).
    var nutrientBuffer: MTLBuffer { usingNutA ? nutrientA : nutrientB }
    private var nextNutrientBuffer: MTLBuffer { usingNutA ? nutrientB : nutrientA }
    /// Current RDE chemicals (rde mode; post-step).
    var chemUBuffer: MTLBuffer { usingChemA ? chemUA : chemUB }
    var chemVBuffer: MTLBuffer { usingChemA ? chemVA : chemVB }
    private var nextChemUBuffer: MTLBuffer { usingChemA ? chemUB : chemUA }
    private var nextChemVBuffer: MTLBuffer { usingChemA ? chemVB : chemVA }

    private var pComputeNormals: MTLComputePipelineState!
    private var pIntegrateForces: MTLComputePipelineState!
    private var pFoodStep: MTLComputePipelineState!
    private var pDivisionDecide: MTLComputePipelineState!
    private var pDivisionExecute: MTLComputePipelineState!
    private var pFinalizeFood: MTLComputePipelineState!
    private var pComputeRadii: MTLComputePipelineState!
    private var pStats: MTLComputePipelineState!
    // --- M3 spatial hash + repulsion ---
    private var pGridCount: MTLComputePipelineState!
    private var pGridScanLocal: MTLComputePipelineState!
    private var pGridScanBlocks: MTLComputePipelineState!
    private var pGridScanOffset: MTLComputePipelineState!
    private var pGridScatter: MTLComputePipelineState!
    private var pRepulsionAccumulate: MTLComputePipelineState!
    // --- M5 growth modes ---
    private var pBoundMax: MTLComputePipelineState!
    private var pClearBound: MTLComputePipelineState!
    private var pIlluminate: MTLComputePipelineState!
    private var pNutrientDiffuse: MTLComputePipelineState!
    private var pSeedChemicals: MTLComputePipelineState!
    private var pRdeStep: MTLComputePipelineState!
    private var pCellTint: MTLComputePipelineState!
    // TEMPORARY DEBUG (CELLDBG; remove after M3 gates are green):
    private var pParamEcho: MTLComputePipelineState!
    private var pQueryFingerprint: MTLComputePipelineState!
    private var pZeroFloat3: MTLComputePipelineState!

    init(device: MTLDevice, queue: MTLCommandQueue, maxCells: Int = 2_000_000) {
        self.device = device
        self.queue = queue
        self.maxCells = maxCells
        params.maxCells = UInt32(maxCells)

        do {
            let lib = try device.makeLibrary(source: Shaders.sim, options: nil)
            func pipeline(_ name: String) throws -> MTLComputePipelineState {
                guard let fn = lib.makeFunction(name: name) else {
                    throw ShaderError.missingFunction(name)
                }
                return try device.makeComputePipelineState(function: fn)
            }
            pComputeNormals = try pipeline("computeNormals")
            pIntegrateForces = try pipeline("integrateForces")
            pFoodStep = try pipeline("foodStep")
            pDivisionDecide = try pipeline("divisionDecide")
            pDivisionExecute = try pipeline("divisionExecute")
            pFinalizeFood = try pipeline("finalizeFood")
            pComputeRadii = try pipeline("computeRadii")
            pStats = try pipeline("statsKernel")
            pGridCount = try pipeline("gridCount")
            pGridScanLocal = try pipeline("gridScanLocal")
            pGridScanBlocks = try pipeline("gridScanBlocks")
            pGridScanOffset = try pipeline("gridScanOffset")
            pGridScatter = try pipeline("gridScatter")
            pRepulsionAccumulate = try pipeline("repulsionAccumulate")
            // --- M5 growth modes ---
            pBoundMax = try pipeline("boundMax")
            pClearBound = try pipeline("clearBound")
            pIlluminate = try pipeline("illuminate")
            pNutrientDiffuse = try pipeline("nutrientDiffuse")
            pSeedChemicals = try pipeline("seedChemicals")
            pRdeStep = try pipeline("rdeStep")
            pCellTint = try pipeline("cellTint")
            // TEMPORARY DEBUG (CELLDBG; remove after M3 gates are green):
            pParamEcho = try pipeline("paramEcho")
            pQueryFingerprint = try pipeline("queryFingerprint")
            pZeroFloat3 = try pipeline("zeroFloat3")
        } catch {
            fatalError("CellSim: failed to build Metal compute pipelines: \(error)")
        }

        // Layout drift guard: the MSL SimParams mirror is appended-only; a
        // mismatch here corrupts every kernel silently.
        precondition(MemoryLayout<SimParams>.size == 144,
                     "SimParams layout drift: \(MemoryLayout<SimParams>.size) != 144 (see Shaders.swift)")

        let f3 = MemoryLayout<SIMD3<Float>>.stride
        positionsA = device.makeBuffer(length: maxCells * f3, options: .storageModeShared)!
        positionsA.label = "positionsA"
        positionsB = device.makeBuffer(length: maxCells * f3, options: .storageModeShared)!
        positionsB.label = "positionsB"
        normalsBuffer = device.makeBuffer(length: maxCells * f3, options: .storageModeShared)!
        normalsBuffer.label = "normals"
        linksBuffer = device.makeBuffer(length: maxCells * Self.maxValence * MemoryLayout<UInt32>.stride,
                                        options: .storageModeShared)!
        linksBuffer.label = "links"
        linkCountsBuffer = device.makeBuffer(length: maxCells * MemoryLayout<UInt32>.stride,
                                             options: .storageModeShared)!
        linkCountsBuffer.label = "linkCounts"
        foodBuffer = device.makeBuffer(length: maxCells * MemoryLayout<Float>.stride,
                                       options: .storageModeShared)!
        foodBuffer.label = "food"
        rngBuffer = device.makeBuffer(length: maxCells * MemoryLayout<UInt64>.stride,
                                      options: .storageModeShared)!
        rngBuffer.label = "rngState"
        splitFlagsBuffer = device.makeBuffer(length: maxCells * MemoryLayout<UInt32>.stride,
                                             options: .storageModeShared)!
        splitFlagsBuffer.label = "splitFlags"
        counterBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride,
                                          options: .storageModeShared)!
        counterBuffer.label = "cellCounter"
        radiiBuffer = device.makeBuffer(length: maxCells * MemoryLayout<Float>.stride,
                                        options: .storageModeShared)!
        radiiBuffer.label = "radii"
        paramsBuffer = device.makeBuffer(length: MemoryLayout<SimParams>.size,
                                         options: .storageModeShared)!
        paramsBuffer.label = "simParams"
        statsPartialsBuffer = device.makeBuffer(
            length: (maxCells / 64 + 1) * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared)!
        statsPartialsBuffer.label = "statsPartials"

        // M3 spatial hash: sized for T_max = 2^gridMaxBits once (a fixed
        // function of maxCells) — the live table size T is just a param, so
        // growth never reallocates. binCounts starts zeroed; the scatter pass
        // keeps it zeroed (fill cursor decrements back to 0). collisionOffset
        // is zeroed once: when repulsion is disabled the kernel-side guard in
        // integrateForces ignores whatever it holds.
        let tMax = 1 << Int(Self.gridMaxBits)
        binCountsBuffer = device.makeBuffer(length: tMax * MemoryLayout<UInt32>.stride,
                                            options: .storageModeShared)!
        binCountsBuffer.label = "binCounts"
        binStartsBuffer = device.makeBuffer(length: (tMax + 1) * MemoryLayout<UInt32>.stride,
                                            options: .storageModeShared)!
        binStartsBuffer.label = "binStarts"
        blockSumsBuffer = device.makeBuffer(length: (tMax / 64 + 2) * MemoryLayout<UInt32>.stride,
                                            options: .storageModeShared)!
        blockSumsBuffer.label = "gridBlockSums"
        particleIndicesBuffer = device.makeBuffer(length: maxCells * MemoryLayout<UInt32>.stride,
                                                  options: .storageModeShared)!
        particleIndicesBuffer.label = "particleIndices"
        collisionOffsetBuffer = device.makeBuffer(length: maxCells * f3,
                                                  options: .storageModeShared)!
        collisionOffsetBuffer.label = "collisionOffset"
        // M5 growth-mode chemicals (preallocated like everything else; unused
        // in uniform mode). Double buffers for the Jacobi passes.
        nutrientA = device.makeBuffer(length: maxCells * MemoryLayout<Float>.stride,
                                      options: .storageModeShared)!
        nutrientA.label = "nutrientA"
        nutrientB = device.makeBuffer(length: maxCells * MemoryLayout<Float>.stride,
                                      options: .storageModeShared)!
        nutrientB.label = "nutrientB"
        chemUA = device.makeBuffer(length: maxCells * MemoryLayout<Float>.stride,
                                   options: .storageModeShared)!
        chemUA.label = "chemUA"
        chemUB = device.makeBuffer(length: maxCells * MemoryLayout<Float>.stride,
                                   options: .storageModeShared)!
        chemUB.label = "chemUB"
        chemVA = device.makeBuffer(length: maxCells * MemoryLayout<Float>.stride,
                                   options: .storageModeShared)!
        chemVA.label = "chemVA"
        chemVB = device.makeBuffer(length: maxCells * MemoryLayout<Float>.stride,
                                   options: .storageModeShared)!
        chemVB.label = "chemVB"
        boundBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride,
                                        options: .storageModeShared)!
        boundBuffer.label = "lightBound2"
        tintBuffer = device.makeBuffer(length: maxCells * f3,
                                       options: .storageModeShared)!
        tintBuffer.label = "cellTint"
        litFlagsBuffer = device.makeBuffer(length: maxCells * MemoryLayout<UInt32>.stride,
                                           options: .storageModeShared)!
        litFlagsBuffer.label = "litFlags"
        // M6 coarse occupancy bitfield: 2^21 slots → 2^18 words = 256 KB.
        // Blit-cleared at every grid rebuild (256 KB ≈ microseconds).
        occupancyBuffer = device.makeBuffer(
            length: (1 << 18) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        occupancyBuffer.label = "lightOccupancy"
        memset(binCountsBuffer.contents(), 0, binCountsBuffer.length)
        memset(collisionOffsetBuffer.contents(), 0, collisionOffsetBuffer.length)
        memset(nutrientA.contents(), 0, nutrientA.length)
        memset(nutrientB.contents(), 0, nutrientB.length)
        memset(boundBuffer.contents(), 0, boundBuffer.length)
        memset(litFlagsBuffer.contents(), 0, litFlagsBuffer.length)
        memset(occupancyBuffer.contents(), 0, occupancyBuffer.length)
    }

    // MARK: - Setup

    /// Load a fresh cell set (positions + ring-ordered link lists), seed the
    /// per-cell RNG streams, and prime normals/radii with a synchronous pass.
    func reset(positions: [SIMD3<Float>], links: [[UInt32]]) {
        precondition(positions.count <= maxCells, "cell set exceeds maxCells")
        precondition(positions.count == links.count, "positions/links mismatch")
        cellCount = positions.count
        params.cellCount = UInt32(cellCount)
        // M3: size the hash for the initial count (shrink allowed here — the
        // counts are re-zeroed below, so bins dropped by a smaller T are clean),
        // then push params. Bins beyond the current T stay zero from init and
        // are only touched if T grows later (grow-only while a run is live).
        setGridBits(forCellCount: cellCount)
        memset(binCountsBuffer.contents(), 0, binCountsBuffer.length)
        syncParams()
        usingA = true

        positions.withUnsafeBytes { src in
            positionsA.contents().copyMemory(from: src.baseAddress!,
                                             byteCount: positions.count * MemoryLayout<SIMD3<Float>>.stride)
        }

        // Strided layout: cell i's ring lives at [i * maxValence, i * maxValence + count).
        var flatLinks = [UInt32](repeating: 0, count: cellCount * Self.maxValence)
        var counts = [UInt32](repeating: 0, count: cellCount)
        for (i, ring) in links.enumerated() {
            precondition(ring.count <= Self.maxValence, "valence overflow at cell \(i)")
            counts[i] = UInt32(ring.count)
            for (k, l) in ring.enumerated() {
                flatLinks[i * Self.maxValence + k] = l
            }
        }
        flatLinks.withUnsafeBytes { src in
            linksBuffer.contents().copyMemory(from: src.baseAddress!,
                                              byteCount: flatLinks.count * MemoryLayout<UInt32>.stride)
        }
        counts.withUnsafeBytes { src in
            linkCountsBuffer.contents().copyMemory(from: src.baseAddress!,
                                                   byteCount: counts.count * MemoryLayout<UInt32>.stride)
        }

        // Per-cell RNG streams + initial food, deterministic from rngSeed.
        // Initial food is uniform in [0, splitThreshold): without it every cell
        // would cross the threshold in the same step and the first division
        // wave would be artificially synchronised.
        let seed = UInt64(params.rngSeed)
        var rng = [UInt64](repeating: 0, count: cellCount)
        var food = [Float](repeating: 0, count: cellCount)
        for i in 0..<cellCount {
            let s = Self.splitmix64(seed &+ UInt64(i) &* 0x9E3779B97F4A7C15)
            rng[i] = s
            let u = Self.splitmix64(s &+ 0xDA3E39CB94B95B9D)
            food[i] = Float(u >> 11) * (1.0 / 9007199254740992.0) * params.splitThreshold
        }
        rng.withUnsafeBytes { src in
            rngBuffer.contents().copyMemory(from: src.baseAddress!,
                                            byteCount: rng.count * MemoryLayout<UInt64>.stride)
        }
        food.withUnsafeBytes { src in
            foodBuffer.contents().copyMemory(from: src.baseAddress!,
                                             byteCount: food.count * MemoryLayout<Float>.stride)
        }

        // Clear division state.
        let zeros = [UInt32](repeating: 0, count: cellCount)
        zeros.withUnsafeBytes { src in
            splitFlagsBuffer.contents().copyMemory(from: src.baseAddress!,
                                                   byteCount: zeros.count * MemoryLayout<UInt32>.stride)
        }
        counterBuffer.contents().assumingMemoryBound(to: UInt32.self)[0] = UInt32(cellCount)

        // M5 chemical state: nutrient starts at 0 (both buffers — a previous
        // run may have left values either side of the swap); u/v are seeded by
        // the seedChemicals kernel in the reset CB below.
        usingNutA = true
        usingChemA = true
        stepIndex = 0
        memset(nutrientA.contents(), 0, nutrientA.length)
        memset(nutrientB.contents(), 0, nutrientB.length)
        memset(boundBuffer.contents(), 0, boundBuffer.length)
        memset(litFlagsBuffer.contents(), 0, litFlagsBuffer.length)

        guard let cb = queue.makeCommandBuffer() else { fatalError("no command buffer") }
        encodeNormals(cb, width: cellCount)
        encodeRadii(into: cb, positions: positionsA, width: cellCount)
        encodeSeedChemicals(cb, width: cellCount)
        encodeCellTint(cb, width: cellCount)
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// SplitMix64, host-side only: derives per-cell RNG streams and the
    /// initial food from the single rngSeed.
    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    // MARK: - Spatial hash capacity (M3)

    /// Derive the packed hash's per-axis bit split for `n` cells: total bits
    /// k = smallest with 2^k >= 2n (T >= 2 x cells keeps chains short),
    /// clamped to [gridMinBits, gridMaxBits]; split as evenly as possible
    /// (a = k/3, c takes the remainder). Every axis keeps >= 2 bits, which is
    /// what makes the packed hash injective on each query's 3x3x3
    /// neighbourhood (per-axis neighbourhood span 2 < per-axis modulus 2^bits).
    private func setGridBits(forCellCount n: Int) {
        var k = Self.gridMinBits
        while (1 << k) < 2 * n && k < Self.gridMaxBits { k += 1 }
        let a = Int(k) / 3
        params.gridBitsX = UInt32(a)
        params.gridBitsY = UInt32(a)
        params.gridBitsZ = UInt32(Int(k) - 2 * a)
        // M6 coarse occupancy bits: FIXED 7 per axis (NOT derived from the
        // fine table — the first attempt derived max(bits−3, 2), which at
        // high cell counts left a 2^12-slot map for a shell spanning ±18
        // blocks: wrap collisions saturated every slot and the skip never
        // fired). 7 bits per axis covers block coordinates ±63 with no wrap
        // at our scales (structure radius ≤ ~63·8·roi ≈ 800 world units);
        // larger scenes merely lose the skip (conservative). The bitfield is
        // blit-cleared at the start of every grid rebuild (the map must be
        // LIVE during the post-scatter illumination march).
        params.gridCoarseBitsX = 7
        params.gridCoarseBitsY = 7
        params.gridCoarseBitsZ = 7
    }

    /// Current hash table size T = 2^(bx+by+bz).
    var gridBinCount: Int {
        1 << Int(params.gridBitsX + params.gridBitsY + params.gridBitsZ)
    }

    /// Byte accounting of the M3 grid structures at the current cell count
    /// (for the benchlog's grid memory rows).
    func gridMemoryDescription() -> String {
        let t = gridBinCount
        let bins = t * MemoryLayout<UInt32>.stride
        let starts = (t + 1) * MemoryLayout<UInt32>.stride
        let indices = cellCount * MemoryLayout<UInt32>.stride
        let collision = cellCount * MemoryLayout<SIMD3<Float>>.stride
        let total = bins + starts + indices + collision
        return "grid T=\(t): binCounts \(bins)B + binStarts \(starts)B + particleIndices \(indices)B + collisionOffsets \(collision)B = \(total)B"
    }

    /// Push the Swift-side params struct into the GPU buffer.
    func syncParams() {
        withUnsafeBytes(of: &params) { src in
            paramsBuffer.contents().copyMemory(from: src.baseAddress!,
                                               byteCount: MemoryLayout<SimParams>.size)
        }
    }

    // MARK: - Simulation

    /// Encode one simulation step into `cb`:
    /// normals -> [M3 grid rebuild + repulsionAccumulate, when
    /// repulsionStrength > 0] -> [M5 chemical passes: boundMax + illuminate +
    /// clearBound + nutrientDiffuse (light mode) | rdeStep (rde mode)] ->
    /// integrate -> food -> division (decide, execute, finalize) -> radii.
    /// All kernels bound on the device cell counter; the dispatch width is a
    /// benign over-estimate (see class docs). Buffers swap on the CPU side;
    /// GPU execution order is fixed within the CB (Metal hazard-tracks the
    /// resources across encoders).
    ///
    /// M5 notes:
    /// - The grid is rebuilt whenever growthMode == light EVEN IF repulsion is
    ///   disabled: the light march traverses it. The rebuild is harmless for
    ///   the other modes.
    /// - The chemical passes read the same frozen pre-step state as the
    ///   repulsion pass (positions + radii + the fresh grid), so illumination
    ///   is consistent with the repulsion field. They write only the chemical
    ///   buffers — the force/repulsion path is untouched (lockstep invariant).
    /// - The nutrient/rde buffers swap after their pass, so foodStep and
    ///   divisionExecute (later in this CB) see the post-pass chemicals.
    func step(_ cb: MTLCommandBuffer) {
        guard cellCount > 0 else { return }
        let width = dispatchWidth
        encodeNormals(cb, width: width)
        let needGrid = params.repulsionStrength > 0 || params.growthMode == 1
        if needGrid {
            encodeClearOccupancy(cb)   // M6: the skip map is per-rebuild state
            encodeGridCount(cb, width: width)
            encodeGridScanLocal(cb)
            encodeGridScanBlocks(cb)
            encodeGridScanOffset(cb)
            encodeGridScatter(cb, width: width)
        }
        if params.repulsionStrength > 0 {
            encodeRepulsionAccumulate(cb, width: width)
        }
        if params.growthMode == 1 {
            // The illumination march is the expensive pass (empty-interior
            // traversal — benchlog M5); the shadow pattern evolves on
            // ~100-step scales, so it runs every lightInterval steps with
            // lightInterval× pulsed production (steady state unchanged).
            // Decay + diffusion run every step below.
            if stepIndex % UInt64(max(params.lightInterval, 1)) == 0 {
                encodeBoundMax(cb, width: width)
                encodeIlluminate(cb, width: width)
                encodeClearBound(cb)
            }
            encodeNutrientDiffuse(cb, width: width)
            usingNutA.toggle()
        } else if params.growthMode == 2 {
            encodeRdeStep(cb, width: width)
            usingChemA.toggle()
        }
        // M5 per-cell render colour: AFTER the division block so the daughters
        // of THIS step are tinted too (they are born inside divisionExecute,
        // which writes their chemicals into the current buffers — tint must
        // be recomputed from those, or last-step daughters render with a
        // stale/zero tint = black speckles at silhouettes; found by the M6
        // state-round-trip validation). Uniform/light modes still write the
        // neutral albedo for every cell, so no golden pixels change.
        encodeCellTint(cb, width: width)
        encodeIntegrate(cb, width: width)
        if params.divisionEnabled != 0 {
            encodeFoodStep(cb, width: width)
            encodeDivisionDecide(cb, width: width)
            encodeDivisionExecute(cb, width: width)
            encodeFinalizeFood(cb, width: width)
            // Tint AFTER division: daughters inherit the parent's u/v, and
            // cellTint must see them (see comment above).
            encodeCellTint(cb, width: width)
        }
        encodeRadii(into: cb, positions: nextPositionsBuffer, width: width)
        usingA.toggle()
        stepIndex &+= 1
    }

    /// Over-estimated dispatch width (class docs: under-coverage is benign).
    private var dispatchWidth: Int {
        min(maxCells, max(64, cellCount * 2))
    }

    private func encodeNormals(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pComputeNormals)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(linksBuffer, offset: 0, index: 1)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 2)
        enc.setBuffer(normalsBuffer, offset: 0, index: 3)
        enc.setBuffer(counterBuffer, offset: 0, index: 4)
        enc.setBuffer(paramsBuffer, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeIntegrate(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pIntegrateForces)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(normalsBuffer, offset: 0, index: 1)
        enc.setBuffer(linksBuffer, offset: 0, index: 2)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 3)
        enc.setBuffer(nextPositionsBuffer, offset: 0, index: 4)
        enc.setBuffer(counterBuffer, offset: 0, index: 5)
        enc.setBuffer(paramsBuffer, offset: 0, index: 6)
        enc.setBuffer(collisionOffsetBuffer, offset: 0, index: 7)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: M3 spatial hash + repulsion encoding
    //
    // The five grid passes must ALWAYS run together in one command buffer:
    // the scatter pass is what zeroes binCounts again (fill cursor), so a
    // partial rebuild would leave the counts dirty for the next step.

    /// Clear the coarse occupancy bitfield (blit fill — 256 KB, trivially
    /// fast) in the same command buffer BEFORE the gridCount pass that sets
    /// it again. Called by every grid-rebuild path (step, encodeGridRebuild,
    /// probe): the map must be live during the post-scatter illumination
    /// march, so it is per-rebuild state.
    private func encodeClearOccupancy(_ cb: MTLCommandBuffer) {
        guard let enc = cb.makeBlitCommandEncoder() else { return }
        enc.fill(buffer: occupancyBuffer, range: 0..<occupancyBuffer.length, value: 0)
        enc.endEncoding()
    }

    private func encodeGridCount(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pGridCount)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(binCountsBuffer, offset: 0, index: 1)
        enc.setBuffer(occupancyBuffer, offset: 0, index: 2)   // M6 coarse occupancy bits
        enc.setBuffer(counterBuffer, offset: 0, index: 3)
        enc.setBuffer(paramsBuffer, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Number of scan threadgroups: ceil(T/64). T = 2^k (k >= 8) is always a
    /// multiple of 64, so the bins fill whole threadgroups exactly;
    /// binStarts[T] (grand total, ending the last bin's range) is written by
    /// the scanBlocks kernel, not scanned as a virtual entry.
    private var scanThreadgroups: Int { (gridBinCount + 63) / 64 }

    private func encodeGridScanLocal(_ cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pGridScanLocal)
        enc.setBuffer(binCountsBuffer, offset: 0, index: 0)
        enc.setBuffer(binStartsBuffer, offset: 0, index: 1)
        enc.setBuffer(blockSumsBuffer, offset: 0, index: 2)
        enc.setBuffer(paramsBuffer, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: scanThreadgroups * 64, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeGridScanBlocks(_ cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pGridScanBlocks)
        enc.setBuffer(blockSumsBuffer, offset: 0, index: 0)
        enc.setBuffer(binStartsBuffer, offset: 0, index: 1)
        enc.setBuffer(paramsBuffer, offset: 0, index: 2)
        // EXACTLY ONE threadgroup: gridScanBlocks is a single-threadgroup
        // algorithm (one shared[64] scan, one writer of binStarts[T]). The
        // block-sum COUNT (T/64) is derived inside the kernel from params;
        // dispatching ceil(T/64) threadgroups here raced ceil(T/64) copies
        // over the same entries (the M3 3x root cause).
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeGridScanOffset(_ cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pGridScanOffset)
        enc.setBuffer(binStartsBuffer, offset: 0, index: 0)
        enc.setBuffer(blockSumsBuffer, offset: 0, index: 1)
        enc.setBuffer(paramsBuffer, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: gridBinCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeGridScatter(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pGridScatter)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(binCountsBuffer, offset: 0, index: 1)
        enc.setBuffer(binStartsBuffer, offset: 0, index: 2)
        enc.setBuffer(particleIndicesBuffer, offset: 0, index: 3)
        enc.setBuffer(counterBuffer, offset: 0, index: 4)
        enc.setBuffer(paramsBuffer, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeRepulsionAccumulate(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pRepulsionAccumulate)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(normalsBuffer, offset: 0, index: 1)
        enc.setBuffer(linksBuffer, offset: 0, index: 2)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 3)
        enc.setBuffer(binStartsBuffer, offset: 0, index: 4)
        enc.setBuffer(particleIndicesBuffer, offset: 0, index: 5)
        enc.setBuffer(collisionOffsetBuffer, offset: 0, index: 6)
        enc.setBuffer(counterBuffer, offset: 0, index: 7)
        enc.setBuffer(paramsBuffer, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Grid rebuild + repulsion accumulate ONLY (no integrate) at the exact
    /// current cell count. Diagnostic for --lockstep: the collision field can
    /// be compared cell-by-cell against the CPU brute-force reference before
    /// any feedback amplification. Host must be in sync (GPU idle).
    /// STAP 0 protocol: collisionOffset is explicitly zeroed in the same CB
    /// before the (exactly one) accumulate pass.
    func encodeRepulsionProbe(_ cb: MTLCommandBuffer) {
        guard cellCount > 0 else { return }
        encodeClearOccupancy(cb)
        encodeGridCount(cb, width: cellCount)
        encodeGridScanLocal(cb)
        encodeGridScanBlocks(cb)
        encodeGridScanOffset(cb)
        encodeGridScatter(cb, width: cellCount)
        encodeZeroCollisionOffsets(cb, count: cellCount)
        encodeRepulsionAccumulate(cb, width: cellCount)
    }

    /// Grid rebuild ONLY (count -> scanLocal -> scanBlocks -> scanOffset ->
    /// scatter) at the exact current cell count, no repulsion accumulate.
    /// M4 render-time use: the ray renderer needs the spatial index for the
    /// FINAL state, built ONCE before its tile command buffers (the per-step
    /// rebuild lives inside step() only). The scatter pass leaves binCounts
    /// zeroed again (fill cursor), so the table stays reusable. Host must be
    /// in sync (GPU idle) and gridBits must match the live count — the headless
    /// batch loop syncs with allowGridGrowth: true after every batch.
    func encodeGridRebuild(_ cb: MTLCommandBuffer) {
        guard cellCount > 0 else { return }
        encodeClearOccupancy(cb)   // M6: skip map always corresponds to the live grid
        encodeGridCount(cb, width: cellCount)
        encodeGridScanLocal(cb)
        encodeGridScanBlocks(cb)
        encodeGridScanOffset(cb)
        encodeGridScatter(cb, width: cellCount)
    }

    /// Zero the first `count` entries of collisionOffset (compute kernel in
    /// the same CB, per the STAP 0 protocol: no stale-data ambiguity before
    /// the exactly-once accumulate pass).
    private func encodeZeroCollisionOffsets(_ cb: MTLCommandBuffer, count: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pZeroFloat3)
        enc.setBuffer(collisionOffsetBuffer, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: TEMPORARY DEBUG (CELLDBG; remove after M3 gates are green)

    /// What the GPU actually sees in the params buffer (layout ground truth).
    /// Allocations here are diagnostic-only, never in the per-step path.
    func dbgEchoParams() -> [UInt32] {
        let buf = device.makeBuffer(length: 10 * MemoryLayout<UInt32>.stride,
                                    options: .storageModeShared)!
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return [] }
        enc.setComputePipelineState(pParamEcho)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(paramsBuffer, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let p = buf.contents().assumingMemoryBound(to: UInt32.self)
        return Array(UnsafeBufferPointer(start: p, count: 10))
    }

    /// STAP diagnostic (kept; CELLDBG-gated in runLockstep): encode ONLY the
    /// single grid pass `stage` at the exact current cell count — used to
    /// bisect a broken rebuild pass-by-pass when the lockstep probe fails:
    /// 1 count, 2 scanLocal, 3 scanBlocks, 4 scanOffset, 5 scatter,
    /// 6 zero collisionOffset + accumulate. The count pass is atomic-add, so
    /// it must run EXACTLY ONCE per rebuild; the scatter pass is what zeroes
    /// binCounts again.
    func dbgEncodeProbeStage(_ cb: MTLCommandBuffer, stage: Int, width: Int) {
        switch stage {
        case 1: encodeGridCount(cb, width: width)
        case 2: encodeGridScanLocal(cb)
        case 3: encodeGridScanBlocks(cb)
        case 4: encodeGridScanOffset(cb)
        case 5: encodeGridScatter(cb, width: width)
        case 6:
            encodeZeroCollisionOffsets(cb, count: width)
            encodeRepulsionAccumulate(cb, width: width)
        default: break
        }
    }

    /// queryFingerprint(cell 0) output: 27 hashes, then unique count, total
    /// candidate visits, candidates passing distance+linked (30 uints).
    func dbgQueryFingerprint() -> [UInt32] {
        let buf = device.makeBuffer(length: 30 * MemoryLayout<UInt32>.stride,
                                    options: .storageModeShared)!
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return [] }
        encodeQueryFingerprint(into: enc, out: buf)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let p = buf.contents().assumingMemoryBound(to: UInt32.self)
        return Array(UnsafeBufferPointer(start: p, count: 30))
    }

    private func encodeQueryFingerprint(into enc: MTLComputeCommandEncoder, out: MTLBuffer) {
        enc.setComputePipelineState(pQueryFingerprint)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(binStartsBuffer, offset: 0, index: 1)
        enc.setBuffer(particleIndicesBuffer, offset: 0, index: 2)
        enc.setBuffer(linksBuffer, offset: 0, index: 3)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 4)
        enc.setBuffer(out, offset: 0, index: 5)
        enc.setBuffer(counterBuffer, offset: 0, index: 6)
        enc.setBuffer(paramsBuffer, offset: 0, index: 7)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    }

    private func encodeFoodStep(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pFoodStep)
        enc.setBuffer(rngBuffer, offset: 0, index: 0)
        enc.setBuffer(foodBuffer, offset: 0, index: 1)
        enc.setBuffer(counterBuffer, offset: 0, index: 2)
        enc.setBuffer(paramsBuffer, offset: 0, index: 3)
        enc.setBuffer(nutrientBuffer, offset: 0, index: 4)   // M5: light-mode coupling
        enc.setBuffer(chemUBuffer, offset: 0, index: 5)      // M5: rde coupling
        enc.setBuffer(chemVBuffer, offset: 0, index: 6)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeDivisionDecide(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pDivisionDecide)
        enc.setBuffer(foodBuffer, offset: 0, index: 0)
        enc.setBuffer(linksBuffer, offset: 0, index: 1)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 2)
        enc.setBuffer(splitFlagsBuffer, offset: 0, index: 3)
        enc.setBuffer(counterBuffer, offset: 0, index: 4)
        enc.setBuffer(paramsBuffer, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeDivisionExecute(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pDivisionExecute)
        enc.setBuffer(nextPositionsBuffer, offset: 0, index: 0)
        enc.setBuffer(linksBuffer, offset: 0, index: 1)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 2)
        enc.setBuffer(foodBuffer, offset: 0, index: 3)
        enc.setBuffer(rngBuffer, offset: 0, index: 4)
        enc.setBuffer(splitFlagsBuffer, offset: 0, index: 5)
        enc.setBuffer(counterBuffer, offset: 0, index: 6)
        enc.setBuffer(paramsBuffer, offset: 0, index: 7)
        enc.setBuffer(nutrientBuffer, offset: 0, index: 8)   // M5: cytoplasm split
        enc.setBuffer(chemUBuffer, offset: 0, index: 9)      // M5: daughter inheritance
        enc.setBuffer(chemVBuffer, offset: 0, index: 10)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeFinalizeFood(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pFinalizeFood)
        enc.setBuffer(splitFlagsBuffer, offset: 0, index: 0)
        enc.setBuffer(foodBuffer, offset: 0, index: 1)
        enc.setBuffer(counterBuffer, offset: 0, index: 2)
        enc.setBuffer(paramsBuffer, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: M5 growth-mode chemical passes (see Shaders.swift kernel comments)

    private func encodeBoundMax(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pBoundMax)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(boundBuffer, offset: 0, index: 1)
        enc.setBuffer(counterBuffer, offset: 0, index: 2)
        enc.setBuffer(paramsBuffer, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Exactly one thread: zeroes the atomic max |P|² accumulator for the
    /// next step (encoder order: boundMax -> illuminate -> clearBound).
    private func encodeClearBound(_ cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pClearBound)
        enc.setBuffer(boundBuffer, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeIlluminate(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pIlluminate)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(radiiBuffer, offset: 0, index: 1)
        enc.setBuffer(binStartsBuffer, offset: 0, index: 2)
        enc.setBuffer(particleIndicesBuffer, offset: 0, index: 3)
        enc.setBuffer(nutrientBuffer, offset: 0, index: 4)
        enc.setBuffer(boundBuffer, offset: 0, index: 5)
        enc.setBuffer(counterBuffer, offset: 0, index: 6)
        enc.setBuffer(paramsBuffer, offset: 0, index: 7)
        enc.setBuffer(litFlagsBuffer, offset: 0, index: 8)
        enc.setBuffer(occupancyBuffer, offset: 0, index: 9)   // M6 coarse occupancy skip
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeNutrientDiffuse(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pNutrientDiffuse)
        enc.setBuffer(linksBuffer, offset: 0, index: 0)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 1)
        enc.setBuffer(nutrientBuffer, offset: 0, index: 2)
        enc.setBuffer(nextNutrientBuffer, offset: 0, index: 3)
        enc.setBuffer(counterBuffer, offset: 0, index: 4)
        enc.setBuffer(paramsBuffer, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeSeedChemicals(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pSeedChemicals)
        enc.setBuffer(linksBuffer, offset: 0, index: 0)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 1)
        enc.setBuffer(chemUBuffer, offset: 0, index: 2)
        enc.setBuffer(chemVBuffer, offset: 0, index: 3)
        enc.setBuffer(counterBuffer, offset: 0, index: 4)
        enc.setBuffer(paramsBuffer, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeCellTint(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pCellTint)
        enc.setBuffer(chemUBuffer, offset: 0, index: 0)
        enc.setBuffer(chemVBuffer, offset: 0, index: 1)
        enc.setBuffer(tintBuffer, offset: 0, index: 2)
        enc.setBuffer(counterBuffer, offset: 0, index: 3)
        enc.setBuffer(paramsBuffer, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeRdeStep(_ cb: MTLCommandBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pRdeStep)
        enc.setBuffer(linksBuffer, offset: 0, index: 0)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 1)
        enc.setBuffer(chemUBuffer, offset: 0, index: 2)
        enc.setBuffer(chemVBuffer, offset: 0, index: 3)
        enc.setBuffer(nextChemUBuffer, offset: 0, index: 4)
        enc.setBuffer(nextChemVBuffer, offset: 0, index: 5)
        enc.setBuffer(counterBuffer, offset: 0, index: 6)
        enc.setBuffer(paramsBuffer, offset: 0, index: 7)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Re-seed the RDE chemicals (u = 1, v = seeded noise) in a waited command
    /// buffer. GUI growth-mode switches call this when entering rde mode: v = 0
    /// everywhere is a fixed point of the dynamics (no pattern would form from
    /// a dead field). Deterministic (same seed → same pattern).
    func reseedChemicals() {
        guard cellCount > 0, let cb = queue.makeCommandBuffer() else { return }
        encodeSeedChemicals(cb, width: cellCount)
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Recompute the per-cell illumination verdicts (litFlags) for the CURRENT
    /// state — render-time use for loaded states (a state snapshot stores
    /// geometry, not litFlags; the directional solid style needs the verdict
    /// the growth's last step produced). Runs boundMax + illuminate + clearBound
    /// at the exact cell count; host must be in sync (GPU idle). CONTRACT: the
    /// caller must rebuild the grid (encodeGridRebuild — which also refreshes
    /// the M6 coarse occupancy the march skips on) in the SAME or an earlier
    /// command buffer. The nutrient side effect (one decay+production pass) is
    /// harmless at render time.
    func encodeIlluminationRefresh(_ cb: MTLCommandBuffer) {
        guard cellCount > 0 else { return }
        encodeBoundMax(cb, width: cellCount)
        encodeIlluminate(cb, width: cellCount)
        encodeClearBound(cb)
    }

    private func encodeRadii(into cb: MTLCommandBuffer, positions: MTLBuffer, width: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pComputeRadii)
        enc.setBuffer(positions, offset: 0, index: 0)
        enc.setBuffer(linksBuffer, offset: 0, index: 1)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 2)
        enc.setBuffer(radiiBuffer, offset: 0, index: 3)
        enc.setBuffer(counterBuffer, offset: 0, index: 4)
        enc.setBuffer(paramsBuffer, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Radii pass at the exact current cell count (host must be in sync).
    /// Used before screenshots so freshly grown daughters have valid radii.
    func refreshRadii(_ cb: MTLCommandBuffer) {
        guard cellCount > 0 else { return }
        encodeRadii(into: cb, positions: positionsBuffer, width: cellCount)
    }

    // MARK: - Stats

    /// Encode the reduction kernel (call in a CB that also runs/ran sim work;
    /// read results only after the CB completes).
    func encodeStats(_ cb: MTLCommandBuffer) {
        guard cellCount > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pStats)
        enc.setBuffer(positionsBuffer, offset: 0, index: 0)
        enc.setBuffer(linksBuffer, offset: 0, index: 1)
        enc.setBuffer(linkCountsBuffer, offset: 0, index: 2)
        enc.setBuffer(statsPartialsBuffer, offset: 0, index: 3)
        enc.setBuffer(counterBuffer, offset: 0, index: 4)
        enc.setBuffer(paramsBuffer, offset: 0, index: 5)
        let tgs = (dispatchWidth + 63) / 64
        enc.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Mean link length (call after the CB containing encodeStats completed).
    func readMeanLinkLength() -> Float {
        guard cellCount > 0 else { return 0 }
        let tgs = (cellCount + 63) / 64
        let p = statsPartialsBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        var sum: Float = 0
        var count: Float = 0
        for i in 0..<tgs {
            sum += p[i].x
            count += p[i].y
        }
        return count > 0 ? sum / count : 0
    }

    // MARK: - Host sync / readback

    /// Read the device cell counter into the host mirror. ONLY call while the
    /// GPU is idle (after waitUntilCompleted, or inside addCompletedHandler).
    ///
    /// allowGridGrowth: grow the hash table T when the cell count outgrew it.
    /// Only pass true from paths that guarantee NO command buffer is in
    /// flight (headless batch loops after waitUntilCompleted): the update
    /// writes the params buffer, and an already-enqueued step could otherwise
    /// execute its count and scatter kernels with different hash bits,
    /// corrupting that step's bin bookkeeping. The GUI completion handler
    /// keeps the reset-time T (correct for any count, just less efficient).
    func syncCellCountFromDevice(allowGridGrowth: Bool = false) {
        let p = counterBuffer.contents().assumingMemoryBound(to: UInt32.self)
        cellCount = Int(min(p[0], UInt32(maxCells)))
        params.cellCount = UInt32(cellCount)
        if allowGridGrowth {
            let old = params.gridBitsX + params.gridBitsY + params.gridBitsZ
            setGridBits(forCellCount: cellCount)
            if params.gridBitsX + params.gridBitsY + params.gridBitsZ != old {
                syncParams()
            }
        }
    }

    // Raw buffer views for CPU-side validation (--selftest). Valid only while
    // the GPU is idle.
    var positionsPointer: UnsafeMutablePointer<SIMD3<Float>> {
        positionsBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
    }
    var linksPointer: UnsafeMutablePointer<UInt32> {
        linksBuffer.contents().assumingMemoryBound(to: UInt32.self)
    }
    var linkCountsPointer: UnsafeMutablePointer<UInt32> {
        linkCountsBuffer.contents().assumingMemoryBound(to: UInt32.self)
    }
    var foodPointer: UnsafeMutablePointer<Float> {
        foodBuffer.contents().assumingMemoryBound(to: Float.self)
    }
    var collisionOffsetPointer: UnsafeMutablePointer<SIMD3<Float>> {
        collisionOffsetBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
    }
    var binCountsPointer: UnsafeMutablePointer<UInt32> {
        binCountsBuffer.contents().assumingMemoryBound(to: UInt32.self)
    }
    var binStartsPointer: UnsafeMutablePointer<UInt32> {
        binStartsBuffer.contents().assumingMemoryBound(to: UInt32.self)
    }
    var particleIndicesPointer: UnsafeMutablePointer<UInt32> {
        particleIndicesBuffer.contents().assumingMemoryBound(to: UInt32.self)
    }
    var blockSumsPointer: UnsafeMutablePointer<UInt32> {
        blockSumsBuffer.contents().assumingMemoryBound(to: UInt32.self)
    }

    /// Direct position access (readback; only for small cell counts / validation).
    func readPositions() -> [SIMD3<Float>] {
        let p = positionsBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
        return Array(UnsafeBufferPointer(start: p, count: cellCount))
    }

    /// Collision field readback (--lockstep probe; only for small cell counts).
    func readCollisionOffsets() -> [SIMD3<Float>] {
        let p = collisionOffsetBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
        return Array(UnsafeBufferPointer(start: p, count: cellCount))
    }

    // --- M5 chemical readback (validation/probe; only for small cell counts,
    // GPU must be idle) ---

    var normalsPointer: UnsafeMutablePointer<SIMD3<Float>> {
        normalsBuffer.contents().assumingMemoryBound(to: SIMD3<Float>.self)
    }
    var nutrientPointer: UnsafeMutablePointer<Float> {
        nutrientBuffer.contents().assumingMemoryBound(to: Float.self)
    }
    var chemUPointer: UnsafeMutablePointer<Float> {
        chemUBuffer.contents().assumingMemoryBound(to: Float.self)
    }
    var chemVPointer: UnsafeMutablePointer<Float> {
        chemVBuffer.contents().assumingMemoryBound(to: Float.self)
    }
    var radiiPointer: UnsafeMutablePointer<Float> {
        radiiBuffer.contents().assumingMemoryBound(to: Float.self)
    }

    func readNutrient() -> [Float] {
        Array(UnsafeBufferPointer(start: nutrientPointer, count: cellCount))
    }
    func readChemU() -> [Float] {
        Array(UnsafeBufferPointer(start: chemUPointer, count: cellCount))
    }
    func readChemV() -> [Float] {
        Array(UnsafeBufferPointer(start: chemVPointer, count: cellCount))
    }

    // MARK: - State snapshots (CGST; M6 unification)
    //
    // One writer/reader pair for every consumer (CLI --state-save/--state-load,
    // golden .state files, GUI Save/Load). Binary format, all little-endian:
    //
    //   v1 (M4): "CGST" | u32 ver=1 | u32 count | u32 maxValence
    //            | positions (N x 16 B: 3 f32 + pad)
    //            | links (N x maxValence x u32) | linkCounts (N x u32)
    //   v2 (M5): v1 + growth-config block (u32 growthMode + 13 floats: the
    //            light dir + light/rde coupling). Loaded by restoring the
    //            config and then reset()-ing (chemicals/RNG/food re-seeded).
    //   v3 (M6): v1 body + FULL per-cell state + raw SimParams — everything
    //            the sim needs to CONTINUE exactly where it left off:
    //              food (N x f32) | nutrient (N x f32) | chemU (N x f32)
    //            | chemV (N x f32) | tint (N x 16 B) | rngState (N x u64)
    //            | SimParams (128 B raw)
    //            Loaded WITHOUT reset(): buffers installed directly, normals/
    //            radii/tint recomputed (deterministic functions of the state).
    //
    // GPU must be idle for both save and load.

    enum StateSnapshotError: Error {
        case notCGST(String)
        case unsupportedVersion(Int)
        case sizeMismatch(path: String, got: Int, expected: Int)
        case maxValenceMismatch(Int)
    }

    /// Write the complete live state (v3). GPU must be idle.
    func saveState(to path: String) throws {
        let count = cellCount
        var bytes = [UInt8]()
        bytes.reserveCapacity(16 + count * (MemoryLayout<SIMD3<Float>>.stride
                                             + CellSim.maxValence * MemoryLayout<UInt32>.stride
                                             + 4 * MemoryLayout<UInt32>.stride
                                             + MemoryLayout<Float>.stride * 4
                                             + MemoryLayout<UInt64>.stride)
                              + MemoryLayout<SimParams>.size)
        func append(_ v: UInt32) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { bytes.append(contentsOf: $0) }
        }
        func append(_ v: Float) {
            var x = v.bitPattern.littleEndian
            withUnsafeBytes(of: &x) { bytes.append(contentsOf: $0) }
        }
        func append(_ v: UInt64) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { bytes.append(contentsOf: $0) }
        }
        bytes.append(contentsOf: Array("CGST".utf8))
        append(3 as UInt32)                     // version (v3: full state)
        append(UInt32(count))
        append(UInt32(CellSim.maxValence))
        let pos = UnsafeBufferPointer(start: positionsPointer, count: count)
        for p in pos {
            for c in [p.x, p.y, p.z] {
                var f = c.bitPattern.littleEndian
                withUnsafeBytes(of: &f) { bytes.append(contentsOf: $0) }
            }
            var pad: UInt32 = 0
            withUnsafeBytes(of: &pad) { bytes.append(contentsOf: $0) }
        }
        let links = UnsafeBufferPointer(start: linksPointer, count: count * CellSim.maxValence)
        for l in links { append(l) }
        let counts = UnsafeBufferPointer(start: linkCountsPointer, count: count)
        for c in counts { append(c) }
        // --- v3 payload: full per-cell state ---
        for f in UnsafeBufferPointer(start: foodPointer, count: count) { append(f) }
        for f in UnsafeBufferPointer(start: nutrientPointer, count: count) { append(f) }
        for f in UnsafeBufferPointer(start: chemUPointer, count: count) { append(f) }
        for f in UnsafeBufferPointer(start: chemVPointer, count: count) { append(f) }
        let tint = UnsafeBufferPointer(start: tintBuffer.contents()
            .assumingMemoryBound(to: SIMD3<Float>.self), count: count)
        for t in tint {
            for c in [t.x, t.y, t.z] {
                var f = c.bitPattern.littleEndian
                withUnsafeBytes(of: &f) { bytes.append(contentsOf: $0) }
            }
            var pad: UInt32 = 0
            withUnsafeBytes(of: &pad) { bytes.append(contentsOf: $0) }
        }
        let rng = UnsafeBufferPointer(start: rngBuffer.contents()
            .assumingMemoryBound(to: UInt64.self), count: count)
        for r in rng { append(r) }
        // Raw SimParams (the simulation contract; appended-only layout).
        withUnsafeBytes(of: &params) { src in
            bytes.append(contentsOf: src.prefix(MemoryLayout<SimParams>.size))
        }
        try Data(bytes).write(to: URL(fileURLWithPath: path))
    }

    /// Load a state snapshot of ANY version (v1/v2/v3) and install it.
    /// Returns the snapshot version + the growth mode it carries (callers
    /// print diagnostics). GPU must be idle.
    ///
    /// v1/v2 semantics preserved from the M4/M5 loader: v2 restores the
    /// growth config BEFORE reset() so cellTint and the unified render light
    /// see the grown mode; reset() re-seeds food/RNG/chemicals (v2 files
    /// predate chemical persistence — chemicals reseed, food re-seeded).
    @discardableResult
    func loadState(from path: String) throws -> (version: Int, growthMode: UInt32) {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        func u32(_ off: Int) -> UInt32 {
            UInt32(data[off]) | (UInt32(data[off+1]) << 8) | (UInt32(data[off+2]) << 16) | (UInt32(data[off+3]) << 24)
        }
        func f32(_ off: Int) -> Float {
            Float(bitPattern: u32(off))
        }
        guard data.count >= 16, String(bytes: data[0..<4], encoding: .ascii) == "CGST" else {
            throw StateSnapshotError.notCGST(path)
        }
        let version = Int(u32(4))
        guard version >= 1 && version <= 3 else {
            throw StateSnapshotError.unsupportedVersion(version)
        }
        let count = Int(u32(8))
        let maxValence = Int(u32(12))
        guard maxValence == CellSim.maxValence else {
            throw StateSnapshotError.maxValenceMismatch(maxValence)
        }
        let baseSize = 16 + count * (MemoryLayout<SIMD3<Float>>.stride + maxValence * 4 + 4)
        let v3Size = baseSize + count * (4 * 4 + MemoryLayout<Float>.stride * 4
                                         + MemoryLayout<UInt64>.stride)
                     + MemoryLayout<SimParams>.size
        let expected = version == 2 ? baseSize + 56      // v2 block: 1 uint + 13 floats
                     : version == 3 ? v3Size
                     : baseSize
        guard data.count == expected else {
            throw StateSnapshotError.sizeMismatch(path: path, got: data.count, expected: expected)
        }

        if version == 2 {
            // Growth block: 1 uint + 13 floats after the linkCounts block.
            var off = baseSize
            params.growthMode = u32(off); off += 4
            params.lightDirX = f32(off); off += 4
            params.lightDirY = f32(off); off += 4
            params.lightDirZ = f32(off); off += 4
            params.lightDiffusionRate = f32(off); off += 4
            params.lightNutrient = f32(off); off += 4
            params.lightDecay = f32(off); off += 4
            params.lightBase = f32(off); off += 4
            params.rdeF = f32(off); off += 4
            params.rdeK = f32(off); off += 4
            params.rdeDu = f32(off); off += 4
            params.rdeDv = f32(off); off += 4
            params.rdeBase = f32(off); off += 4
            params.rdeGain = f32(off); off += 4
        }

        // Geometry blocks (identical layout in every version).
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

        if version <= 2 {
            try resetWithGrowthConfig(positions: positions, links: links)
            return (version, params.growthMode)
        }

        // --- v3: full restore, NO reseeding ---
        var off = baseSize
        func readFloats(_ n: Int) -> [Float] {
            var v = [Float](repeating: 0, count: n)
            data.withUnsafeBytes { src in
                v.withUnsafeMutableBytes { dst in
                    dst.copyBytes(from: UnsafeRawBufferPointer(rebasing: src.dropFirst(off).prefix(n * 4)))
                }
            }
            off += n * 4
            return v
        }
        let food = readFloats(count)
        let nutrient = readFloats(count)
        let chemU = readFloats(count)
        let chemV = readFloats(count)
        // Tint: N x 16 B (3 f32 + pad).
        var tint = [SIMD3<Float>](repeating: .zero, count: count)
        data.withUnsafeBytes { src in
            tint.withUnsafeMutableBytes { dst in
                dst.copyBytes(from: UnsafeRawBufferPointer(rebasing: src.dropFirst(off).prefix(count * MemoryLayout<SIMD3<Float>>.stride)))
            }
        }
        off += count * MemoryLayout<SIMD3<Float>>.stride
        // RNG: N x u64.
        var rng = [UInt64](repeating: 0, count: count)
        data.withUnsafeBytes { src in
            rng.withUnsafeMutableBytes { dst in
                dst.copyBytes(from: UnsafeRawBufferPointer(rebasing: src.dropFirst(off).prefix(count * MemoryLayout<UInt64>.stride)))
            }
        }
        off += count * MemoryLayout<UInt64>.stride
        // Raw SimParams.
        var restored = SimParams()
        data.withUnsafeBytes { src in
            withUnsafeMutableBytes(of: &restored) { dst in
                dst.copyBytes(from: UnsafeRawBufferPointer(rebasing: src.dropFirst(off).prefix(MemoryLayout<SimParams>.size)))
            }
        }
        // Safety clamps on restored params: never exceed THIS sim's capacity;
        // the host mirror of the count is authoritative from the header.
        restored.maxCells = UInt32(min(Int(restored.maxCells), maxCells))
        restored.cellCount = UInt32(count)
        params = restored

        installFullState(positions: positions, flatLinks: flatLinks, counts: counts,
                         food: food, nutrient: nutrient, chemU: chemU, chemV: chemV,
                         tint: tint, rng: rng)
        return (version, params.growthMode)
    }

    /// v1/v2 loader path (historical semantics): install geometry via the
    /// standard reset (which re-seeds food/RNG deterministically and seeds
    /// chemicals).
    private func resetWithGrowthConfig(positions: [SIMD3<Float>], links: [[UInt32]]) throws {
        reset(positions: positions, links: links)
    }

    /// Install a COMPLETE state snapshot (v3) without reseeding anything:
    /// positions/links/counts/food/nutrient/u/v/tint/rng + params, then
    /// recompute the derived per-cell data (normals, radii, tint) in a waited
    /// command buffer. Derived state that only render paths consume
    /// (litFlags) is zeroed — directional renders call
    /// encodeIlluminationRefresh, and the next light-mode step re-marches.
    private func installFullState(positions: [SIMD3<Float>], flatLinks: [UInt32],
                                  counts: [UInt32], food: [Float], nutrient: [Float],
                                  chemU: [Float], chemV: [Float], tint: [SIMD3<Float>],
                                  rng: [UInt64]) {
        precondition(positions.count <= maxCells, "state exceeds maxCells")
        cellCount = positions.count
        params.cellCount = UInt32(cellCount)
        setGridBits(forCellCount: cellCount)
        memset(binCountsBuffer.contents(), 0, binCountsBuffer.length)
        syncParams()
        usingA = true
        usingNutA = true
        usingChemA = true
        stepIndex = 0

        func copy<T>(_ src: [T], to buffer: MTLBuffer) {
            src.withUnsafeBytes { s in
                buffer.contents().copyMemory(from: s.baseAddress!,
                                             byteCount: src.count * MemoryLayout<T>.stride)
            }
        }
        copy(positions, to: positionsA)
        copy(flatLinks, to: linksBuffer)
        copy(counts, to: linkCountsBuffer)
        copy(food, to: foodBuffer)
        copy(nutrient, to: nutrientA)
        copy(chemU, to: chemUA)
        copy(chemV, to: chemVA)
        copy(tint, to: tintBuffer)
        copy(rng, to: rngBuffer)
        counterBuffer.contents().assumingMemoryBound(to: UInt32.self)[0] = UInt32(cellCount)
        memset(boundBuffer.contents(), 0, boundBuffer.length)
        memset(litFlagsBuffer.contents(), 0, litFlagsBuffer.length)

        // Derived per-cell data: normals + radii + tint (tint is a pure
        // function of u/v/growthMode, so recomputing it reproduces the saved
        // tint bit-for-bit; recomputing anyway keeps daughters-after-load
        // consistent).
        guard let cb = queue.makeCommandBuffer() else { fatalError("no command buffer") }
        encodeNormals(cb, width: cellCount)
        encodeRadii(into: cb, positions: positionsA, width: cellCount)
        encodeCellTint(cb, width: cellCount)
        cb.commit()
        cb.waitUntilCompleted()
    }
}

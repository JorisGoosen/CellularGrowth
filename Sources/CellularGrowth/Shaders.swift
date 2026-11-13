//  Shaders.swift
//  Metal shader sources (MSL), compiled at runtime with device.makeLibrary(source:).
//  Kept as string constants so both `swift run` and the .app bundle work
//  without SPM resource plumbing.
//
//  COMPOSED CHUNKS — single-source rule: no verbatim MSL duplicates between
//  the three libraries. The former copies of struct SimParams (render +
//  raytrace) and of binHash / coarseHash / coarseNeighbourhoodClear /
//  lightVisibility (raytrace) are merged into `common`.
//
//    common           canonical struct SimParams (appended-only layout
//                     contract with the Swift SimParams; all three libraries
//                     bind the SAME params buffer memory) + the helpers
//                     shared by more than one library: per-cell RNG, the
//                     fine spatial hash (binHash / binHashOfPos), the coarse
//                     occupancy hash (coarseHash / coarseNeighbourhoodClear)
//                     and the DDA light march (lightVisibility).
//    simKernels       simulation kernels: normals, grid rebuild passes,
//                     repulsion, forces, food, division, chemicals, tint,
//                     pulse, stats (+ the CELLDBG probe kernels).
//    renderKernels    raster impostor shaders: Camera, VOut/FOut, solid +
//                     interactive X-ray (render-only structs stay here).
//    raytraceKernels  offline ray tracer: RenderParams (render-only uniform
//                     block), traversal/AO/shadow helpers, raytraceTile,
//                     raytraceProbe.
//
//  makeLibrary contract (names are load-bearing, do not rename):
//    static let sim      -> Core/CellSim.swift
//    static let render   -> Render/Renderer.swift
//    static let raytrace -> Render/RayRenderer.swift

import Foundation

enum Shaders {

    // MARK: - `common`: canonical SimParams + helpers shared by >1 library

    static let common = """
    #include <metal_stdlib>
    using namespace metal;

    // NOTE: layout must mirror Swift struct SimParams exactly (48 x 4 bytes =
    // 192). New fields are APPENDED at the end on both sides (AGENTS.md
    // SimParams rule). Size history: 144 B through M6, 148 B in V2.1
    // (lightSoft), 160 B from V2.2 (tintMode + heatScale + pulseDecay),
    // 176 B from V2.3 (lamelRate + dalRate + interiorDivider + phaseSwitchX),
    // 192 B from V2.4 (typeMode + differentiationPhase + typeCapV +
    // typeStemV) — the CGST v3 reader accepts every prefix length.
    struct SimParams {
        float linkRestLength;
        float springFactor;
        float planarFactor;
        float bulgeFactor;
        float radiusScale;
        uint  cellCount;         // host mirror only; kernels bound on the device counter
        uint  maxValence;
        // --- M2 (appended) ---
        float foodIncrement;     // per-step food += foodIncrement * U[0,1)
        float splitThreshold;
        uint  divisionEnabled;   // 0/1 flag
        uint  rngSeed;           // consumed host-side at reset only
        uint  maxCells;          // hard capacity enforced by divisionExecute
        // --- M3 (appended) ---
        float roi;               // radius of influence; bin size = roi (h >= roi rule)
        float repulsionStrength; // 0 disables repulsion (host then skips the grid kernels)
        uint  gridBitsX;         // packed hash bits per axis; T = 2^(bx+by+bz)
        uint  gridBitsY;         // (host-derived from the live cell count, clamped
        uint  gridBitsZ;         //  to [2^8, 2^22]; buffers sized for the 2^22 max)
        // --- M5 (appended) ---
        uint  growthMode;        // 0 uniform random, 1 light nutrient, 2 Gray-Scott RDE
        float lightDirX;         // unit vector TOWARD the light; three separate
        float lightDirY;         // floats — no float3-in-struct alignment traps
        float lightDirZ;         // (host normalizes before syncParams)
        float lightDiffusionRate;// Jacobi nutrient diffusion along links, [0,1]
        float lightNutrient;     // nutrient produced per step in lit cells
        float lightDecay;        // first-order nutrient decay per step, [0,1)
        float lightBase;         // food coupling floor: increment ∝ (lightBase + nutrient)
        uint  lightInterval;     // illumination-march interval in steps (march is
                                 // the expensive pass; the shadow pattern evolves
                                 // on ~100-step scales — production is pulsed
                                 // lightInterval× so the average rate is unchanged)
        float rdeF;              // Gray-Scott feed rate
        float rdeK;              // Gray-Scott kill rate
        float rdeDu;             // u diffusion (graph-normalized, see rdeStep)
        float rdeDv;             // v diffusion
        float rdeBase;           // food coupling floor in rde mode
        float rdeGain;           // food coupling gain on the v chemical
        // --- M6 (appended) ---
        uint  gridCoarseBitsX;   // coarse occupancy hash bits per axis (max(bits-3, 2)):
        uint  gridCoarseBitsY;   // a bit per 8^3-fine-bin block for the light-march
        uint  gridCoarseBitsZ;   // empty-interior skip (collisions conservative)
        uint  lightCoarseSkip;   // 1 = coarse skip enabled (kill switch for A/B)
        // --- V2.1 (appended; struct grows 144 -> 148 bytes) ---
        float lightSoft;         // light angular radius (radians) of the soft light
                                 // disk: 0 = the legacy binary lit/unlit verdict,
                                 // > 0 = fractional visibility (see lightVisibility).
                                 // Shared by the growth illumination AND the ray
                                 // tracer's shadow-ray cone (one physical light).
        // --- V2.2 (appended; struct grows 148 -> 160 bytes) ---
        uint  tintMode;          // cellTint colour mode: 0 auto (chemical iff
                                 // growthMode == rde — the pre-V2.2 behaviour),
                                 // 1 none (neutral), 2 chemical (rde u/v map in
                                 // every mode), 3 food/nutrient heatmap,
                                 // 4 mitosis pulse
        float heatScale;         // food-heatmap normalization scale;
                                 // <= 1e-6 selects the shader-side auto default
                                 // (see the cellTint comment)
        float pulseDecay;        // per-step multiplier on the division pulse
                                 // (0.95 default; division writes 1.0)
        // --- V2.3 (appended; struct grows 160 -> 176 bytes) ---
        float lamelRate;         // phase-2 food-increment multiplier for LAMEL
                                 // cells at EDGE links (neighbour = stem/cap)
        float dalRate;           // phase-2 food-increment multiplier for DAL
                                 // cells at edge links
        float interiorDivider;   // phase-2 interior rate: lamel<->dal links
                                 // (no edge neighbour) grow at
                                 // lamelRate / max(interiorDivider, 1)
        uint  phaseSwitchX;      // two-phase growth: phase 2 (voeg-only
                                 // division) begins when the global division
                                 // counter reaches this many EXECUTED
                                 // divisions; 0 = never (pure phase 1).
                                 // Type-0 (untyped) runs ignore the phase
                                 // entirely — the rule matrix is a no-op
                                 // (see divisionDecide / foodStep).
        // --- V2.4 (appended; struct grows 176 -> 192 bytes) ---
        uint  typeMode;          // rule-matrix mode: 0 band (the V2.3 polar
                                 // matrix: stem edge-only, voeg dormant/band-
                                 // only), 1 region (RDE-typed runs: every
                                 // type divides uniformly within its own
                                 // region — the patterned rate is carried by
                                 // the v coupling in drive, and the two-phase
                                 // machinery is not applied). Type-0 cells
                                 // behave identically in BOTH modes (the
                                 // type-0 contract).
        uint  differentiationPhase; // 1 = the DIFFERENTIATION PHASE: the whole
                                 // division block (food + decide + execute +
                                 // finalize) is skipped — the division
                                 // counter is frozen and food stops
                                 // accumulating (growth resumes without a
                                 // synchronized burst) — while the RDE keeps
                                 // stepping and typeTransition converts
                                 // undifferentiated cells. Only rde mode
                                 // types; in uniform/light modes the phase is
                                 // a pure growth pause.
        float typeCapV;          // typeTransition: v >= typeCapV -> cap (type 4)
        float typeStemV;         // typeTransition: v <= typeStemV -> stem
                                 // (type 1); between the thresholds the cell
                                 // stays undifferentiated (type 0). Host
                                 // validates typeStemV < typeCapV.
    };
    // (alignment 16!). Links are stored strided per cell (cell * maxValence + k)
    // in CYCLIC RING ORDER (neighbors consecutive around the vertex), so the
    // area-weighted vertex normal is the sum of cross products of consecutive
    // ring vectors. Rings are initialized CCW as seen from outside, which makes
    // these cross products point outward. Cell division (below) rebuilds rings
    // in place and preserves this cyclic order — do not re-sort.
    //
    // The authoritative live cell count is cellCounter[0]: kernels bound on it,
    // never on params.cellCount (a host-side mirror only). divisionExecute grows
    // the counter with atomic_fetch_add; the host reads it back only while the
    // GPU is idle (CellSim.syncCellCountFromDevice).
    //
    // Dispatch width policy: step() over-dispatches with
    // min(maxCells, 2 * host cellCount). Under-coverage (only possible with
    // pathological within-batch growth) merely stalls affected cells until the
    // next host sync — it cannot corrupt state, because divisionDecide and
    // divisionExecute share the same width, so a stale claim flag is never
    // consumed, and every other kernel only skips work it would redo next step.

    constant uint kMaxValence = 16;   // must equal CellSim.maxValence (params.maxValence)

    // --- RNG: PCG32-style per-cell stream (fixed seed => deterministic runs) ---
    // One 64-bit LCG state per cell + O'Neill output permutation. Every kernel's
    // per-cell math is index-independent, so runs are deterministic up to the
    // index order in which daughters happen to be allocated (topology and
    // geometry do not depend on that order).
    constant uint64_t kRngMul = 6364136223846793005ULL;
    constant uint64_t kRngInc = 1442695040888963407ULL;

    inline uint pcgOutput(uint64_t state) {
        uint x = (uint)(((state >> 18u) ^ state) >> 27u);
        uint rot = (uint)(state >> 59u);
        return (x >> rot) | (x << ((32u - rot) & 31u));
    }

    // Advance the cell's RNG state; return a uniform float in [0, 1).
    inline float rand01(device uint64_t* state) {
        uint64_t s = *state * kRngMul + kRngInc;
        *state = s;
        return (float)(pcgOutput(s) >> 8) * (1.0f / 16777216.0f);
    }

    // ---- M3: spatial hash grid for repulsion (paper §4) --------------------
    //
    // Hashed sparse uniform grid, rebuilt EVERY step (everything moves).
    // Bin size h = roi: the 3x3x3 bin neighbourhood of a query cell spans
    // [ix-1, ix+1] per axis, a cube of side 3*roi around the cell's own bin,
    // which provably contains every point within roi of the cell (h >= roi
    // rule: the worst case has the cell at a bin corner and the candidate a
    // full roi beyond the neighbouring bin face, still inside the cube).
    // Rebuild = counting sort in five passes:
    //   gridCount       pass 1: atomic_fetch_add on binCounts[hash(P)]
    //   gridScanLocal   pass 2a: exclusive scan of 64-bin chunks per threadgroup
    //   gridScanBlocks  pass 2b: exclusive scan of the per-threadgroup sums
    //   gridScanOffset  pass 2c: fold block offsets back -> global exclusive prefix
    //   gridScatter     pass 3: scatter particle indices; binCounts doubles as
    //                   the per-bin fill cursor and ends at 0 again, so the
    //                   counts array self-cleans for the next rebuild (the
    //                   count pass simply adds on top of zeros).
    //
    // Hash: bit-PACKED bin coordinates, no mixing:
    //   h = (ix & maskX) | ((iy & maskY) << bitsX) | ((iz & maskZ) << (bitsX+bitsY))
    // Chosen over the classic multiplicative hash (ix*73856093 ^ iy*19349663 ^
    // iz*83492791) & (T-1): with a mixed hash, two DIFFERENT coordinates of
    // one query's 3x3x3 neighbourhood can collide, the same bin would be
    // visited twice, and every candidate in it would be double-counted — a
    // real fidelity bug (each r in A must contribute exactly once). The
    // packed form is injective on every 3x3x3 neighbourhood by construction:
    // per axis the neighbourhood spans 2 consecutive values, strictly less
    // than the per-axis modulus 2^bits (bits >= 2). Far-apart coordinate
    // wrap-around (|coord| >= 2^bits per axis) only merges distant bins,
    // which is harmless: the distance test filters those candidates.
    // T = 2^(bitsX+bitsY+bitsZ) is a power of two derived host-side from the
    // live cell count (T >= 2 x cells keeps chains short), clamped to
    // [2^8, 2^22]; buffers are preallocated for the 2^22 maximum, so T never
    // triggers reallocation. Memory stays O(maxCells) regardless of how
    // thin/branchy the structure is (a dense grid over the bounding box
    // would not: it is O(bbox/h^3), ruinous for dendritic forms).

    static inline uint binHash(int3 c, constant SimParams& p) {
        uint maskX = (1u << p.gridBitsX) - 1u;
        uint maskY = (1u << p.gridBitsY) - 1u;
        uint maskZ = (1u << p.gridBitsZ) - 1u;
        return ((uint)c.x & maskX)
             | (((uint)c.y & maskY) << p.gridBitsX)
             | (((uint)c.z & maskZ) << (p.gridBitsX + p.gridBitsY));
    }

    static inline uint binHashOfPos(float3 P, constant SimParams& p) {
        return binHash(int3((int)floor(P.x / p.roi),
                            (int)floor(P.y / p.roi),
                            (int)floor(P.z / p.roi)), p);
    }

    // M6: coarse occupancy hash — the SAME packed scheme over the 8³-block
    // coordinates with FIXED 7 bits per axis (params.gridCoarseBits*; sized
    // for block coordinates ±63 — see CellSim.setGridBits). Injectivity on a
    // 3×3×3 neighbourhood is NOT required: counts ADD over colliding block
    // coordinates, so a non-zero count may over-report occupancy (skip
    // disabled — conservative); a count of 0 still proves no cell maps to
    // the slot at all.
    static inline uint coarseHash(int3 c, constant SimParams& p) {
        uint maskX = (1u << p.gridCoarseBitsX) - 1u;
        uint maskY = (1u << p.gridCoarseBitsY) - 1u;
        uint maskZ = (1u << p.gridCoarseBitsZ) - 1u;
        return ((uint)c.x & maskX)
             | (((uint)c.y & maskY) << p.gridCoarseBitsX)
             | (((uint)c.z & maskZ) << (p.gridCoarseBitsX + p.gridCoarseBitsY));
    }

    // M6: coarse occupancy test — true when ALL 27 coarse slots around the
    // current block are CLEAR, i.e. no cell anywhere hashes into the blocks
    // that could contain a blocker of a ray segment inside the central block.
    // Safety proof (why the skip is exact, not approximate): a sphere the
    // ray crosses inside the central coarse block has its centre P within
    // rad < h of the crossing point X, so P's fine bin is within 1 fine bin
    // of X's bin; X's bin lies in the central block, hence P's BLOCK lies in
    // the 3×3×3 coarse neighbourhood of it. All 27 bits clear ⇒ no such P
    // ⇒ nothing can block the segment inside the central block.
    static bool coarseNeighbourhoodClear(int3 c, device const uint* occupancy,
                                         constant SimParams& p) {
        int3 cc = int3(floor(float3(c) * 0.125f));   // floor block coord (exact, see gridCount)
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint h = coarseHash(cc + int3(dx, dy, dz), p);
                    if (occupancy[h >> 5] & (1u << (h & 31u))) return false;
                }
            }
        }
        return true;
    }

    // Fractional light visibility (V2.1): the fraction of the light DISK
    // (angular radius lightSoft, radians — a physical soft light) visible
    // from ro along direction rd. Blockers are the SAME centerline-crossing
    // spheres the legacy binary march tested (dLine2 <= rad^2 — the DDA
    // candidate set and bin walk are UNCHANGED); each blocker occludes
    //   coverage = clamp((lightSoft + alpha - phi) / (2*lightSoft), 0, 1)
    // of the disk, where phi = angle(rd, blocker centre) and
    // alpha = asin(rad/d) is its angular radius as seen from ro (linear
    // ramp: full coverage once the blocker disk contains the light disk,
    // zero beyond its rim). The product over blockers assumes independent
    // occluders. lightSoft = 0 reduces coverage to the binary (phi < alpha)
    // verdict — bit-identical to the pre-V2.1 lightOccluded, including the
    // early exit at the same candidate the boolean `return true` fired at.
    // Penumbra approximation (documented): blockers that do NOT cross the
    // centerline (the penumbra cone skirt) are not in the DDA candidate set
    // and contribute nothing — the softness comes from the angular size of
    // centerline-crossing blockers, which carries nearly all of the
    // occlusion energy. A cone-widened march would need wider bin
    // neighbourhoods per DDA step; not worth it (measured in the V2.1
    // bench: the fraction cost is dominated by the LOST early exit, not by
    // the coverage arithmetic).
    //
    // M6 coarse occupancy skip unchanged (only ever skipped over regions
    // proven free of cells — verdict-identical for any lightSoft).
    // selfIdx2 (V2.1b): a SECOND cell index to skip in the blocker test —
    // the foam wall faces sit INSIDE two chambers' spheres (the wall disk is
    // shared), so the shadow march from a wall point must skip both cells or
    // every wall would read as fully self-shadowed. Callers without a second
    // self pass 0xFFFFFFFFu (never matches — behaviour bit-identical to the
    // pre-V2.1b single-self form; the growth illumination does exactly that).
    static float lightVisibility(float3 ro, float3 rd, float t0, float t1, uint selfIdx,
                                 uint selfIdx2,
                                 float lightSoft,
                                 device const float3* positions,
                                 device const float* radii,
                                 device const uint* binStarts,
                                 device const uint* particleIndices,
                                 device const uint* occupancy,
                                 constant SimParams& sp)
    {
        if (!(t1 > t0)) return 1.0f;
        const bool binary = (lightSoft <= 1e-6f);
        const float h = sp.roi;
        float vis = 1.0f;
        float3 p0 = ro + rd * t0;
        int3 c = int3((int)floor(p0.x / h), (int)floor(p0.y / h), (int)floor(p0.z / h));
        int3 st = int3(rd.x > 0.0f ? 1 : -1, rd.y > 0.0f ? 1 : -1, rd.z > 0.0f ? 1 : -1);
        // +inf where rd == 0: that axis never advances (IEEE division).
        float3 tDelta = fabs(float3(h) / rd);
        float3 tMax;
        tMax.x = rd.x != 0.0f ? t0 + ((rd.x > 0.0f ? (float)(c.x + 1) * h : (float)c.x * h) - p0.x) / rd.x : INFINITY;
        tMax.y = rd.y != 0.0f ? t0 + ((rd.y > 0.0f ? (float)(c.y + 1) * h : (float)c.y * h) - p0.y) / rd.y : INFINITY;
        tMax.z = rd.z != 0.0f ? t0 + ((rd.z > 0.0f ? (float)(c.z + 1) * h : (float)c.z * h) - p0.z) / rd.z : INFINITY;

        float tCur = t0;
        // Safety bound: identical rationale to traverseGrid (a hung kernel
        // stalls the whole machine; 8192 bins is orders above any reachable
        // ray length: t1 <= sceneR + |P| + 2*roi <= ~2 * max |P|).
        for (uint iter = 0; iter < 8192u && tCur < t1; ++iter) {
            // --- M6 coarse skip: jump over provably-empty 8³-bin blocks ---
            // (lightCoarseSkip is the A/B kill switch; the skip is only ever
            // taken over regions proven free of cells — see
            // coarseNeighbourhoodClear — so the verdict is identical either way)
            if (sp.lightCoarseSkip != 0u && coarseNeighbourhoodClear(c, occupancy, sp)) {
                int3 bc0 = int3(floor(float3(c) * 0.125f));   // current block (floored)
                float3 blockLo = float3(bc0 * 8) * h;
                float3 blockHi = float3((bc0 + int3(1)) * 8) * h;
                float tExitB = INFINITY;
                if (rd.x > 0.0f)      tExitB = min(tExitB, (blockHi.x - ro.x) / rd.x);
                else if (rd.x < 0.0f) tExitB = min(tExitB, (blockLo.x - ro.x) / rd.x);
                if (rd.y > 0.0f)      tExitB = min(tExitB, (blockHi.y - ro.y) / rd.y);
                else if (rd.y < 0.0f) tExitB = min(tExitB, (blockLo.y - ro.y) / rd.y);
                if (rd.z > 0.0f)      tExitB = min(tExitB, (blockHi.z - ro.z) / rd.z);
                else if (rd.z < 0.0f) tExitB = min(tExitB, (blockLo.z - ro.z) / rd.z);
                if (tExitB >= t1) return 1.0f;   // segment ends inside the clear block
                if (tExitB > tCur + 1e-4f) {
                    // Jump: reinitialize the DDA state at the block exit
                    // (same formulas as the loop head; tDelta/st unchanged).
                    tCur = tExitB;
                    p0 = ro + rd * tCur;
                    c = int3((int)floor(p0.x / h), (int)floor(p0.y / h), (int)floor(p0.z / h));
                    tMax.x = rd.x != 0.0f ? tCur + ((rd.x > 0.0f ? (float)(c.x + 1) * h : (float)c.x * h) - p0.x) / rd.x : INFINITY;
                    tMax.y = rd.y != 0.0f ? tCur + ((rd.y > 0.0f ? (float)(c.y + 1) * h : (float)c.y * h) - p0.y) / rd.y : INFINITY;
                    tMax.z = rd.z != 0.0f ? tCur + ((rd.z > 0.0f ? (float)(c.z + 1) * h : (float)c.z * h) - p0.z) / rd.z : INFINITY;
                    continue;
                }
                // Degenerate (exit ~ tCur, ray clipping the block corner):
                // fall through to the ordinary fine step.
            }
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        uint hh = binHash(c + int3(dx, dy, dz), sp);
                        uint s = binStarts[hh];
                        uint e = binStarts[hh + 1];
                        for (uint idx = s; idx < e; ++idx) {
                            uint r = particleIndices[idx];
                            if (r == selfIdx || r == selfIdx2) continue;
                            // Occlusion radius = the renderer's per-cell radius
                            // (mean link distance x radiusScale), CLAMPED below
                            // roi to stay inside the grid path's proof (radii
                            // >= roi exist only on stretched cells, typically
                            // << 1% of a grown state — they under-shadow
                            // slightly; documented in the M5 report).
                            float rad = min(radii[r], 0.99f * h);
                            if (rad <= 0.0f) continue;
                            float3 C = float3(positions[r]);
                            float3 toC = C - ro;
                            float tc = dot(toC, rd);
                            float dLine2 = dot(toC, toC) - tc * tc;
                            if (dLine2 > rad * rad) continue;
                            float thc = sqrt(max(rad * rad - dLine2, 0.0f));
                            float ta = tc - thc;
                            float tb = tc + thc;
                            if (tb > t0 && ta < t1) {
                                // Blocker crosses the segment: angular coverage
                                // of the light disk. d > rad here unless the
                                // start point itself lies INSIDE the blocker
                                // (tb > t0 with the centre behind ro forces
                                // d < rad) — full coverage either way.
                                float d = sqrt(dLine2 + tc * tc);
                                float coverage;
                                if (binary || d <= rad) {
                                    coverage = 1.0f;
                                } else {
                                    float phi = acos(clamp(tc / d, -1.0f, 1.0f));
                                    float alpha = asin(min(rad / d, 1.0f));
                                    coverage = clamp((lightSoft + alpha - phi) / (2.0f * lightSoft), 0.0f, 1.0f);
                                }
                                vis *= 1.0f - coverage;
                                if (vis < 1e-3f) return 0.0f;   // fully blocked; stop marching
                            }
                        }
                    }
                }
            }
            if (tCur >= t1) break;
            // Advance to the next bin (ties broken consistently, as in the
            // renderer: ranges stay a partition of [t0, t1] either way).
            float tExit = min(min(tMax.x, tMax.y), tMax.z);
            if (tExit >= t1) break;
            if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
                c.x += st.x; tCur = tMax.x; tMax.x += tDelta.x;
            } else if (tMax.y <= tMax.z) {
                c.y += st.y; tCur = tMax.y; tMax.y += tDelta.y;
            } else {
                c.z += st.z; tCur = tMax.z; tMax.z += tDelta.z;
            }
        }
        return clamp(vis, 0.0f, 1.0f);
    }

    // ---- V2.1b: foam (power-diagram chamber) helpers -----------------------
    // Shared by the impostor fragment (renderKernels cellFragment) and the
    // ray tracer's foam paths (raytraceKernels) — single-sourced here.
    //
    // GEOMETRY. A cell's rendered volume is its POWER-DIAGRAM CHAMBER:
    //   chamber(i) = sphere(c_i, r_i) ∩ ⋂_{j ∈ ring(i)} {x : |x−c_i|² − r_i²
    //                                                    ≤ |x−c_j|² − r_j²}
    // i.e. the sphere truncated by the POWER-BISECTOR half-spaces toward its
    // ring neighbours. Additively weighted (power) bisectors are PLANES, so
    // the walls between pressed cells are truly flat and sit FAIRLY between
    // unequal cells (unlike the chord-midpoint plane of plain Voronoi). With
    // c_i at the origin and D = c_j − c_i the bisector is
    //   x·D = (|c_j|² − |c_i|² + r_i² − r_j²)/2,
    // a plane with normal D — in the canonical form below, normalised from
    // the LOWER-indexed cell so the two chambers sharing a wall compute
    // bit-identical plane values (the raster depth test then sees identical
    // depths — no z-fighting, per the milestone contract).
    //
    // VISIBILITY FACT (drives the whole foam design): the union of chambers
    // equals the union of spheres EXACTLY — any point x in some sphere has a
    // global power-minimiser k with |x−c_k|² − r_k² < 0, hence x ∈ sphere_k,
    // and being the minimiser x also satisfies every ring half-space of k, so
    // x ∈ chamber(k); ring-only truncation loses nothing. Therefore the
    // OPAQUE nearest surface of the chamber union is always a SPHERE CAP —
    // walls are strictly interior interfaces (only measure-zero wall-rim rays
    // hit a wall first). Foam therefore renders TRANSLUCENTLY: the chamber
    // ENTRY faces along the ray are composited front-to-back with a per-face
    // opacity (foamOpacity), so the flat walls show through the curved front
    // caps — "walls behind walls", the soap-froth depth. Every wall crossed
    // by a ray is provably the ENTRY of the following chamber: the wall point
    // lies on the shared disk (plane ∩ sphere_i = plane ∩ sphere_j, equal by
    // the bisector identity), so the ray enters sphere_j before/at the wall
    // and the bisector constraint raises j's entry bound exactly to the wall
    // (the plane's signed distance along the ray is monotone, so the entry is
    // raised at most once per plane). Chambers tile the union, entries are
    // strictly ordered along the ray, and gaps between disconnected blobs are
    // simply transparent.
    //
    // Radius-scale contract: a wall has REAL area only when the two spheres
    // overlap; at radiusScale 0.5 (mean link distance ~1) neighbours merely
    // touch and walls degenerate to points — foam mode wants radiusScale 0.7
    // (wall-disk radius sqrt(r² − h²) ≈ 0.49 r at equal cells, d = 2h).

    // Clip cell `r`'s ray-sphere interval [ta, tb] (from centre C, radius
    // rad, tc = closest-approach parameter, dLine2 = squared centre-line
    // distance — all precomputed by the caller) against the POWER-BISECTOR
    // half-spaces toward its ring neighbours. Returns false when the chamber
    // is empty along the ray. Outputs: e/x = the clipped interval, wallJ =
    // 0xFFFFFFFF for a SPHERE (cap) entry, else the ring neighbour whose
    // bisector produced the entry bound (a flat WALL entry; the plane is
    // recomputed for shading via foamWallNormal — canonical, see above).
    static bool foamChamberInterval(float3 ro, float3 rd, float3 C, float rad,
                                    float tc, float dLine2, uint r,
                                    device const float3* positions,
                                    device const float* radii,
                                    device const uint* links,
                                    device const uint* linkCounts,
                                    uint maxValence,
                                    thread float* outE, thread float* outX,
                                    thread uint* outWallJ)
    {
        float thc = sqrt(max(rad * rad - dLine2, 0.0f));
        float e = tc - thc;
        float x = tc + thc;
        uint wallJ = 0xFFFFFFFFu;
        uint n = linkCounts[r];
        uint base = r * maxValence;
        for (uint k = 0; k < n; ++k) {
            uint j = links[base + k];
            if (j == r) continue;
            // Canonical plane of the (ia, ib) pair, ia = the lower index:
            // unit normal nh (ia -> ib) and offset hh with
            //   chamber(ia): x·nh <= hh      chamber(ib): x·nh >= hh
            uint ia = min(r, j), ib = max(r, j);
            float3 ca = float3(positions[ia]);
            float3 cb = float3(positions[ib]);
            float3 D = cb - ca;
            float d2 = dot(D, D);
            if (d2 < 1e-12f) continue;          // coincident centres: no bisector
            float invD = rsqrt(d2);
            float3 nh = D * invD;
            float hh = (dot(cb, cb) - dot(ca, ca)
                        + radii[ia] * radii[ia] - radii[ib] * radii[ib]) * (0.5f * invD);
            // Signed distance of the ray against the plane, f(t) = rn + t·dn
            // (monotone in t): clip the interval, keep the chamber's near side.
            float rn = dot(ro, nh);
            float dn = dot(rd, nh);
            if (r == ia) {
                if (dn > 1e-9f) {
                    float ub = (hh - rn) / dn;
                    if (ub < x) x = ub;
                } else if (dn < -1e-9f) {
                    float lb = (hh - rn) / dn;
                    if (lb > e) { e = lb; wallJ = j; }
                } else if (rn > hh) {
                    return false;               // ray parallel, outside the chamber
                }
            } else {
                if (dn < -1e-9f) {
                    float ub = (hh - rn) / dn;
                    if (ub < x) x = ub;
                } else if (dn > 1e-9f) {
                    float lb = (hh - rn) / dn;
                    if (lb > e) { e = lb; wallJ = j; }
                } else if (rn < hh) {
                    return false;
                }
            }
            if (e > x) return false;            // chamber empty along the ray
        }
        if (!(e < x)) return false;             // degenerate (measure-zero) contact
        *outE = e;
        *outX = x;
        *outWallJ = wallJ;
        return true;
    }

    // Canonical unit normal of the wall shared by cells r and j (from the
    // lower to the higher index — both sides of the wall derive the SAME
    // vector; the caller flips it toward the camera for shading).
    static inline float3 foamWallNormal(uint r, uint j,
                                        device const float3* positions) {
        uint ia = min(r, j), ib = max(r, j);
        float3 D = float3(positions[ib]) - float3(positions[ia]);
        return D * rsqrt(max(dot(D, D), 1e-20f));
    }
    """

    // MARK: - `simKernels`: simulation kernels (M1 forces; M2 food + division; M3 grid; M5 chemicals; V2.2 tint/pulse)

    static let simKernels = """

    // Buffer table:
    //   0 positions (const)    1 links (const)      2 linkCounts (const)
    //   3 normals (write)      4 cellCounter        5 params
    kernel void computeNormals(device const float3* positions [[buffer(0)]],
                               device const uint*   links      [[buffer(1)]],
                               device const uint*   linkCounts [[buffer(2)]],
                               device float3* normals           [[buffer(3)]],
                               device const uint*   cellCounter [[buffer(4)]],
                               constant SimParams&  params      [[buffer(5)]],
                               uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float3 P = float3(positions[gid]);
        uint   n = linkCounts[gid];
        uint base = gid * params.maxValence;
        float3 acc = float3(0.0);
        if (n >= 3) {
            float3 prev = float3(positions[links[base + n - 1]]) - P;
            for (uint k = 0; k < n; ++k) {
                float3 cur = float3(positions[links[base + k]]) - P;
                acc += cross(prev, cur);
                prev = cur;
            }
        }
        float len2 = dot(acc, acc);
        // Fallback for a degenerate ring; cannot happen on a healthy closed manifold.
        normals[gid] = float3(len2 > 1e-20f ? acc * rsqrt(len2) : float3(0.0, 1.0, 0.0));
    }

    // Pass 1: count particles per hashed bin. M6: also SET the coarse
    // occupancy bit for the cell's 8³-fine-bin coordinate block (the light
    // march's empty-interior skip map). The bitfield is blit-cleared at the
    // start of this rebuild and must stay LIVE through the post-scatter
    // illumination march. A bit is SET iff any cell's block-coordinate hashes
    // to that slot — collisions are harmless (a spuriously set bit only
    // disables the skip, never mis-skips).
    //
    // Buffer table:
    //   0 positions (const)    1 binCounts (atomic add)   2 occupancy (atomic or)
    //   3 cellCounter   4 params
    kernel void gridCount(device const float3* positions [[buffer(0)]],
                          device atomic_uint* binCounts  [[buffer(1)]],
                          device atomic_uint* occupancy  [[buffer(2)]],
                          device const uint*  cellCounter [[buffer(3)]],
                          constant SimParams& params      [[buffer(4)]],
                          uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        int3 bc = int3((int)floor(positions[gid].x / params.roi),
                       (int)floor(positions[gid].y / params.roi),
                       (int)floor(positions[gid].z / params.roi));
        atomic_fetch_add_explicit(&binCounts[binHash(bc, params)], 1u, memory_order_relaxed);
        // Coarse block coordinate = fine bin coordinate >> 3, FLOORED (the
        // block of bin c is [floor(c/8)*8, floor(c/8)*8+8)); c can be
        // negative, so the shift must be arithmetic — computed exactly via
        // the power-of-two float multiply + floor instead (c*0.125 is exact:
        // |c| < 2^24).
        int3 cc = int3(floor(float3(bc) * 0.125f));
        uint ch = coarseHash(cc, params);
        atomic_fetch_or_explicit(&occupancy[ch >> 5], 1u << (ch & 31u), memory_order_relaxed);
    }

    // Pass 2a: exclusive prefix scan, level 1. Scans the T bin counts into
    // binStarts (LOCAL exclusive values; binStarts[T] = grand total is written
    // by gridScanBlocks, so every query can read the candidate range
    // [binStarts[h], binStarts[h+1])). Threadgroup size 64, consistent with
    // the rest of the pipeline: one bin per thread, local exclusive scan via
    // inclusive Hillis-Steele in shared memory, then the threadgroup total
    // goes to blockSums[tg] (level 2 scans that).
    // NOTE: T = 2^k (k >= 8) is always a multiple of 64, so the T bins fill
    // whole threadgroups exactly; a "virtual entry T" inside the scan would
    // otherwise sit in a threadgroup of its own with block sum 0 and break
    // the last bin's range (found in M3 lockstep testing).
    //
    // Buffer table:
    //   0 binCounts (const)   1 binStarts (write)   2 blockSums (write)   3 params
    kernel void gridScanLocal(device const uint* binCounts [[buffer(0)]],
                              device uint*   binStarts     [[buffer(1)]],
                              device uint*   blockSums     [[buffer(2)]],
                              constant SimParams& params   [[buffer(3)]],
                              uint gid  [[thread_position_in_grid]],
                              uint ltid [[thread_position_in_threadgroup]],
                              uint tg   [[threadgroup_position_in_grid]])
    {
        uint T = (1u << params.gridBitsX) * (1u << params.gridBitsY) * (1u << params.gridBitsZ);
        threadgroup uint shared[64];
        uint count = (gid < T) ? binCounts[gid] : 0u;
        shared[ltid] = count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1; off < 64; off <<= 1) {
            uint v = (ltid >= off) ? shared[ltid - off] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            shared[ltid] += v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // shared[ltid] = inclusive prefix within this threadgroup.
        if (gid < T) binStarts[gid] = shared[ltid] - count;   // local exclusive
        if (ltid == 63u) blockSums[tg] = shared[63u];
    }

    // Pass 2b: exclusive prefix scan, level 2. Exactly ONE threadgroup
    // (dispatchThreadgroups(1, 64) — enforced by the encoder) exclusively
    // scans blockSums (B = T/64 entries) in place, each thread walking a
    // contiguous chunk, and thread 63 stores the grand total to binStarts[T]
    // so the last bin's query range [binStarts[T-1], binStarts[T]) is
    // complete. Worst case B = 65,536 at T = 2^22 (chunk 1024) — a single
    // threadgroup is the task-mandated two-level shape; the cost stays a
    // small fraction of a step (benchlog M3).
    // ROOT CAUSE (M3 3x bug): this kernel was originally dispatched with
    // ceil(T/64) threadgroups while being a single-threadgroup algorithm —
    // ceil(T/64) racing copies read/wrote the same blockSums entries. The
    // race collapsed the blockSums fold to equal values, and because the
    // packed hash places the z bits ABOVE the 6-bit intra-threadgroup offset
    // (bx+by = 6 at subdiv 3), the 27 query bins then shared only 9 distinct
    // slot ranges (one per (x,y) offset), each summed once per dz => the
    // collision field came out EXACTLY 3.000x the correct value.
    // B is derived here from T (T is a power of two >= 2^8, so T/64 is exact)
    // rather than from threadgroups_per_grid, which is 1 for the mandated
    // single-threadgroup dispatch.
    //
    // Buffer table:
    //   0 blockSums (read + write)   1 binStarts (write: entry T only)   2 params
    kernel void gridScanBlocks(device uint* blockSums     [[buffer(0)]],
                               device uint*   binStarts   [[buffer(1)]],
                               constant SimParams& params [[buffer(2)]],
                               uint ltid [[thread_position_in_threadgroup]])
    {
        uint T = (1u << params.gridBitsX) * (1u << params.gridBitsY) * (1u << params.gridBitsZ);
        uint B = T >> 6;            // block sums = one per scanLocal threadgroup
        uint chunk = (B + 63u) / 64u;
        uint running = 0u;
        for (uint k = 0; k < chunk; ++k) {
            uint idx = ltid * chunk + k;
            if (idx < B) {
                uint v = blockSums[idx];
                blockSums[idx] = running;   // exclusive, in place
                running += v;
            }
        }
        threadgroup uint shared[64];
        shared[ltid] = running;   // this thread's chunk total
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1; off < 64; off <<= 1) {
            uint v = (ltid >= off) ? shared[ltid - off] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            shared[ltid] += v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // shared[ltid] = inclusive prefix of the chunk totals; fold the
        // exclusive offset back into this thread's chunk.
        uint myOff = (ltid > 0u) ? shared[ltid - 1u] : 0u;
        for (uint k = 0; k < chunk; ++k) {
            uint idx = ltid * chunk + k;
            if (idx < B) blockSums[idx] += myOff;
        }
        // Grand total = inclusive prefix over ALL chunk totals; binStarts[T]
        // ends the last bin's range. (shared[63] is this thread's own value:
        // no cross-thread read, no extra barrier needed.)
        if (ltid == 63u) {
            uint T = (1u << params.gridBitsX) * (1u << params.gridBitsY) * (1u << params.gridBitsZ);
            binStarts[T] = shared[63u];
        }
    }

    // Pass 2c: fold the (now exclusive) block sums back into binStarts, which
    // becomes the global exclusive prefix over the T bins (entry T = grand
    // total, written by gridScanBlocks).
    //
    // Buffer table:
    //   0 binStarts (read + write)   1 blockSums (const)   2 params
    kernel void gridScanOffset(device uint* binStarts       [[buffer(0)]],
                               device const uint* blockSums [[buffer(1)]],
                               constant SimParams& params   [[buffer(2)]],
                               uint gid [[thread_position_in_grid]])
    {
        uint T = (1u << params.gridBitsX) * (1u << params.gridBitsY) * (1u << params.gridBitsZ);
        if (gid >= T) return;
        binStarts[gid] += blockSums[gid >> 6];
    }

    // Pass 3: scatter particle indices into their bin's slice. binCounts[h]
    // is the fill cursor: atomic decrement hands out slots binStarts[h] ..
    // binStarts[h]+count-1 and leaves the counter at 0 (self-cleaning).
    // Within-bin order is hardware-scheduled (atomic order): the repulsion
    // sum of a query cell is therefore not bitwise fixed run-to-run in
    // principle; on a fixed device/driver it is stable in practice, and the
    // golden-image check is the canary for that assumption. Cross-
    // implementation equality is handled by the --lockstep tolerances.
    //
    // Buffer table:
    //   0 positions (const)            1 binCounts (atomic sub)
    //   2 binStarts (const)            3 particleIndices (write)
    //   4 cellCounter                  5 params
    kernel void gridScatter(device const float3* positions [[buffer(0)]],
                            device atomic_uint* binCounts  [[buffer(1)]],
                            device const uint*  binStarts  [[buffer(2)]],
                            device uint*        particleIndices [[buffer(3)]],
                            device const uint*  cellCounter [[buffer(4)]],
                            constant SimParams& params      [[buffer(5)]],
                            uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        uint h = binHashOfPos(float3(positions[gid]), params);
        uint slot = atomic_fetch_sub_explicit(&binCounts[h], 1u, memory_order_relaxed) - 1u;
        particleIndices[binStarts[h] + slot] = gid;
    }

    // Paper §4, collisionOffset:
    //   collisionOffset = repulsionStrength * sum_{r in A}
    //                       ((roi^2 - |P - P_r|^2)/roi^2) * hat(P - P_r)
    // A = all particles within roi that are NOT directly linked (linked pairs
    // are handled by spring/planar/bulge). hat(P - P_r) = (P - P_r)/|P - P_r|
    // points AWAY from the neighbour (repulsive); the weight falls linearly
    // (in squared distance) from 1 at contact to 0 at roi. integrateForces
    // ADDS this to the position (paper: "a collisionOffset vector to be added
    // to each particle's position") — an additive over-damped offset, no
    // momentum (AGENTS hard rule 5).
    //
    // Query: walk the 27 bins around the cell's own; the packed hash makes
    // the 27 lookups distinct (header comment above). Candidates are filtered
    // by distance <= roi and by the not-linked test against the cell's own
    // ring (<= maxValence entries, loaded once up front). Degenerate case
    // d ~ 0 (exact overlap): hat() is undefined — push along the cell's own
    // normal instead (deterministic; the weight is ~1 there, so the pair
    // separates immediately).
    //
    // Separate accumulate pass (rather than fusing into integrateForces):
    // (a) the collision field can be read back and compared cell-by-cell
    //     against the CPU brute-force reference in --lockstep, isolating grid
    //     correctness from feedback amplification; (b) keeps register pressure
    //     in the hot integrate kernel low. Cost: one extra full-size float3
    //     buffer + one extra pass (measured in benchlog M3).
    //
    // Buffer table:
    //   0 positions (const)       1 normals (const)      2 links (const)
    //   3 linkCounts (const)      4 binStarts (const)    5 particleIndices (const)
    //   6 collisionOffset (write) 7 cellCounter          8 params
    kernel void repulsionAccumulate(device const float3* positions [[buffer(0)]],
                                    device const float3* normals   [[buffer(1)]],
                                    device const uint*   links     [[buffer(2)]],
                                    device const uint*   linkCounts [[buffer(3)]],
                                    device const uint*   binStarts [[buffer(4)]],
                                    device const uint*   particleIndices [[buffer(5)]],
                                    device float3*       collisionOffset [[buffer(6)]],
                                    device const uint*   cellCounter [[buffer(7)]],
                                    constant SimParams&  params      [[buffer(8)]],
                                    uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float3 P = float3(positions[gid]);
        float3 N = float3(normals[gid]);
        uint   n = linkCounts[gid];
        uint   base = gid * params.maxValence;

        uint ring[kMaxValence];
        for (uint k = 0; k < n; ++k) ring[k] = links[base + k];

        float  roi  = params.roi;
        float  roi2 = roi * roi;
        float  invRoi2 = 1.0f / roi2;
        int3   bc = int3((int)floor(P.x / roi), (int)floor(P.y / roi), (int)floor(P.z / roi));

        float3 offset = float3(0.0);
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint h = binHash(bc + int3(dx, dy, dz), params);
                    uint s = binStarts[h];
                    uint e = binStarts[h + 1];
                    for (uint idx = s; idx < e; ++idx) {
                        uint r = particleIndices[idx];
                        if (r == gid) continue;
                        float3 D = P - float3(positions[r]);
                        float  d2 = dot(D, D);
                        if (d2 > roi2) continue;
                        bool linked = false;
                        for (uint k = 0; k < n; ++k) {
                            if (ring[k] == r) { linked = true; break; }
                        }
                        if (linked) continue;
                        float d = sqrt(d2);
                        float3 dir = d > 1e-6f ? D / d : N;
                        offset += ((roi2 - d2) * invRoi2) * dir;
                    }
                }
            }
        }
        collisionOffset[gid] = params.repulsionStrength * offset;
    }

    // Paper §4 "Mathematical Model":
    //   springTarget = 1/n * sum_r ( L_r + linkRestLength * hat(P - L_r) )
    //   planarTarget = 1/n * sum_r L_r
    //   dotN_r       = (L_r - P) . N
    //   bulgeDist    = 1/n * sum_r ( sqrt(linkRestLength^2 - |L_r - P|^2 + dotN_r^2) + dotN_r )
    //                  (the paper prints |L_r|^2 — a typo; see docs/reference, "Formula verification")
    //   bulgeTarget  = P + bulgeDist * N
    //   P'           = P + springFactor*(springTarget - P)
    //                    + planarFactor*(planarTarget - P)
    //                    + bulgeFactor*(bulgeTarget - P)
    // No momentum: over-damped relaxation (viscous medium), stable against the
    // energy injected by cell division.
    //
    // M3: the paper's collisionOffset (repulsionAccumulate, computed from the
    // SAME frozen pre-step state) is added to the position here. The guard
    // keeps behaviour identical when repulsion is disabled — the host then
    // skips the grid kernels entirely and the buffer holds stale data.
    //
    // Buffer table:
    //   0 positions (const)    1 normals (const)    2 links (const)
    //   3 linkCounts (const)   4 newPositions (write)   5 cellCounter   6 params
    //   7 collisionOffset (const, M3)
    kernel void integrateForces(device const float3* positions    [[buffer(0)]],
                                device const float3* normals      [[buffer(1)]],
                                device const uint*   links               [[buffer(2)]],
                                device const uint*   linkCounts          [[buffer(3)]],
                                device float3* newPositions       [[buffer(4)]],
                                device const uint*   cellCounter         [[buffer(5)]],
                                constant SimParams&  params              [[buffer(6)]],
                                device const float3* collisionOffset [[buffer(7)]],
                                uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float3 P = float3(positions[gid]);
        float3 N = float3(normals[gid]);
        uint   n = linkCounts[gid];
        uint base = gid * params.maxValence;

        float  L  = params.linkRestLength;
        float  L2 = L * L;
        float3 springTarget = float3(0.0);
        float3 planarTarget = float3(0.0);
        float  bulgeSum     = 0.0;

        for (uint k = 0; k < n; ++k) {
            float3 Lp  = float3(positions[links[base + k]]);
            float3 toP = P - Lp;
            float  d   = length(toP);
            // (1) spring: push to rest length along current link direction
            springTarget += Lp + (d > 1e-9f ? toP * (L / d) : float3(0.0));
            // (2) planar: pull toward neighbour centroid (torsion-spring-like)
            planarTarget += Lp;
            // (3) bulge: outward push along N restoring a compressed link to rest length
            float3 lv   = Lp - P;
            float  dotN = dot(lv, N);
            float  t2   = dot(lv, lv) - dotN * dotN;   // squared tangential length
            float  push = sqrt(max(L2 - t2, 0.0f));    // radicand < 0 => link stretched, no push
            // (3) bulge: outward push along N restoring a compressed link to rest
            // length. Per-link x_r = dotN + sqrt(L^2 - t^2); the sqrt is
            // domain-clamped (radicand < 0 => link stretched, no root), but the
            // SUM TERM KEEPS ITS SIGN: a stretched link then contributes
            // dotN < 0, i.e. an inward pull. Clamping the per-link value at 0
            // (the M1 choice) leaves the sheet with no scale-restoring force:
            // division pumps compression in, the sheet relaxes by expanding,
            // and the expansion never reverses — an inflation ratchet (seen in
            // M2 testing: mean link length drifting to ~1.4 and staying there
            // even after growth stops). The paper's formula is unclamped; the
            // rest state (links at rest length on a convex patch) still gives
            // x_r = |dotN| + dotN = 0, so the resting ball is unchanged.
            bulgeSum += push + dotN;
        }

        float invN = n > 0 ? 1.0f / float(n) : 0.0f;
        springTarget *= invN;
        planarTarget *= invN;
        float  bulgeDist = bulgeSum * invN;

        float3 Pn = P
            + params.springFactor * (springTarget - P)
            + params.planarFactor * (planarTarget - P)
            + params.bulgeFactor  * (bulgeDist * N);

        // M3 repulsion: additive over-damped offset (no momentum), computed
        // from the same frozen pre-step state as the three targets above.
        if (params.repulsionStrength > 0.0f) {
            Pn += float3(collisionOffset[gid]);
        }

        newPositions[gid] = float3(Pn);
    }

    // Paper §4: "Uniformly adding a random amount to each cell at each time
    // step." Each cell owns a PCG32-style RNG stream (fixed seed). Runs when
    // division is enabled (food without division has no effect).
    //
    // M5 growth-mode coupling (paper §4 methods 2 and 3): the per-step food
    // increment is scaled by a per-cell drive factor.
    //   uniform (0): drive = 1                       — bit-identical to M2-M4
    //   light   (1): drive = lightBase + nutrient    — nutrient from illuminate
    //   rde     (2): drive = rdeBase + rdeGain * v   — v from rdeStep
    // The RNG stream is consumed EXACTLY once per cell in every mode (one
    // rand01 call), so uniform-mode runs stay bit-identical to the pre-M5
    // semantics and the per-cell RNG streams stay in lockstep across modes.
    //
    // V2.3 type-dependent increment multiplier (rate), multiplied in AFTER
    // drive and BEFORE rand01 — for rate == 1.0f the product is BIT-IDENTICAL
    // to the pre-V2.3 expression (x*1.0f == x exactly), which is the type-0
    // contract: untyped runs (all cells type 0) reproduce the historical
    // semantics byte-for-byte. The rate is the phase-2 half of the rule
    // matrix (the decide-pass gate is the other half — see divisionDecide):
    //   phase 1 (divCounter < phaseSwitchX, or phaseSwitchX == 0):
    //     type 0: 1.0 (contract)    type 1 stem: 1.0 (gate lives in decide)
    //     type 2/3 voeg: 0 — dormant (their food stays at the RESET value,
    //                     uniform [0, splitThreshold): at the switch the top
    //                     of that distribution is already near the threshold,
    //                     so phase 2 starts promptly — no synchronized burst
    //                     though: the ring-distance-2 independent set and the
    //                     spread of reset foods spread the first wave out)
    //     type 4 cap: 1.0
    //   phase 2 (divCounter >= phaseSwitchX > 0): ONLY voeg cells divide, so
    //     stem/cap get 0; a voeg cell's rate depends on its LINK CLASSES
    //     (edge = a link to a stem/cap cell; interior = a link to the
    //     COMPLEMENTARY voeg type):
    //     lamel: edge -> lamelRate; lamel<->dal interior -> interiorRate;
    //            lamel<->lamel only -> 0 (never)
    //     dal:   edge -> dalRate;   lamel<->dal interior -> interiorRate;
    //            dal<->dal only -> 0 (never)
    //     interiorRate = lamelRate / max(interiorDivider, 1) (default /10).
    // V2.4 REGION mode (typeMode 1, RDE-typed runs): the band matrix above is
    // NOT applied — every typed region grows at rate 1.0 (uniform within its
    // region). The patterned growth rate is carried entirely by the DRIVE
    // (rdeBase + rdeGain·v): the differentiation phase froze each typed
    // cell's v, so cap regions (v ≈ 0.35+) grow several times faster per
    // capita than stem regions (v ≈ 0) — regional rate differences from the
    // chemical map, with the decide-pass gate reduced to "everything
    // eligible" (see typeEligible). The two-phase machinery is band-specific
    // and is not applied in region mode. Type 0 cells keep rate 1.0 in both
    // modes (the type-0 contract).
    // The RNG stream is still consumed EXACTLY once per cell at rate 0 (the
    // product is simply 0) — stream parity across configs is preserved.
    //
    // Buffer table:
    //   0 rngState (read+write)   1 food (write)   2 cellCounter   3 params
    //   4 nutrient (const, current buffer; light mode)
    //   5 chemU (const, current buffer; rde mode)   6 chemV (const; rde mode)
    //   7 cellType (const; V2.3 rate matrix)
    //   8 divCounter (const; V2.3 phase clock — read-only here; the increment
    //     happens in divisionExecute LATER in the same command buffer, so
    //     every foodStep thread of a step sees the same pre-step phase)
    //   9 links (const; V2.3 edge/interior link classes)
    //  10 linkCounts (const; V2.3)
    kernel void foodStep(device uint64_t* rngState    [[buffer(0)]],
                         device float*   food         [[buffer(1)]],
                         device const uint* cellCounter [[buffer(2)]],
                         constant SimParams& params     [[buffer(3)]],
                         device const float* nutrient   [[buffer(4)]],
                         device const float* chemU      [[buffer(5)]],
                         device const float* chemV      [[buffer(6)]],
                         device const uint*  cellType   [[buffer(7)]],
                         device const uint*  divCounter [[buffer(8)]],
                         device const uint*  links      [[buffer(9)]],
                         device const uint*  linkCounts [[buffer(10)]],
                         uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float drive = 1.0f;
        if (params.growthMode == 1u) {
            drive = params.lightBase + nutrient[gid];
        } else if (params.growthMode == 2u) {
            drive = params.rdeBase + params.rdeGain * chemV[gid];
        }
        // V2.3 type rate (1.0 for type 0 — the no-op contract; see above).
        float rate = 1.0f;
        uint type = cellType[gid];
        if (type != 0u) {
            if (params.typeMode == 1u) {
                // V2.4 region mode: uniform within the region (see above).
                rate = 1.0f;
            } else {
            const bool phase2 = (params.phaseSwitchX != 0u)
                             && (divCounter[0] >= params.phaseSwitchX);
            if (!phase2) {
                if (type == 2u || type == 3u) rate = 0.0f;   // voeg dormant
            } else if (type == 1u || type == 4u) {
                rate = 0.0f;                                 // only voeg cells grow
            } else {
                // Voeg cell in phase 2: classify the ring links (frozen data).
                uint n = linkCounts[gid];
                uint base = gid * params.maxValence;
                bool edge = false;       // neighbour = stem (1) or cap (4)
                bool interior = false;   // neighbour = the complementary voeg type
                for (uint k = 0; k < n; ++k) {
                    uint tj = cellType[links[base + k]];
                    if (tj == 1u || tj == 4u) edge = true;
                    else if (type == 2u ? tj == 3u : tj == 2u) interior = true;
                }
                const float interiorRate = params.lamelRate
                                         / max(params.interiorDivider, 1.0f);
                rate = edge ? (type == 2u ? params.lamelRate : params.dalRate)
                            : (interior ? interiorRate : 0.0f);
            }
            }
        }
        food[gid] += params.foodIncrement * drive * rate * rand01(&rngState[gid]);
    }

    // ---- M5: growth-mode chemicals (paper §4 methods 2 and 3) --------------
    //
    // Light mode (method 3): a fixed directional light; a cell is LIT when no
    // other cell's sphere blocks the straight segment from the cell toward the
    // light. Lit cells produce nutrient (lightNutrient per step); nutrient
    // diffuses along the links (nutrientDiffuse) and decays first-order
    // (lightDecay). The food increment is proportional to (lightBase +
    // nutrient) — see foodStep.
    //
    // RDE mode (method 2): Gray-Scott à la Turk on the cell graph (rdeStep);
    // the v chemical drives the food increment.
    //
    // Nutrient dynamics design note (paper is silent on the nutrient time
    // course — it only says light "stimulates nutrient creation" and the
    // nutrient "can diffuse"): pure accumulation is unbounded, and consuming
    // nutrient at division drives every cell's long-run growth rate toward
    // equality (division rate is the only sink, a negative feedback that
    // erases the lit/dark gradient). A first-order decay keeps a STEADY
    // gradient: lit steady state ~ lightNutrient/lightDecay + diffusive
    // inflow, dark steady state ~ inflow/lightDecay — the lit/dark ratio is
    // controlled by lightDecay vs lightDiffusionRate, which is exactly the
    // paper's low-diffusion (plant-like) vs high-diffusion (organ-like) knob.

    // Numerical clamp for the nutrient level (far above the operating range
    // lightNutrient/lightDecay ~ O(1); purely guards pathological params).
    constant float kNutrientMax = 64.0f;

    // Per-step scene bound for the light march: max |P|^2 over the CURRENT
    // positions, atomically maxed as the uint bit pattern (monotonic for
    // non-negative floats). illuminate reads it to bound its rays; clearBound
    // zeroes the accumulator again (same command buffer, encoder order fixes
    // visibility). This avoids any host readback or host-side param update
    // (which would be a mid-flight-params hazard in the GUI).
    //
    // Buffer table:
    //   0 positions (const)   1 bound2 (atomic max)   2 cellCounter   3 params
    kernel void boundMax(device const float3* positions [[buffer(0)]],
                         device atomic_uint* bound2     [[buffer(1)]],
                         device const uint*  cellCounter [[buffer(2)]],
                         constant SimParams& params     [[buffer(3)]],
                         uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float3 P = float3(positions[gid]);
        atomic_fetch_max_explicit(&bound2[0], as_type<uint>(dot(P, P)), memory_order_relaxed);
    }

    // Buffer table:
    //   0 bound2 (write 0)   — exactly one thread
    kernel void clearBound(device atomic_uint* bound2 [[buffer(0)]],
                           uint gid [[thread_position_in_grid]])
    {
        if (gid != 0u) return;
        atomic_store_explicit(&bound2[0], 0u, memory_order_relaxed);
    }

    // One thread per cell: illumination march + nutrient production. Runs on
    // the FROZEN pre-step state (positions + radii + the grid built from them
    // this step), before integrateForces — the march must match the grid
    // exactly, and the nutrient is consumed later in the same step by
    // foodStep (post-diffuse buffer). SCHEDULED every lightInterval steps
    // (host-side; production is pulsed lightInterval× so the average
    // production rate — and hence the nutrient steady state — is unchanged);
    // decay + diffusion run EVERY step in nutrientDiffuse.
    // M5 unified light: the per-cell illumination verdict is also STORED —
    // V2.1: the FRACTIONAL visible fraction of the soft light disk
    // (lightVisibility; 0 = fully shaded, 1 = fully lit). Consumers: the
    // growth coupling (smoother lit/dark gradient than the binary verdict),
    // the impostor fragment's per-cell shadow factor, and the ray tracer's
    // directional solid style (per-pixel shadow rays refine it).
    // (CellSim.encodeIlluminationRefresh recomputes it for a loaded state.)
    // Production scales with the fraction: nutrient += lightNutrient *
    // lightInterval * vis — the binary verdict is the lightSoft = 0 special
    // case (vis in {0, 1}).
    //
    // Buffer table:
    //   0 positions (const)       1 radii (const)        2 binStarts (const)
    //   3 particleIndices (const) 4 nutrient (read+write) 5 bound2 (const)
    //   6 cellCounter             7 params               8 litFraction (write)
    //   9 occupancy (const; M6 coarse skip map — written by gridCount in the
    //     SAME rebuild this step, cleared by the blit fill before it)
    kernel void illuminate(device const float3* positions [[buffer(0)]],
                           device const float*   radii    [[buffer(1)]],
                           device const uint*    binStarts [[buffer(2)]],
                           device const uint*    particleIndices [[buffer(3)]],
                           device float*         nutrient [[buffer(4)]],
                           device const uint*    bound2   [[buffer(5)]],
                           device const uint*    cellCounter [[buffer(6)]],
                           constant SimParams&   params   [[buffer(7)]],
                           device float*         litFraction [[buffer(8)]],
                           device const uint*    occupancy [[buffer(9)]],
                           uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float n = nutrient[gid];

        float3 P = float3(positions[gid]);
        float3 L = normalize(float3(params.lightDirX, params.lightDirY, params.lightDirZ));
        // Ray bound: every blocking cell centre lies within sceneR of the
        // origin, so the farthest possible hit parameter is
        // |P| + sceneR + rad <= |P| + sceneR + roi (see the M5 report); 2*roi
        // adds slack for the radius clamp.
        float sceneR = sqrt(as_type<float>(bound2[0]));
        float t1 = sceneR + length(P) + 2.0f * params.roi;

        float vis = lightVisibility(P, L, 1e-3f, t1, gid, 0xFFFFFFFFu, params.lightSoft,
                                    positions, radii, binStarts, particleIndices,
                                    occupancy, params);
        if (vis > 0.0f) {
            n += params.lightNutrient * (float)params.lightInterval * vis;   // pulsed production
        }
        litFraction[gid] = vis;
        nutrient[gid] = clamp(n, 0.0f, kNutrientMax);
    }

    // M5 nutrient decay + diffusion (paper: nutrient "can also diffuse from
    // one cell to another along the links between cells"): one Jacobi pass per
    // step, with the first-order decay folded in:
    //   nutrient' = (1 - lightDecay) * (nutrient + lightDiffusionRate *
    //                                   (avg(ring) - nutrient))
    // Double-buffered like positions (read current, write next; host swaps).
    // Stability: lightDiffusionRate in [0, 1] makes the diffusion a convex
    // combination of the cell and its ring (CLI clamps); the decay factor
    // shrinks the whole update — both preserve non-negativity.
    //
    // Buffer table:
    //   0 links (const)   1 linkCounts (const)   2 nutrientCur (const)
    //   3 nutrientNext (write)   4 cellCounter   5 params
    kernel void nutrientDiffuse(device const uint* links      [[buffer(0)]],
                                device const uint*   linkCounts [[buffer(1)]],
                                device const float*  nutrientCur [[buffer(2)]],
                                device float*        nutrientNext [[buffer(3)]],
                                device const uint*   cellCounter [[buffer(4)]],
                                constant SimParams&  params      [[buffer(5)]],
                                uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float n = nutrientCur[gid];
        uint   cnt = linkCounts[gid];
        uint   base = gid * params.maxValence;
        float acc = 0.0f;
        for (uint k = 0; k < cnt; ++k) {
            acc += nutrientCur[links[base + k]];
        }
        float avg = cnt > 0 ? acc / float(cnt) : n;
        float diffused = n + params.lightDiffusionRate * (avg - n);
        nutrientNext[gid] = (1.0f - params.lightDecay) * diffused;
    }

    // Deterministic per-cell integer hash for the RDE seed noise (independent
    // of the per-cell food RNG streams — seeding must not perturb food
    // semantics or the lockstep streams).
    static inline uint chemHash(uint x) {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    // M5 RDE initialization: u = 1, v = sparse CLUSTER seeds. Nucleation
    // physics (measured, rde-probe): an ISOLATED seed cell loses v to diffusion
    // at Dv·v per step while gaining u·v² — it only grows above
    // v > (F + k + Dv)/u ≈ 0.6, so single-cell seeds at 0.5 die unless a lucky
    // adjacent pair happens to seed together (at subdiv 4 two such pairs
    // nucleated the whole pattern; at subdiv 3 zero pairs formed and the field
    // went extinct — the probe's first subdiv-3 run). A seed CLUSTER (a cell
    // and its whole ring) has vAvg ≈ v inside, no diffusive loss, and grows
    // robustly — the classic Turk-style nucleation. Cluster centres: ~1% of
    // cells (hash of gid + rngSeed), rings read from the link arrays (frozen
    // at reset). Background noise stays far below threshold (amplitude 0.005).
    // Deterministic; independent of the per-cell food RNG streams.
    //
    // Buffer table:
    //   0 links (const)   1 linkCounts (const)   2 chemU (write)   3 chemV (write)
    //   4 cellCounter   5 params
    kernel void seedChemicals(device const uint* links    [[buffer(0)]],
                              device const uint*   linkCounts [[buffer(1)]],
                              device float* u          [[buffer(2)]],
                              device float* v          [[buffer(3)]],
                              device const uint* cellCounter [[buffer(4)]],
                              constant SimParams& params     [[buffer(5)]],
                              uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        u[gid] = 1.0f;
        float r = (float)(chemHash(gid * 747796405u ^ params.rngSeed * 2654435761u) & 0xFFFFFFu)
                  * (1.0f / 16777216.0f);
        float seeded = (float)(chemHash(gid * 2654435761u ^ params.rngSeed * 747796405u) & 0xFFFFu)
                       * (1.0f / 65535.0f);
        bool centre = seeded < 0.01f;
        bool inCluster = centre;
        if (!inCluster) {
            uint base = gid * params.maxValence;
            uint n = linkCounts[gid];
            for (uint k = 0; k < n && !inCluster; ++k) {
                uint j = links[base + k];
                float sn = (float)(chemHash(j * 2654435761u ^ params.rngSeed * 747796405u) & 0xFFFFu)
                           * (1.0f / 65535.0f);
                inCluster = sn < 0.01f;
            }
        }
        v[gid] = centre ? 0.5f : (inCluster ? 0.35f : 0.005f * r);
    }

    // M5 Gray-Scott RDE over the cell graph (paper §4 method 2, à la Turk
    // [5]): chemicals stored per cell, diffusing along the direct links.
    //   du/dt = Du·∇²u − u·v² + F·(1−u)
    //   dv/dt = Dv·∇²v + u·v² − (F+k)·v
    // Explicit Euler with dt = 1 per simulation step; ∇² is the graph
    // Laplacian (avg(ring) − x), i.e. per-link flux normalized by valence —
    // NOT the 4-neighbour grid stencil D·(Σnbrs − 4x) of the classic
    // formulations. Defaults Du = 0.35, Dv = 0.15: half the classic Gray-Scott
    // texture mixing weights (Karl Sims' 3x3 convolution: Σw = 1, center −1)
    // in this normalization, which halves the pattern wavelength so spots
    // localize even on the 642-cell validation ball (at 1.0/0.5 the u
    // depletion homogenizes over that small domain and the field goes extinct
    // after nucleation — measured, rde-probe). Stability: the diffusion part
    // is a convex combination of the cell and its ring for Du, Dv in [0, 1];
    // reaction magnitudes are O(1) at dt = 1 (u·v² <= ~0.25 in the spotted
    // regime) — verified by --rde-probe.
    // u, v are clamped to [0, 2] as a numerical guard only.
    //
    // Division inheritance (divisionExecute): daughters copy the parent's
    // u, v (mitosis: concentrations, not quantities).
    //
    // V2.4 differentiation contract — "new types IGNORE the RDE (no reaction
    // participation)": a cell with type != 0 does not update its chemicals —
    // next = current (no reaction, no diffusion update). Its FROZEN values
    // are still read by untyped neighbours' Laplacian stencil (the stencil
    // shape and valence normalization are untouched — the simplest race-free
    // reading; frozen v-high regions then PIN the pattern: it cannot drift or
    // erode under the type map, and the frozen v is what sets the regional
    // growth drive in foodStep). For all-type-0 runs the branch is a no-op:
    // the reaction math below executes verbatim (the type-0 contract — one
    // extra buffer read per cell, no arithmetic change).
    //
    // Buffer table:
    //   0 links (const)   1 linkCounts (const)   2 chemUCur (const)
    //   3 chemVCur (const)   4 chemUNext (write)   5 chemVNext (write)
    //   6 cellCounter   7 params   8 cellType (const; V2.4 reaction freeze)
    kernel void rdeStep(device const uint* links      [[buffer(0)]],
                        device const uint*   linkCounts [[buffer(1)]],
                        device const float*  chemUCur   [[buffer(2)]],
                        device const float*  chemVCur   [[buffer(3)]],
                        device float*        chemUNext  [[buffer(4)]],
                        device float*        chemVNext  [[buffer(5)]],
                        device const uint*   cellCounter [[buffer(6)]],
                        constant SimParams&  params     [[buffer(7)]],
                        device const uint*   cellType   [[buffer(8)]],
                        uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float u = chemUCur[gid];
        float v = chemVCur[gid];
        uint   cnt = linkCounts[gid];
        uint   base = gid * params.maxValence;
        float uAvg = 0.0f, vAvg = 0.0f;
        for (uint k = 0; k < cnt; ++k) {
            uint j = links[base + k];
            uAvg += chemUCur[j];
            vAvg += chemVCur[j];
        }
        if (cnt > 0) {
            float invN = 1.0f / float(cnt);
            uAvg *= invN;
            vAvg *= invN;
        } else {
            uAvg = u; vAvg = v;
        }
        if (cellType[gid] != 0u) {
            // V2.4: differentiated — no reaction participation; freeze.
            chemUNext[gid] = u;
            chemVNext[gid] = v;
            return;
        }
        float uv2 = u * v * v;
        float du = params.rdeDu * (uAvg - u) - uv2 + params.rdeF * (1.0f - u);
        float dv = params.rdeDv * (vAvg - v) + uv2 - (params.rdeF + params.rdeK) * v;
        chemUNext[gid] = clamp(u + du, 0.0f, 2.0f);
        chemVNext[gid] = clamp(v + dv, 0.0f, 2.0f);
    }

    // M5/V2.2/V2.3: per-cell RENDER colour. Tint modes (params.tintMode,
    // V2.2 — appended enum; 0 reproduces the pre-V2.2 behaviour exactly, so
    // every existing golden stays byte-identical):
    //   0 auto     — chemical colours iff growthMode == 2 (rde), else the
    //                neutral albedo
    //   1 none     — neutral albedo always
    //   2 chemical — the rde u/v mapping in EVERY growth mode (chemicals are
    //                seeded at reset in all modes; only rde evolves them)
    //   3 food     — growth-substance heatmap (mapping below)
    //   4 pulse    — base tint (auto rule) blended with the magenta division
    //                pulse (see pulseDecayStep / divisionExecute)
    //   5 type     — V2.3 TYPE colours: stem white, lamel warm white, dal
    //                cool white, cap red; type 0 cells stay neutral. The
    //                palette is deliberately NEAR-WHITE for the three
    //                non-cap types (white / warm white / cool white) so the
    //                foam walls — which average ADJACENT cells' tints —
    //                blend to near-white inside the band (only the cap
    //                boundary shows a strong red<->white blend), and the
    //                red cap reads as the form's one saturated mass.
    //   6 type-pulse — mode 5 + the mode-4 mitosis-pulse overlay: the V2.3
    //                phase-2 validation view. In phase 2 ONLY voeg cells
    //                divide, so the magenta flashes must light up the BAND
    //                and nowhere else — a direct visual proof of the rule
    //                matrix (checked quantitatively by the per-type pulse
    //                means in the typed checkpoint stats).
    //
    // In rde mode the chemicals tint the cells: base GREEN, shifted toward
    // BLUE proportional to u and toward RED proportional to v (documented
    // mapping below); in uniform/light mode the neutral albedo is written —
    // bit-identical to the pre-M5 hardcoded colours (the impostor fragment's
    // float3(0.87, 0.85, 0.82) and RayRenderSettings.albedo), so the M3/M4
    // goldens are unaffected.
    //
    // RDE mapping (linear blend of three base colours):
    //   green (0.15, 0.65, 0.20), blue (0.15, 0.25, 0.85), red (0.85, 0.15, 0.15)
    //   wBlue = 0.30 * clamp(u, 0, 1)   — u's background is 1, so the field
    //                                     carries a teal-green cast; u-depleted
    //                                     spot interiors lose the blue cast
    //   wRed  = 1.60 * clamp(v, 0, 1)   — v's dynamic range is ~0.4 (vs u's
    //                                     1.0); the 1.6 scale makes spots
    //                                     (v ≈ 0.35–0.45) read as clear red
    //                                     patches (wRed ≈ 0.56–0.72)
    //   wGreen = max(1 − wBlue − wRed, 0) — low-concentration cells stay green
    //
    // V2.2 food-heatmap mapping (the documented "ONE clear mapping"): ONE
    // scalar — the cell's growth substance, normalized by heatScale — over a
    // cool-to-warm ramp:
    //   value = food + nutrient   in light mode: nutrient IS the light-made
    //                             growth substance and food the accumulated
    //                             driver; their sum is the cell's growth
    //                             potential (nutrient is 0 in the other modes)
    //   value = food              otherwise
    //   heatScale <= 1e-6 (the default) selects the AUTO normalization:
    //     light mode: splitThreshold + lightNutrient/lightDecay — food
    //       oscillates up to ~splitThreshold before division resets it, and
    //       the lit nutrient steady state is lightNutrient/lightDecay, so
    //       the sum spans exactly this range;
    //     otherwise:  splitThreshold (division keeps food below
    //       ~threshold + one increment).
    //   Ramp (4 linear stops — deep blue -> green -> yellow -> red):
    //     0.0 (0.05, 0.08, 0.55)   1/3 (0.15, 0.70, 0.25)
    //     2/3 (0.95, 0.80, 0.20)   1.0 (0.90, 0.15, 0.10)
    //
    // V2.2 mitosis pulse: divisionExecute writes pulse = 1.0 for the parent
    // and the daughter; pulseDecayStep multiplies it by params.pulseDecay
    // every step (default 0.95 — half-life ln(0.5)/ln(0.95) ≈ 13.5 steps,
    // weight > 0.1 for ~45 steps). The pulse hue is MAGENTA (0.95, 0.15,
    // 0.85) — roughly the complement of both the neutral green-ish albedo
    // and the rde green field, so the flash reads against either base:
    //   col = mix(base, pulseHue, min(pulse, 1))
    // In light mode divisions concentrate in the lit band, so the decaying
    // pulses trace the growth wavefront propagating across the surface (the
    // paper §7 "rhythmic pulses and waves of growth"); in uniform mode they
    // speckle everywhere. Validated in the V2.2 report (screenshots + the
    // pulse ring-R² checkpoint stats).
    //
    // Runs every step, AFTER the division block (M6: the daughters born this
    // step inherit the parent's u/v inside divisionExecute, so tint must be
    // recomputed from the post-division chemicals or last-step daughters
    // render with a stale/zero tint). The pre-integrate call (division off)
    // keeps tints fresh from the chemical passes.
    //
    // Buffer table:
    //   0 chemU (const)   1 chemV (const)   2 tint (write)   3 cellCounter
    //   4 params
    //   5 food (const; V2.2 food heatmap)  6 nutrient (const; V2.2)
    //   7 pulse (const; V2.2 mitosis pulse)
    //   8 cellType (const; V2.3 type colours, modes 5/6)
    kernel void cellTint(device const float* chemU    [[buffer(0)]],
                         device const float* chemV    [[buffer(1)]],
                         device float3*      tint     [[buffer(2)]],
                         device const uint*  cellCounter [[buffer(3)]],
                         constant SimParams& params   [[buffer(4)]],
                         device const float* food     [[buffer(5)]],
                         device const float* nutrient [[buffer(6)]],
                         device const float* pulse    [[buffer(7)]],
                         device const uint*  cellType [[buffer(8)]],
                         uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        const uint mode = params.tintMode;
        float3 col = float3(0.87, 0.85, 0.82);   // neutral albedo (pre-M5 value)
        const bool chemical = (mode == 2u) || (mode == 0u && params.growthMode == 2u);
        if (chemical) {
            float uC = clamp(chemU[gid], 0.0f, 1.0f);
            float vC = clamp(chemV[gid], 0.0f, 1.0f);
            float wBlue = 0.30f * uC;
            float wRed  = 1.60f * vC;
            float wGreen = max(1.0f - wBlue - wRed, 0.0f);
            col = wGreen * float3(0.15, 0.65, 0.20)
                + wBlue  * float3(0.15, 0.25, 0.85)
                + wRed   * float3(0.85, 0.15, 0.15);
        } else if (mode == 3u) {
            // Food/nutrient heatmap (mapping documented above).
            float scale = params.heatScale;
            if (scale <= 1e-6f) {
                scale = (params.growthMode == 1u)
                    ? (params.splitThreshold
                       + params.lightNutrient / max(params.lightDecay, 1e-3f))
                    : params.splitThreshold;
            }
            float value = (params.growthMode == 1u)
                        ? (food[gid] + nutrient[gid])
                        : food[gid];
            float h = clamp(value / max(scale, 1e-6f), 0.0f, 1.0f);
            const float3 c0 = float3(0.05, 0.08, 0.55);   // deep blue (cold)
            const float3 c1 = float3(0.15, 0.70, 0.25);   // green
            const float3 c2 = float3(0.95, 0.80, 0.20);   // yellow
            const float3 c3 = float3(0.90, 0.15, 0.10);   // red (hot)
            if (h < 1.0f / 3.0f)      col = mix(c0, c1, h * 3.0f);
            else if (h < 2.0f / 3.0f) col = mix(c1, c2, (h - 1.0f / 3.0f) * 3.0f);
            else                      col = mix(c2, c3, (h - 2.0f / 3.0f) * 3.0f);
        } else if (mode == 5u || mode == 6u) {
            // V2.3 type colours (palette rationale in the comment above).
            // Type 0 keeps the neutral albedo — an untyped run renders
            // exactly as before in every mode.
            uint t = cellType[gid];
            col = (t == 1u) ? float3(1.00, 1.00, 1.00)    // stem: white
                : (t == 2u) ? float3(1.00, 0.88, 0.72)    // lamel: warm white
                : (t == 3u) ? float3(0.72, 0.86, 1.00)    // dal: cool white
                : (t == 4u) ? float3(0.80, 0.10, 0.10)    // cap: red
                : float3(0.87, 0.85, 0.82);               // type 0: neutral
        }
        // V2.2 pulse blend (over the base tint of every mode); V2.3 mode 6
        // (type-pulse) shares it.
        if (mode == 4u || mode == 6u) {
            float p = clamp(pulse[gid], 0.0f, 1.0f);
            col = mix(col, float3(0.95, 0.15, 0.85), p);
        }
        tint[gid] = col;
    }

    // V2.2 mitosis pulse decay: pulse *= pulseDecay every step, flushed to
    // zero below 1e-4 (the tint weight there is < 0.01% — invisible; the
    // flush keeps the buffer free of denormal clutter). Runs BEFORE the
    // division block, so a division THIS step flashes at pulse = 1.0 and the
    // decay starts on the NEXT step. Encoded every step regardless of the
    // tint mode (one FMA per cell — benchlog V2.2: ~nil cost), so switching
    // tint modes mid-run never shows stale pulses.
    //
    // Buffer table:
    //   0 pulse (read + write)   1 cellCounter   2 params
    kernel void pulseDecayStep(device float*        pulse [[buffer(0)]],
                               device const uint*   cellCounter [[buffer(1)]],
                               constant SimParams&  params [[buffer(2)]],
                               uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float p = pulse[gid] * params.pulseDecay;
        pulse[gid] = p < 1e-4f ? 0.0f : p;
    }

    // ---- cell division, pass 1: claim (paper §4, Figure 7) ----
    //
    // A cell whose food exceeds splitThreshold becomes a split candidate. To
    // keep the concurrent split mechanics race-free, candidates form a
    // deterministic independent set at ring-distance 2: a candidate may claim
    // a split only if no LOWER-indexed candidate lies within ring-distance 2
    // (distance 1 = linked cells share their ring slots; distance 2 = cells
    // sharing a neighbour both touch that neighbour's ring). Blocked cells
    // keep their food above the threshold and simply retry next step — nothing
    // is lost. This replaces a CAS reservation loop with a conflict-free
    // min-index rule evaluated on FROZEN data: nothing in this kernel writes
    // links/positions/food, so every thread sees the same state.
    //
    // The claim flag is consumed by divisionExecute, which MUST be dispatched
    // with the same width (claims beyond the width are then never consumed).
    //
    // V2.3 TYPE-DEPENDENT RULE MATRIX (gate, before the historical checks):
    //
    //   Phase 1 (divCounter < phaseSwitchX, or phaseSwitchX == 0):
    //     type 0 (none): NO gate — the historical uniform rule verbatim
    //                    (the type-0 contract: untyped runs are untouched)
    //     type 1 (stem): divides ONLY with a voeg neighbour (lamel/dal in its
    //                    ring) — edge growth at the band
    //     type 2/3 (voeg): never (dormant; their food multiplier is also 0 —
    //                    see foodStep)
    //     type 4 (cap): uniform (the old rule)
    //   Phase 2 (divCounter >= phaseSwitchX > 0): ONLY voeg cells divide
    //     (type 0 exempt by contract — an untyped run stays uniform in both
    //     phases):
    //     type 1/4 (stem/cap): never
    //     type 2 (lamel): needs an EDGE link (stem/cap neighbour) or an
    //                     interior lamel<->dal link; lamel<->lamel never
    //     type 3 (dal):   same (edge or a lamel neighbour)
    //
    // V2.4 REGION mode (params.typeMode == 1 — RDE-typed runs): the band
    // matrix above assumes a POLAR band (its edge/interior link classes are
    // band concepts); an RDE type map has no band. In region mode EVERY type
    // divides uniformly within its own region (stem anywhere, cap anywhere —
    // "patterned growth = regional rate differences": the rates come from the
    // frozen v coupling in foodStep, not from the matrix). The two-phase
    // machinery is band-specific and is not applied in region mode. Type 0 is
    // unconditionally eligible in BOTH modes (the type-0 contract: an
    // all-type-0 run restores the historical predicate exactly).
    // Mode summary (the full matrix, per mode):
    //   band  (polar typing):  phase 1 — stem edge-only, voeg dormant, cap
    //          uniform; phase 2 — voeg-only (edge/interior links).
    //   region (RDE typing):   every type uniform, both phases (the phase
    //          switch is ignored).
    //   all-type-0:            the historical uniform rule (either mode).
    //
    // BLOCKER PREDICATE (V2.3, critical): the independent-set test must count
    // only MATRIX-ELIGIBLE cells as blockers, not every food-heavy cell. With
    // the historical food-only predicate, a cell that hoards food above the
    // threshold while being permanently INELIGIBLE under the matrix (a stem
    // cell with no voeg neighbour; any stem/cap cell in phase 2 — their food
    // multiplier is 0, so the hoard never clears) sits at ring-distance <= 2
    // of the growth front forever and BLOCKS every higher-indexed eligible
    // candidate — measured: the stem edge front froze after 13 divisions
    // (t2 = 10 isolation probe, 2000 steps). The M2 "blocked cells simply
    // retry" claim relies on blockers eventually dividing themselves clear;
    // the matrix breaks that, so the blocker set must be the ELIGIBLE set.
    // For type-0 runs typeEligible is constant true, which restores the
    // historical predicate EXACTLY (the type-0 contract).
    //
    // Race-freedom of the matrix: every type check is a READ of FROZEN data —
    // links, linkCounts and cellType are not written by any kernel between
    // the step start and this pass (divisionExecute, the only writer of
    // cellType, runs AFTER decide in the same command buffer), and decide
    // itself writes ONLY splitFlags[gid]. The M2 ring-distance-2 argument
    // therefore extends verbatim: the claim set is the set of matrix-eligible
    // local minima (two eligible cells within ring-distance 2 — the lower
    // index blocks the higher, whichever it is), so concurrent claimants
    // still have pairwise distance >= 3 and disjoint write sets, and the
    // added reads cannot observe a concurrent write (there is none). The
    // phase read (divCounter[0]) is frozen within the step for the same
    // reason: divisionExecute's atomic increment happens LATER in the same
    // command buffer, so every decide thread of a step evaluates the same
    // pre-step phase — the phase switches at the first step boundary AFTER
    // the X-th executed division (exactly once; the counter is monotonic).
    //
    // V2.3: rule-matrix eligibility of ONE cell — the decide gate for the
    // candidate itself and the blocker predicate for its ring-distance <= 2
    // neighbours (same predicate both roles — see the blocker-predicate note
    // above). Type 0 (none) is unconditionally eligible: the matrix is a
    // no-op on untyped runs. V2.4: typeMode selects the matrix (band | region
    // — see the mode summary above). All reads are of frozen data.
    static inline bool typeEligible(uint c,
                                    device const uint* links,
                                    device const uint* linkCounts,
                                    device const uint* cellType,
                                    uint typeMode, bool phase2, uint maxValence) {
        uint t = cellType[c];
        if (t == 0u) return true;
        if (typeMode == 1u) return true;   // V2.4 region: uniform within region
        uint n = linkCounts[c];
        uint base = c * maxValence;
        if (!phase2) {
            if (t == 2u || t == 3u) return false;   // voeg dormant in phase 1
            if (t == 1u) {
                // Stem: eligible only at the band edge (a voeg neighbour).
                for (uint k = 0; k < n; ++k) {
                    uint tj = cellType[links[base + k]];
                    if (tj == 2u || tj == 3u) return true;
                }
                return false;
            }
            return true;                            // cap: uniform
        }
        if (t == 1u || t == 4u) return false;       // phase 2: voeg-only
        bool edge = false;       // neighbour = stem (1) or cap (4)
        bool interior = false;   // neighbour = the complementary voeg type
        for (uint k = 0; k < n; ++k) {
            uint tj = cellType[links[base + k]];
            if (tj == 1u || tj == 4u) edge = true;
            else if (t == 2u ? tj == 3u : tj == 2u) interior = true;
        }
        return edge || interior;
    }

    // Buffer table:
    //   0 food (const)          1 links (const)     2 linkCounts (const)
    //   3 splitFlags (write)    4 cellCounter       5 params
    //   6 cellType (const; V2.3 rule matrix)
    //   7 divCounter (const; V2.3 phase clock)
    kernel void divisionDecide(device const float* food      [[buffer(0)]],
                               device const uint*   links          [[buffer(1)]],
                               device const uint*   linkCounts     [[buffer(2)]],
                               device uint*         splitFlags     [[buffer(3)]],
                               device const uint*   cellCounter    [[buffer(4)]],
                               constant SimParams&  params         [[buffer(5)]],
                               device const uint*   cellType       [[buffer(6)]],
                               device const uint*   divCounter     [[buffer(7)]],
                               uint gid [[thread_position_in_grid]])
    {
        uint count = cellCounter[0];
        if (gid >= count) { splitFlags[gid] = 0u; return; }
        if (params.divisionEnabled == 0u) { splitFlags[gid] = 0u; return; }
        if (food[gid] <= params.splitThreshold) { splitFlags[gid] = 0u; return; }

        // V2.3 rule matrix gate (type 0 exempt — typeEligible is constant
        // true there and this is a no-op read). V2.4: the mode selects the
        // matrix (band | region).
        const bool phase2 = (params.phaseSwitchX != 0u)
                         && (divCounter[0] >= params.phaseSwitchX);
        if (!typeEligible(gid, links, linkCounts, cellType, params.typeMode,
                          phase2, params.maxValence)) {
            splitFlags[gid] = 0u;
            return;
        }

        uint n = linkCounts[gid];
        if (n < 3u || n > params.maxValence) { splitFlags[gid] = 0u; return; }

        uint base = gid * params.maxValence;
        for (uint k = 0; k < n; ++k) {
            uint j = links[base + k];
            if (j == gid) { splitFlags[gid] = 0u; return; }   // corrupt ring: defer
            if (food[j] > params.splitThreshold && j < gid
                && typeEligible(j, links, linkCounts, cellType, params.typeMode,
                                phase2, params.maxValence)) {
                splitFlags[gid] = 0u; return;
            }
            uint jn = linkCounts[j];
            if (jn > params.maxValence) { splitFlags[gid] = 0u; return; }
            uint jbase = j * params.maxValence;
            for (uint k2 = 0; k2 < jn; ++k2) {
                uint l = links[jbase + k2];
                if (l != gid && food[l] > params.splitThreshold && l < gid
                    && typeEligible(l, links, linkCounts, cellType, params.typeMode,
                                    phase2, params.maxValence)) {
                    splitFlags[gid] = 0u;
                    return;
                }
            }
        }
        splitFlags[gid] = 1u;
    }

    // ---- helpers shared by divisionExecute ----

    // Index of `target` in `cell`'s ring, or -1.
    static inline int findSlot(device const uint* links, device const uint* linkCounts,
                               uint cell, uint target, uint maxValence) {
        uint base = cell * maxValence;
        uint n = linkCounts[cell];
        for (uint k = 0; k < n; ++k) {
            if (links[base + k] == target) return (int)k;
        }
        return -1;
    }

    static inline bool inList(thread const uint* list, uint count, uint value) {
        for (uint k = 0; k < count; ++k) {
            if (list[k] == value) return true;
        }
        return false;
    }

    // Where to insert the daughter in a plane neighbour's ring so it ends up
    // beside the parent on the daughter-arc side. Returns the linear insertion
    // index, or -1 when neither ring neighbour of the parent's slot lies in
    // the daughter arc (non-contiguous split: caller defers).
    static inline int insertionSlot(device const uint* links, device const uint* linkCounts,
                                    uint cell, uint parentSlot,
                                    thread const uint* bIdx, uint bCount, uint maxValence) {
        uint base = cell * maxValence;
        uint n = linkCounts[cell];
        uint prevS = (parentSlot + n - 1u) % n;
        uint nextS = (parentSlot + 1u) % n;
        if (inList(bIdx, bCount, links[base + nextS])) return (int)nextS;      // after parent
        if (inList(bIdx, bCount, links[base + prevS])) return (int)parentSlot; // before parent
        return -1;
    }

    // Insert `value` at linear position `slot`, shifting the tail up. The ring
    // is stored linearly with the cyclic wrap implicit, so a mid-array insert
    // preserves the cyclic order.
    static inline void insertIntoRing(device uint* links, device uint* linkCounts,
                                      uint cell, uint slot, uint value, uint maxValence) {
        uint base = cell * maxValence;
        uint n = linkCounts[cell];
        for (uint k = n; k > slot; --k) {
            links[base + k] = links[base + k - 1u];
        }
        links[base + slot] = value;
        linkCounts[cell] = n + 1u;
    }

    // ---- cell division, pass 2: mechanics ----
    //
    // Topology rule (paper §4): two of the cell's links are chosen to represent
    // the plane of cleavage. Links to one side of the plane stay connected to
    // the parent; links to the other side are disconnected from the parent and
    // relinked to the daughter; the two chosen (plane) links connect to BOTH
    // parent and daughter; a new direct link parent<->daughter is created.
    // Net effect per split: +1 cell, +3 links — Euler-consistent for a closed
    // triangulation (dV=1, dE=3, dF=4, so sum valence grows by 6 = 6*dV).
    //
    // Choices where the paper is silent (see the M2 report for the full write-up):
    //   - Plane links: the two ring links with the largest angular separation
    //     (most nearly opposite). The cleavage plane contains P, the axis
    //     through those two links, and the surface normal N; its tangential
    //     normal is m = normalize(cross(N, w)) with w = normalize(h1 - h2).
    //     NOTE: the naive m = normalize(cross(d1, d2)) is degenerate here:
    //     d1/d2 are near-tangent link directions, so cross(d1, d2) ~ +/-N, and
    //     on a smooth convex patch every neighbour sits slightly below the
    //     tangent plane — all on the SAME side, no partition. The plane spanned
    //     by the chosen axis and N instead cuts the ring into two arcs, which
    //     is the paper's intent; the two chosen links lie in it and are
    //     therefore the ones connected to both parent and daughter.
    //   - Daughter placement: P + 0.5 * linkRestLength along the bisector of
    //     the daughter-side arc (mean of the arc's link vectors).
    //   - Food reset: parent halved (finalizeFood), daughter starts at half
    //     the parent's pre-split food.
    //   - Daughter RNG stream: one LCG advance of the parent's stream.
    //
    // Race-freedom: claimants (decide pass) are pairwise at ring-distance >= 3,
    // so their write sets (own ring, daughter slot, one slot per B-side
    // neighbour, one insertion per plane neighbour — plus the M5 chemical
    // writes: own + daughter slot in nutrient/chemU/chemV, the V2.2 pulse
    // write, and the V2.3 daughter TYPE write) are disjoint. Reads
    // in this kernel touch only cells within ring-distance 1, which no
    // claimant writes. The daughter slot is >= the step-start count, disjoint
    // from the positions integrateForces wrote into this same buffer.
    // V2.3 additions to the write set: cellType[d] (the daughter inherits the
    // parent's type — a FRESH slot, disjoint like the chemical writes) and
    // one atomic increment of divisionCounter[0] per successful split (the
    // phase clock; consumed by decide/foodStep of the NEXT step — encoder
    // order makes that read see the completed increment).
    //
    // Ring handling preserves the CYCLIC CCW order (no re-sort): the ring is
    // walked in cyclic order rotated to start at the first plane link; the
    // parent keeps [plane1, daughter, plane2, parent-side...]; the daughter
    // gets [plane1, daughter-side arc..., plane2, parent]. Consecutive pairs in
    // both new rings are consecutive pairs of the original ring or new links —
    // hence valid edges. The daughter-side entries must form ONE contiguous arc
    // flanked by the two plane links (structurally guaranteed for star-shaped
    // rings: the side value varies like sin of the angle to the axis, so its
    // sign changes exactly across the axis links); violations defer the split,
    // which is always safe (food is kept).
    //
    // Valence guard: a split is skipped if it would push the parent
    // (aCount + 3), the daughter (bCount + 3) or either plane neighbour (+1)
    // past maxValence.
    //
    // Buffer table:
    //   0 positions (post-integrate; read + daughter write)
    //   1 links (read + write)          2 linkCounts (read + write)
    //   3 food (read; daughter write)   4 rngState (read; daughter write)
    //   5 splitFlags (read + write)     6 cellCounter (atomic; fetch_add/fetch_sub)
    //   7 params
    //   8 nutrient (read + write; M5: cytoplasm split between parent/daughter)
    //   9 chemU (read parent, write daughter; M5 rde inheritance)
    //  10 chemV (read parent, write daughter; M5 rde inheritance)
    //  11 pulse (write; V2.2: parent + daughter flash at 1.0)
    //  12 cellType (read parent, write daughter; V2.3 type inheritance)
    //  13 divisionCounter (atomic add; V2.3 phase clock)
    kernel void divisionExecute(device float3* positions    [[buffer(0)]],
                                device uint*   links        [[buffer(1)]],
                                device uint*   linkCounts   [[buffer(2)]],
                                device float*  food         [[buffer(3)]],
                                device uint64_t* rngState   [[buffer(4)]],
                                device uint*   splitFlags   [[buffer(5)]],
                                device atomic_uint* cellCounter [[buffer(6)]],
                                constant SimParams& params   [[buffer(7)]],
                                device float*  nutrient     [[buffer(8)]],
                                device float*  chemU        [[buffer(9)]],
                                device float*  chemV        [[buffer(10)]],
                                device float*  pulse        [[buffer(11)]],
                                device uint*   cellType     [[buffer(12)]],
                                device atomic_uint* divisionCounter [[buffer(13)]],
                                uint gid [[thread_position_in_grid]])
    {
        if (gid >= atomic_load_explicit(&cellCounter[0], memory_order_relaxed)) return;
        if (splitFlags[gid] == 0u) return;

        uint n = linkCounts[gid];
        if (n < 3u || n > kMaxValence) { splitFlags[gid] = 0u; return; }
        uint base = gid * params.maxValence;

        // Stage the ring (cell ids + post-integrate positions) and compute a
        // fresh area-weighted normal from the same walk (the normals buffer
        // still holds pre-integration values).
        float3 P = positions[gid];
        float3 ringPos[kMaxValence];
        uint   ringCell[kMaxValence];
        float3 acc = float3(0.0);
        float3 prev = positions[links[base + n - 1]] - P;
        for (uint k = 0; k < n; ++k) {
            ringCell[k] = links[base + k];
            ringPos[k] = positions[ringCell[k]];
            float3 cur = ringPos[k] - P;
            acc += cross(prev, cur);
            prev = cur;
        }
        float acc2 = dot(acc, acc);
        if (acc2 < 1e-20f) { splitFlags[gid] = 0u; return; }
        float3 N = acc * rsqrt(acc2);

        // Choose the plane-of-cleavage links: the most nearly opposite pair.
        float bestDot = 2.0f;
        uint k1 = 0, k2 = 1;
        for (uint a = 0; a < n; ++a) {
            float3 da = ringPos[a] - P;
            float la = length(da);
            if (la < 1e-9f) { splitFlags[gid] = 0u; return; }
            da /= la;
            for (uint b = a + 1; b < n; ++b) {
                float3 db = ringPos[b] - P;
                float lb = length(db);
                if (lb < 1e-9f) { splitFlags[gid] = 0u; return; }
                float dt = dot(da, db / lb);
                if (dt < bestDot) { bestDot = dt; k1 = a; k2 = b; }
            }
        }

        // Cleavage plane normal (tangential): m = normalize(cross(N, w)).
        float3 h1 = normalize(ringPos[k1] - P);
        float3 h2 = normalize(ringPos[k2] - P);
        float3 w = h1 - h2;
        float wLen = length(w);
        if (wLen < 1e-5f) { splitFlags[gid] = 0u; return; }   // chosen links parallel
        w /= wLen;
        float3 m = cross(N, w);
        float m2 = dot(m, m);
        if (m2 < 1e-12f) { splitFlags[gid] = 0u; return; }    // N parallel to axis
        m *= rsqrt(m2);

        // Classify the ring in cyclic order rotated to start at k1.
        // Daughter side (B): dot(L - P, m) > 0; parent side (A): <= 0. k2 sits
        // at rotated offset k2rot; entries between k1 and k2 must all be B,
        // entries after k2 must all be A — otherwise the arc is not contiguous
        // and the split would break ring validity: defer.
        uint k2rot = (k2 + n - k1) % n;
        uint   bIdx[kMaxValence];
        float3 bVec[kMaxValence];
        uint bCount = 0;
        for (uint t = 1; t < k2rot; ++t) {
            uint r = (k1 + t) % n;
            if (dot(ringPos[r] - P, m) <= 0.0f) { splitFlags[gid] = 0u; return; }
            bIdx[bCount] = ringCell[r];
            bVec[bCount] = ringPos[r] - P;
            ++bCount;
        }
        if (bCount == 0u) { splitFlags[gid] = 0u; return; }   // no daughter arc
        uint aCount = 0;
        for (uint t = k2rot + 1; t < n; ++t) {
            uint r = (k1 + t) % n;
            if (dot(ringPos[r] - P, m) > 0.0f) { splitFlags[gid] = 0u; return; }
            ++aCount;
        }

        uint plane1 = ringCell[k1];
        uint plane2 = ringCell[k2];
        if (plane1 == plane2) { splitFlags[gid] = 0u; return; }   // duplicated ring entry

        // Valence guards (parent keeps aCount+3, daughter gets bCount+3, each
        // plane neighbour gains one link).
        if (aCount + 3u > params.maxValence) { splitFlags[gid] = 0u; return; }
        if (bCount + 3u > params.maxValence) { splitFlags[gid] = 0u; return; }
        if (linkCounts[plane1] + 1u > params.maxValence) { splitFlags[gid] = 0u; return; }
        if (linkCounts[plane2] + 1u > params.maxValence) { splitFlags[gid] = 0u; return; }

        // Pre-scan neighbour rings (read-only) before allocating anything, so a
        // failed scan leaves no partial state behind.
        uint bSlot[kMaxValence];
        for (uint b = 0; b < bCount; ++b) {
            int s = findSlot(links, linkCounts, bIdx[b], gid, params.maxValence);
            if (s < 0) { splitFlags[gid] = 0u; return; }   // asymmetric link: defer
            bSlot[b] = (uint)s;
        }
        int p1Slot = findSlot(links, linkCounts, plane1, gid, params.maxValence);
        int p2Slot = findSlot(links, linkCounts, plane2, gid, params.maxValence);
        if (p1Slot < 0 || p2Slot < 0) { splitFlags[gid] = 0u; return; }
        int p1At = insertionSlot(links, linkCounts, plane1, (uint)p1Slot, bIdx, bCount, params.maxValence);
        int p2At = insertionSlot(links, linkCounts, plane2, (uint)p2Slot, bIdx, bCount, params.maxValence);
        if (p1At < 0 || p2At < 0) { splitFlags[gid] = 0u; return; }

        // Daughter placement. Scope suggestion: bisector of the daughter-side
        // arc at ~0.5 * linkRestLength. Tested alternative (kept here): radial
        // birth along the surface normal. The bisector birth lands INSIDE the
        // arc's territory (the arc members already occupy that region), which
        // creates a strong compression spike at every division — a fold seed
        // (folds then self-reinforce without repulsion, paper Fig. 5). The
        // radial birth only creates a mild bump that spring/planar smooth out.
        float3 bis = float3(0.0);
        for (uint b = 0; b < bCount; ++b) bis += bVec[b];
        float bisLen = length(bis);
        if (bisLen < 1e-6f) { splitFlags[gid] = 0u; return; }
        const bool radialBirth = false;
        float3 daughterPos;
        if (radialBirth) {
            daughterPos = P + (0.5f * params.linkRestLength) * N;
        } else {
            daughterPos = P + (0.5f * params.linkRestLength) * (bis / bisLen);
        }

        // Fast capacity pre-check (plain load, no RMW storm at the cap); the
        // atomic below stays authoritative for the rare concurrent overrun.
        if (atomic_load_explicit(&cellCounter[0], memory_order_relaxed) >= params.maxCells) {
            splitFlags[gid] = 0u;   // revoke the claim: nothing happened
            return;
        }

        // Allocate the daughter slot; respect the hard capacity. If concurrent
        // allocations overshoot, each over allocation is handed back below.
        uint d = atomic_fetch_add_explicit(&cellCounter[0], 1u, memory_order_relaxed);
        if (d >= params.maxCells) {
            atomic_fetch_sub_explicit(&cellCounter[0], 1u, memory_order_relaxed);
            splitFlags[gid] = 0u;   // revoke the claim: nothing happened
            return;
        }

        // Daughter cell. Its ring: [plane1, daughter-side arc..., plane2, parent]
        // — the original cyclic order restricted to the daughter side.
        positions[d] = P + (0.5f * params.linkRestLength) * (bis / bisLen);
        food[d] = 0.5f * food[gid];
        // M5 chemical inheritance. Nutrient: cytoplasm split (half each; the
        // level stays bounded by the lightDecay sink regardless). RDE u/v:
        // concentrations copied, not halved — pattern continuity across the
        // cleavage plane (Turk-style RDEs on growing surfaces re-seed new
        // cells from their neighbourhood; the parent copy is the cheapest
        // race-free equivalent and the diffusion pass smooths it immediately).
        nutrient[d] = 0.5f * nutrient[gid];
        nutrient[gid] = 0.5f * nutrient[gid];
        chemU[d] = chemU[gid];
        chemV[d] = chemV[gid];
        // V2.3: the daughter inherits the parent's TYPE (same write pattern
        // as the chemical inheritance above: a fresh daughter slot, claims
        // ring-distance >= 3 apart — write sets stay disjoint). Type is the
        // cell's identity in the rule matrix: daughters continue their
        // parent's role (stem stays stem, lamel stays lamel, ...).
        cellType[d] = cellType[gid];
        // V2.2 mitosis pulse: both cells flash at FULL pulse on the step of
        // the division (pulseDecayStep next step starts the decay). Write
        // sets stay disjoint — claims are ring-distance >= 3 apart and the
        // daughter slot is fresh (same race-freedom argument as the chemical
        // writes above).
        pulse[gid] = 1.0f;
        pulse[d] = 1.0f;
        rngState[d] = rngState[gid] * kRngMul + kRngInc;   // fresh stream off the parent's
        splitFlags[d] = 0u;
        uint dbase = d * params.maxValence;
        uint di = 0;
        links[dbase + di++] = plane1;
        for (uint b = 0; b < bCount; ++b) links[dbase + di++] = bIdx[b];
        links[dbase + di++] = plane2;
        links[dbase + di++] = gid;
        linkCounts[d] = di;                                // bCount + 3

        // Parent ring: the B-arc collapsed to the daughter, cyclic order kept.
        uint pi = 0;
        links[base + pi++] = plane1;
        links[base + pi++] = d;
        links[base + pi++] = plane2;
        for (uint t = k2rot + 1; t < n; ++t) {
            links[base + pi++] = ringCell[(k1 + t) % n];
        }
        linkCounts[gid] = pi;                              // aCount + 3

        // B-side neighbours: retarget the parent's slot to the daughter.
        for (uint b = 0; b < bCount; ++b) {
            links[bIdx[b] * params.maxValence + bSlot[b]] = d;
        }
        // Plane neighbours: insert the daughter beside the parent.
        insertIntoRing(links, linkCounts, plane1, (uint)p1At, d, params.maxValence);
        insertIntoRing(links, linkCounts, plane2, (uint)p2At, d, params.maxValence);

        splitFlags[gid] = 1u;   // success: finalizeFood halves the parent's food
        // V2.3 phase clock: one increment per EXECUTED division. decide and
        // foodStep of the NEXT command buffer read this (encoder order);
        // phase 2 begins at the first step boundary after the count reaches
        // params.phaseSwitchX.
        atomic_fetch_add_explicit(&divisionCounter[0], 1u, memory_order_relaxed);
    }

    // ---- cell division, pass 3: parent food reset ----
    //
    // Split parents get half their food (the daughter already received the
    // other half in divisionExecute). This runs in its own pass because the
    // division guard reads neighbours' food; writing the parent's food in the
    // same kernel as the guard could race with those reads.
    //
    // Buffer table:
    //   0 splitFlags (const)   1 food (write)   2 cellCounter   3 params
    kernel void finalizeFood(device const uint* splitFlags [[buffer(0)]],
                             device float* food            [[buffer(1)]],
                             device const uint* cellCounter [[buffer(2)]],
                             constant SimParams& params    [[buffer(3)]],
                             uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        if (splitFlags[gid] != 0u) food[gid] *= 0.5f;
    }

    // ---- V2.4: the differentiation phase — RDE-driven cell-type transition
    // ---- with hysteresis (paper §8: cell differentiation via parameter
    // ---- gradients; the paper is silent on the mechanism — user-brief
    // ---- design, documented here)
    //
    // The DIFFERENTIATION PHASE (params.differentiationPhase == 1) freezes
    // the whole division block (step() skips foodStep/decide/execute/
    // finalize — food stops accumulating too, so growth resumes WITHOUT a
    // synchronized burst: the pre-phase food distribution is preserved) while
    // the RDE keeps stepping. Each step, every UNDIFFERENTIATED cell (type 0)
    // may transition:
    //   v >= typeCapV  -> type 4 (cap)
    //   v <= typeStemV -> type 1 (stem)
    //   else           -> stays 0 (the undifferentiated dead band)
    //
    // HYSTERESIS = TYPE MEMORY (the chosen hysteresis form): a TYPED cell
    // ignores the RDE entirely from the step it transitions on — the early
    // exit below — so a cell's type changes AT MOST ONCE (0 -> 1 or 0 -> 4,
    // never back). Patterns therefore cannot flicker at a threshold boundary:
    // once committed, a cell stays in its region forever. (A full up/down
    // hysteresis BAND — typed cells re-reading the field with offset
    // thresholds — is a small extension of this kernel; type memory is the
    // minimal stable form and the one the brief specifies.) The same
    // "ignores the RDE" rule covers the chemicals themselves: rdeStep freezes
    // u/v for typed cells (no reaction participation). Frozen values are
    // still READ by untyped neighbours' stencil (see rdeStep) — a frozen
    // v-high region pins the pattern and carries the regional growth drive.
    //
    // PIPELINE ORDER + race-freedom (the milestone contract, stated): the
    // kernel is encoded AFTER the division block — same slot family as
    // cellTint. When the phase is off the division block runs and its
    // cellType writes (daughter inheritance) land BEFORE this kernel would
    // (it is not encoded at all then); when the phase is ON the division
    // block is skipped entirely and this kernel is the ONLY cellType writer
    // of the step. Every thread reads chemV[gid] (finalized by the chemical
    // pass at the top of the step — the host swapped the double buffers
    // before the division block — and written by nothing after) and writes
    // ONLY cellType[gid]: no cross-thread reads, nothing else writes cellType
    // inside this CB window, so per-cell writes are trivially race-free. The
    // transition's writes are consumed by NEXT step's foodStep/divisionDecide
    // (encoder order) and by THIS step's trailing cellTint (later encoder —
    // new types render the same step). Host-side re-typing between steps
    // stays safe as before (GPU idle; applyPolarTypes).
    //
    // Only meaningful in rde mode (growthMode == 2): the host encodes it only
    // then. In uniform/light modes the differentiation phase is a pure growth
    // pause — the seeded chemical field is static and would type from noise.
    //
    // Buffer table:
    //   0 chemV (const)   1 cellType (read + write)   2 cellCounter   3 params
    kernel void typeTransition(device const float* chemV    [[buffer(0)]],
                               device uint*        cellType [[buffer(1)]],
                               device const uint*  cellCounter [[buffer(2)]],
                               constant SimParams& params   [[buffer(3)]],
                               uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        uint t = cellType[gid];
        if (t != 0u) return;               // hysteresis = type memory (above)
        float v = chemV[gid];
        if (v >= params.typeCapV)       cellType[gid] = 4u;   // cap
        else if (v <= params.typeStemV) cellType[gid] = 1u;   // stem
        // else: the undifferentiated dead band (typeStemV, typeCapV) — stays 0.
    }

    // Render radius per cell = mean distance to linked cells (paper §5), scaled.
    //
    // Buffer table:
    //   0 positions (const)   1 links (const)      2 linkCounts (const)
    //   3 radii (write)       4 cellCounter        5 params
    kernel void computeRadii(device const float3* positions [[buffer(0)]],
                             device const uint*   links      [[buffer(1)]],
                             device const uint*   linkCounts [[buffer(2)]],
                             device float*        radii      [[buffer(3)]],
                             device const uint*   cellCounter [[buffer(4)]],
                             constant SimParams&  params     [[buffer(5)]],
                             uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float3 P = float3(positions[gid]);
        uint   n = linkCounts[gid];
        uint base = gid * params.maxValence;
        float s = 0.0;
        for (uint k = 0; k < n; ++k) {
            s += distance(float3(positions[links[base + k]]), P);
        }
        radii[gid] = (n > 0 ? s / float(n) : 0.0f) * params.radiusScale;
    }

    // Threadgroup-reduced stats: (sum of link lengths, link count) per threadgroup.
    // Requires dispatch with exactly 64 threads per threadgroup.
    //
    // Buffer table:
    //   0 positions (const)   1 links (const)      2 linkCounts (const)
    //   3 partials (write)    4 cellCounter        5 params
    kernel void statsKernel(device const float3* positions [[buffer(0)]],
                            device const uint*   links      [[buffer(1)]],
                            device const uint*   linkCounts [[buffer(2)]],
                            device float2*       partials   [[buffer(3)]],
                            device const uint*   cellCounter [[buffer(4)]],
                            constant SimParams&  params     [[buffer(5)]],
                            uint gid  [[thread_position_in_grid]],
                            uint ltid [[thread_position_in_threadgroup]],
                            uint tg   [[threadgroup_position_in_grid]])
    {
        threadgroup float2 shared[64];
        float2 v = float2(0.0);
        if (gid < cellCounter[0]) {
            float3 P = float3(positions[gid]);
            uint   n = linkCounts[gid];
            uint base = gid * params.maxValence;
            float s = 0.0;
            for (uint k = 0; k < n; ++k) {
                s += distance(float3(positions[links[base + k]]), P);
            }
            v = float2(s, float(n));
        }
        shared[ltid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 32; off > 0; off >>= 1) {
            if (ltid < off) shared[ltid] += shared[ltid + off];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (ltid == 0) partials[tg] = shared[0];
    }

    // Zero a float3 buffer prefix (clears collisionOffset before the probe's
    // accumulate pass; dispatch with exactly the entry count).
    kernel void zeroFloat3(device float3* buf [[buffer(0)]],
                           uint gid [[thread_position_in_grid]])
    {
        buf[gid] = float3(0.0);
    }

    // ---- TEMPORARY DEBUG KERNELS (CELLDBG diagnostics; remove after the M3
    // ---- validation gates are green) --------------------------------------

    // Ground truth for the params buffer layout (STAP 0): echoes the fields
    // the kernels actually see. uint fields directly; float fields bit-cast.
    //
    // Buffer table:
    //   0 out (write, 10 x uint32)   1 params
    kernel void paramEcho(device uint* out        [[buffer(0)]],
                          constant SimParams& params [[buffer(1)]],
                          uint gid [[thread_position_in_grid]])
    {
        if (gid != 0u) return;
        out[0] = params.gridBitsX;
        out[1] = params.gridBitsY;
        out[2] = params.gridBitsZ;
        out[3] = params.maxValence;
        out[4] = params.cellCount;
        out[5] = params.maxCells;
        out[6] = params.divisionEnabled;
        out[7] = params.rngSeed;
        out[8] = as_type<uint>(params.roi);
        out[9] = as_type<uint>(params.repulsionStrength);
    }

    // STAP 1 hash fingerprint for cell 0: the 27 hash values repulsionAccumulate
    // computes (same binHash, same loop order), the unique-hash count, the
    // total candidate visits, and the candidates passing distance+linked.
    // Discriminator: 9 unique hashes => dead axis in the hash (Groep 1);
    // 27 unique but 3x field => triple accumulation (Groep 2);
    // 27 unique and 1x field but wrong => scan/measurement (Groep 3/4).
    //
    // Buffer table:
    //   0 positions (const)       1 binStarts (const)       2 particleIndices (const)
    //   3 links (const)           4 linkCounts (const)      5 out (write, 30 x uint32)
    //   6 cellCounter             7 params
    kernel void queryFingerprint(device const float3* positions [[buffer(0)]],
                                 device const uint*   binStarts [[buffer(1)]],
                                 device const uint*   particleIndices [[buffer(2)]],
                                 device const uint*   links     [[buffer(3)]],
                                 device const uint*   linkCounts [[buffer(4)]],
                                 device uint*         out       [[buffer(5)]],
                                 device const uint*   cellCounter [[buffer(6)]],
                                 constant SimParams&  params    [[buffer(7)]],
                                 uint gid [[thread_position_in_grid]])
    {
        if (gid != 0u) return;
        float3 P = float3(positions[0]);
        float  roi = params.roi;
        int3   bc = int3((int)floor(P.x / roi), (int)floor(P.y / roi), (int)floor(P.z / roi));
        uint   n = linkCounts[0];
        uint   base = 0u * params.maxValence;

        uint seen[27];
        uint nUnique = 0u;
        uint visits  = 0u;
        uint inRoi   = 0u;
        uint w = 0u;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint h = binHash(bc + int3(dx, dy, dz), params);
                    out[w] = h;
                    ++w;
                    bool dup = false;
                    for (uint t = 0; t < nUnique; ++t) {
                        if (seen[t] == h) { dup = true; break; }
                    }
                    if (!dup) { seen[nUnique++] = h; }
                    uint s = binStarts[h];
                    uint e = binStarts[h + 1];
                    visits += e - s;
                    for (uint idx = s; idx < e; ++idx) {
                        uint r = particleIndices[idx];
                        if (r == 0u) continue;
                        float3 D = P - float3(positions[r]);
                        float  d2 = dot(D, D);
                        if (d2 > roi * roi) continue;
                        bool linked = false;
                        for (uint k = 0; k < n; ++k) {
                            if (links[base + k] == r) { linked = true; break; }
                        }
                        if (linked) continue;
                        ++inRoi;
                    }
                }
            }
        }
        out[27] = nUnique;
        out[28] = visits;
        out[29] = inRoi;
    }
    """

    // MARK: - `renderKernels`: raster impostor shaders (solid + interactive X-ray)

    static let renderKernels = """

    // Layout must mirror Swift struct CameraUniforms (240 bytes: 208 + the
    // V2.1 xrayPad float4 + the V2.1b foamPad/eyePad float4s).
    struct Camera {
        float4x4 view;
        float4x4 proj;
        float4 right;    // world-space camera right (unit)
        float4 up;       // world-space camera up (unit)
        float4 negFwd;   // world-space direction from target toward the camera
        float4 lightDir; // world-space unit vector pointing TOWARD the light
        float4 xrayPad;  // M6: .x = interactive X-ray exposure gain (solid path never reads it)
        float4 foamPad;  // V2.1b: .x = foam opacity (0 = the historical solid
                         // path — cellFragment branches on it; > 0 selects the
                         // foam chamber CSG). Replaces the retired V2.1 edgePad.
        float4 eyePad;   // V2.1b: .xyz = world eye position — the foam CSG runs
                         // in WORLD space (positions/links are world) and needs
                         // the ray origin; .w unused
    };

    // SimParams comes from the shared `common` chunk above — bound to the
    // SAME params buffer as the sim (the cellFragment edge layer reads
    // maxValence for the link stride). Appended-only per AGENTS.md.

    struct VOut {
        float4 clip    [[position]];
        float2 uv;       // [-1,1] billboard coordinates
        float3 centerV;  // view-space sphere center
        float3 right;
        float3 up;
        float3 negFwd;
        float  radius;
        float3 tint;     // M5: per-cell base colour (cellTint kernel; neutral albedo unless rde mode)
        uint  cellId;    // V2.1: instance index (flat) — the fragment reads the cell's ring
    };

    // One camera-facing quad per cell; the fragment shader shades it as a
    // sphere (solid) or as the cell's foam CHAMBER (foam — see cellFragment).
    vertex VOut cellVertex(uint vid [[vertex_id]],
                           uint iid [[instance_id]],
                           device const float3* positions [[buffer(0)]],
                           device const float* radii             [[buffer(1)]],
                           constant Camera& cam                  [[buffer(2)]],
                           device const float3* tint             [[buffer(3)]])
    {
        const float2 corners[4] = { float2(-1.0, -1.0), float2(1.0, -1.0),
                                    float2(-1.0,  1.0), float2(1.0,  1.0) };
        float2 uv = corners[vid];
        float3 P  = float3(positions[iid]);
        float  r  = radii[iid];

        float3 cV = (cam.view * float4(P, 1.0)).xyz;

        VOut o;
        o.uv      = uv;
        o.centerV = cV;
        o.radius  = r;
        o.right   = cam.right.xyz;
        o.up      = cam.up.xyz;
        o.negFwd  = cam.negFwd.xyz;
        o.tint    = float3(tint[iid]);
        o.cellId  = iid;
        o.clip    = cam.proj * float4(cV + float3(uv * r, 0.0), 1.0);
        return o;
    }

    // V2.1b foam vertex: identical to cellVertex except the instance -> cell
    // mapping goes through the drawOrder buffer. The foam pipeline composites
    // premultiplied-alpha chamber faces with NO depth test — the CPU keeps
    // drawOrder sorted back-to-front (farthest instance first) so the fixed-
    // function over-blend is the correct front-to-back composite. Buffer
    // table: 0 positions   1 radii   2 camera   3 tint   4 drawOrder (u32
    // per instance, cell index).
    vertex VOut cellVertexFoam(uint vid [[vertex_id]],
                               uint iid [[instance_id]],
                               device const float3* positions [[buffer(0)]],
                               device const float* radii             [[buffer(1)]],
                               constant Camera& cam                  [[buffer(2)]],
                               device const float3* tint             [[buffer(3)]],
                               device const uint*   drawOrder        [[buffer(4)]])
    {
        const float2 corners[4] = { float2(-1.0, -1.0), float2(1.0, -1.0),
                                    float2(-1.0,  1.0), float2(1.0,  1.0) };
        float2 uv = corners[vid];
        uint cell = drawOrder[iid];
        float3 P  = float3(positions[cell]);
        float  r  = radii[cell];

        float3 cV = (cam.view * float4(P, 1.0)).xyz;

        VOut o;
        o.uv      = uv;
        o.centerV = cV;
        o.radius  = r;
        o.right   = cam.right.xyz;
        o.up      = cam.up.xyz;
        o.negFwd  = cam.negFwd.xyz;
        o.tint    = float3(tint[cell]);
        o.cellId  = cell;
        o.clip    = cam.proj * float4(cV + float3(uv * r, 0.0), 1.0);
        return o;
    }

    // ---- M6: interactive X-ray (additive impostor accumulation) ------------
    //
    // Approximation of the offline X-ray density accumulation (paper §6 style
    // 2) for the realtime path: every cell draws ONE camera-facing density
    // sprite with ADDITIVE blending (blend one/one, no depth test, no depth
    // write) — overlapping sprites sum, so dense knots glow and the
    // background stays dark. The per-sprite profile is the SAME radial
    // profile the ray tracer accumulates (rad·(1-(d/(rad·S))²)², S =
    // profileSupport = 2) with the quad enlarged S× so the skirt is drawn.
    //
    // Known approximation (documented): the film curve L = 1-exp(-Σd·e) is
    // nonlinear in the ACCUMULATED density, but fixed-function additive
    // blending can only sum per-sprite terms in 8-bit sRGB — the sum is then
    // clamped at 1.0 (a hard saturating "film"). Per-sprite the contribution
    // is premultiplied by an exposure gain so the mid densities land in the
    // same tonal range as the offline render at the default exposure 0.1.
    // Per-sprite output = tint·min(dens·gain, 1); the 8-bit clamp provides
    // the tone map. Echoes the offline look (dark background, glowing
    // density); not pixel-equivalent to the ray tracer by design.

    // Enlarged density sprite: quad half-width = radius × profileSupport.
    // Same buffer table as cellVertex.
    vertex VOut cellVertexXray(uint vid [[vertex_id]],
                               uint iid [[instance_id]],
                               device const float3* positions [[buffer(0)]],
                               device const float* radii             [[buffer(1)]],
                               constant Camera& cam                  [[buffer(2)]],
                               device const float3* tint             [[buffer(3)]])
    {
        const float support = 2.0;   // matches RayRenderSettings.profileSupport default
        const float2 corners[4] = { float2(-1.0, -1.0), float2(1.0, -1.0),
                                    float2(-1.0,  1.0), float2(1.0,  1.0) };
        float2 uv = corners[vid];
        float3 P  = float3(positions[iid]);
        float  r  = radii[iid];

        float3 cV = (cam.view * float4(P, 1.0)).xyz;

        VOut o;
        o.uv      = uv;
        o.centerV = cV;
        o.radius  = r;
        o.right   = cam.right.xyz;
        o.up      = cam.up.xyz;
        o.negFwd  = cam.negFwd.xyz;
        o.tint    = float3(tint[iid]);
        o.cellId  = iid;
        o.clip    = cam.proj * float4(cV + float3(uv * r * support, 0.0), 1.0);
        return o;
    }

    // Additive density fragment. uv here spans [-support, support] (the
    // enlarged quad), so d/(rad·S) = |uv| exactly, and the profile value is
    // rad·(1-|uv|²)² — identical math to the ray tracer's per-sphere
    // contribution. Blended additively (one/one); alpha carries the same
    // weight for the (unused) alpha channel.
    fragment float4 cellFragmentXray(VOut in [[stage_in]],
                                     constant Camera& cam [[buffer(0)]])
    {
        float q = 1.0 - dot(in.uv, in.uv);   // 1 - (d/(rad·S))²
        if (q <= 0.0) { discard_fragment(); }
        float dens = in.radius * q * q;      // rad·q² — the offline profile
        float gain = cam.xrayPad.x;
        float a = min(dens * gain, 1.0);
        return float4(in.tint * a, a);       // premultiplied; blend one/one
    }

    struct FOut {
        float4 color [[color(0)]];
        float  depth [[depth(any)]];
    };

    // Solid / foam impostor fragment.
    //
    // SOLID (cam.foamPad.x == 0 — the historical path, M3-golden-locked):
    // V2.1 per-cell FRACTIONAL illumination (litFraction, written by the
    // illuminate kernel) scales the shaded colour; neutral 1.0 outside light
    // mode / before the first march. 0.25 keeps fully shaded cells readable.
    //
    // FOAM (cam.foamPad.x > 0 — V2.1b): the fragment shades the cell's
    // POWER-DIAGRAM CHAMBER (see the foam helper block in `common`) instead
    // of the full sphere. The pixel's exact view ray is intersected with the
    // sphere and clipped against the cell's ring-neighbour power bisectors;
    // the first face along the ray is shaded — a curved CAP (sphere normal,
    // the cell's own tint) where the cell faces open space, or a flat WALL
    // (canonical bisector normal flipped toward the camera, the AVERAGE tint
    // and illumination of the two cells sharing the wall) where it presses a
    // neighbour. Chamber-empty pixels discard, revealing the cell behind.
    // Output is PREMULTIPLIED by the foam opacity and blended over (one /
    // one-minus-source-alpha) with the instances sorted back-to-front — the
    // front-to-back chamber composite ("walls behind walls", soap-froth
    // depth). The chamber entry face of the nearest chamber is always a cap
    // (see the visibility fact in `common`); the flat walls show THROUGH the
    // translucent front caps and in the cut-away regions.
    //
    // Fragment buffer table:
    //   0 camera (constant)   1 positions (const)   2 links (const)
    //   3 linkCounts (const)  4 litFraction (const)  5 SimParams (maxValence stride)
    //   6 tint (const; V2.1b foam wall colour = the two cells' average)
    //   7 radii (const; V2.1b foam CSG needs the neighbours' radii)
    fragment FOut cellFragment(VOut in [[stage_in]],
                               constant Camera& cam [[buffer(0)]],
                               device const float3* positions [[buffer(1)]],
                               device const uint*   links [[buffer(2)]],
                               device const uint*   linkCounts [[buffer(3)]],
                               device const float*  litFraction [[buffer(4)]],
                               constant SimParams&  params [[buffer(5)]],
                               device const float3* tint [[buffer(6)]],
                               device const float*  radii [[buffer(7)]])
    {
        if (cam.foamPad.x > 0.0) {
            // Exact view ray through this fragment's quad point (view space:
            // camera at the origin looking down -Z; the quad point is where
            // the pixel's ray crosses the centre-depth plane).
            float3 rdV = normalize(in.centerV + float3(in.uv * in.radius, 0.0));
            // Same ray in world space (the CSG works on world positions).
            float3 ro = cam.eyePad.xyz;
            float3 rd = normalize(in.right * rdV.x + in.up * rdV.y + in.negFwd * rdV.z);
            float r = in.radius;
            float3 Cw = float3(positions[in.cellId]);
            float3 toC = Cw - ro;
            float tc = dot(toC, rd);
            float dL2 = dot(toC, toC) - tc * tc;
            if (dL2 > r * r) { discard_fragment(); }   // outside the sphere disc
            float e, x;
            uint wallJ;
            if (!foamChamberInterval(ro, rd, Cw, r, tc, dL2, in.cellId,
                                     positions, radii, links, linkCounts,
                                     params.maxValence, &e, &x, &wallJ)) {
                discard_fragment();   // chamber cut away along this ray
            }
            float3 PhW = ro + rd * e;             // world hit point
            float3 N;                             // world shading normal
            float3 tintCol;
            float litV;
            if (wallJ == 0xFFFFFFFFu) {
                N = normalize(PhW - Cw);          // curved cap
                tintCol = in.tint;
                litV = litFraction[in.cellId];
            } else {
                float3 nh = foamWallNormal(in.cellId, wallJ, positions);
                N = dot(nh, rd) > 0.0f ? -nh : nh;   // face the camera
                tintCol = 0.5f * (in.tint + float3(tint[wallJ]));
                litV = 0.5f * (litFraction[in.cellId] + litFraction[wallJ]);
            }
            // Same shading form as the solid path, with the FACE normal.
            float3 nV = float3(dot(N, in.right), dot(N, in.up), dot(N, in.negFwd));
            float diff = max(dot(N, cam.lightDir.xyz), 0.0);
            float rim  = pow(1.0 - nV.z, 2.0);
            float3 col = tintCol * (0.24 + 0.76 * diff);
            col *= (0.25 + 0.75 * litV);
            col *= 1.0 - 0.35 * rim;
            // Depth from the exact hit point (view space, ro = 0).
            float4 clip = cam.proj * float4(rdV * e, 1.0);
            FOut o;
            o.color = float4(cam.foamPad.x * col, cam.foamPad.x);   // premultiplied over
            o.depth = clip.z / clip.w;
            return o;
        }

        // --- historical solid path (unchanged; foamPad.x == 0) ---
        float z2 = 1.0 - dot(in.uv, in.uv);
        if (z2 < 0.0) { discard_fragment(); }   // outside the sphere disc
        float z = sqrt(max(z2, 0.0));

        // Sphere normal in view space, mapped to world via the camera basis.
        float3 nV = normalize(float3(in.uv, z));
        float3 nW = normalize(in.right * nV.x + in.up * nV.y + in.negFwd * nV.z);

        float diff = max(dot(nW, cam.lightDir.xyz), 0.0);
        float rim  = pow(1.0 - nV.z, 2.0);
        // M5: per-cell tint (the cellTint kernel writes the neutral albedo
        // (0.87, 0.85, 0.82) outside rde mode — bit-identical to the previous
        // hardcoded constant, so the M3 goldens are unaffected). The light
        // direction is the SHARED growth light (Renderer fills cam.lightDir
        // from SimParams.lightDirX/Y/Z).
        float3 col = in.tint * (0.24 + 0.76 * diff);
        // V2.1: per-cell fractional illumination as the shadow factor
        // (exactly 1.0 when litFraction == 1.0 — the neutral fill).
        col *= (0.25 + 0.75 * litFraction[in.cellId]);
        col *= 1.0 - 0.35 * rim;

        // True sphere-surface point (view space) -> exact fragment depth.
        float4 clip = cam.proj * float4(in.centerV + in.radius * nV, 1.0);

        FOut o;
        o.color = float4(col, 1.0);
        o.depth = clip.z / clip.w;
        return o;
    }
    """

    // MARK: - `raytraceKernels`: offline ray tracer (M4; V2.1 edges + shadow rays)

    static let raytraceKernels = """

    // ---- M4: offline ray-traced renderer (compute path) --------------------
    //
    // Rays traverse the simulation's OWN spatial hash (CellSim grid: bin size
    // = roi, bit-packed per-axis hash, built ONCE for the final state by
    // CellSim.encodeGridRebuild — the per-step rebuild lives inside step()
    // only). One thread per output pixel; the image is produced in tiles
    // (RayRenderer.tileSize) to bound intermediate memory (one tile texture,
    // AA samples accumulate per-thread in registers).
    //
    // Style 1 — solid (paper §5/§6 style 1, Fig. 8): nearest sphere hit along
    //   the ray (cells treated as spheres, radii = mean link distance x
    //   radiusScale, paper §5). Shading is Lambertian reflectance under a
    //   UNIFORM OMNIDIRECTIONAL diffuse light: with an isotropic light field
    //   the Lambertian irradiance integral E = ∫ V(ω) max(N·ω, 0) dω / π
    //   reduces to a pure visibility quantity — directionally unbiased by
    //   construction (no preferred light direction; this is exactly the
    //   paper's "'ambient occlusion' ... self-shadowing from a uniform
    //   omnidirectional diffuse light"). The occlusion is estimated
    //   ANALYTICALLY from neighbours inside a query radius: a sphere at
    //   distance d (radius rad) subtends half-angle α with sin α = rad/d, and
    //   the cosine-weighted occluded irradiance fraction of that cone (when
    //   it lies fully above the tangent plane) is sin²α·max(N·Ĉ, 0) =
    //   (rad²/d²)·max(N·Ĉ, 0) (for a cone fully inside the upper hemisphere,
    //   ∫_cone (N·ω) dω = π sin²α (N·Ĉ)). Summed over neighbours; overlapping
    //   cones over-count, compensated by the aoStrength knob. Chosen over
    //   hemisphere ray sampling: noise-free at any sample budget (the paper
    //   look is smooth) and ~30x cheaper than 32 traced rays/pixel (M4
    //   report documents the choice).
    //
    // Style 2 — X-ray (paper §5/§6 style 2, Fig. 9): the ray is traced
    //   through the WHOLE structure and every sphere near it contributes to
    //   an accumulated density. Per sphere: smooth radial profile
    //   rad·(1-(d/(rad·S))²)² for d < rad·S (S = profileSupport, default 2),
    //   d = distance from the sphere centre to the ray SEGMENT (C¹ at the
    //   support edge). The paper does not specify the per-sphere density
    //   profile; a hard chord integral (S = 1) shows visible per-sphere arcs
    //   at these cell counts while the paper's X-rays are smooth/cloudy —
    //   S widens the blobs so neighbouring cells blend (artistic choice,
    //   documented in the M4 report). Accumulated along the ray, tone-mapped
    //   with a film curve L = 1 - exp(-x·exposure), optional display gamma,
    //   to grayscale.
    //
    // Grid-traversal correctness (the sim's h >= roi rule does not carry over
    // directly): the 3D-DDA (Amanatides–Woo) walks the bins on the ray
    // CENTERLINE; per visited bin the 27-bin neighbourhood is queried. Proof
    // every relevant sphere is found: if the ray crosses a sphere (centre P,
    // radius rad < roi), the crossing point X lies within rad < h of P, so
    // P's own bin is inside the 3x3x3 neighbourhood of bin(X), and bin(X) is
    // on the centerline, hence visited. The packed hash keeps the 27 lookups
    // of one bin distinct (same argument as repulsionAccumulate). A sphere
    // may then be found from several visited bins: idempotent for the
    // nearest-hit style; for the DENSITY accumulation each sphere is counted
    // exactly ONCE — at the visited bin whose [tEnter, tExit) range contains
    // the sphere's ray entry parameter ta. The per-bin ranges partition
    // [t0, t1] by construction (tExit(b) == tEnter(next b)), so the anchor
    // selects exactly one bin regardless of fp tie-breaking at plane ties.
    //
    // Requires cell radius < roi for the GRID path: a larger radius could put
    // a crossing point more than one bin away per axis and miss spheres.
    // Grown structures DO contain stretched cells (radius up to ~0.5 x max
    // link ≈ 5 > roi 1.6), so the renderer splits the cell set: radii < roi
    // are traversed through the grid (the proof above), radii >= roi go into
    // a host-built BIG-CELL supplement list (indices only) that every ray
    // brute-forces ONCE (few cells: typically << 1% of the structure; empty
    // for pristine balls ⇒ zero cost). The two sets are disjoint, so density
    // contributions add without double counting and the nearest hit is the
    // min over both. The scene bounding sphere (host, inflated by the max
    // cell radius) also guarantees every sphere's ray entry lies strictly
    // inside [t0, t1].

    // Layout must mirror Swift struct RayRenderParams (Render/RayRenderer.swift).
    // float4 members on BOTH sides: no float3-in-struct alignment ambiguity.
    // Render-only uniform block — deliberately NOT appended to SimParams
    // (that struct is the simulation contract, appended-only; render tuning
    // parameters change freely here and are versioned with the goldens).
    // M5: lightDir/directionalLight implement the UNIFIED LIGHT — in
    // directional mode the solid style shades with the same light vector the
    // growth illumination uses (SimParams.lightDirX/Y/Z is copied into
    // lightDir by the host), showing which cells receive light. The default
    // (directionalLight = 0) keeps the paper's directionally unbiased
    // omnidirectional look; the M4 goldens are byte-identical under it.
    struct RenderParams {
        uint  imageW, imageH, tileX, tileY;
        uint  tileW, tileH, samples, style;      // style: 0 = solid, 1 = xray
        float aoRadius, aoStrength, aoFloor, xrayExposure;
        float xrayGamma, tanHalfFov, boundingRadius;
        uint  renderSeed;                        // NOTE: uint — was once in the float group above (M4 bug: bit pattern read as ~0 float)
        float densityScale, profileSupport;      // xray profile support (x cell radius)
        uint  bigCount;                          // big-cell supplement list length (rad >= roi)
        uint  pad2;
        float4 background;
        float4 camEye;
        float4 camRight;    // unit basis, same conventions as raster Camera
        float4 camUp;
        float4 camFwd;      // unit vector eye -> target
        float4 boundingCentre;
        float4 albedo;      // RETIRED M5: per-cell tint buffer is the solid-style albedo (cellTint kernel); field kept for layout/meta compatibility
        float4 lightDir;    // M5: shared growth light direction (unit, toward the light)
        uint  directionalLight;  // M5: 0 = unbiased omnidirectional (default), 1 = shared directional light
        uint  pad3;
        // --- V2.1 (appended; struct grows 216 -> 240 bytes) ---
        float shadows;           // per-pixel shadow strength in directional mode (0 = per-cell fraction only)
        uint  shadowSamples;     // jittered shadow rays per hit pixel (4-8; cost ~ linear)
        uint  xrayColored;       // 0 = legacy grayscale density (bit-identical), 1 = per-channel tinted density
        uint  pad4;
        // --- V2.1b (appended; struct grows 240 -> 248 bytes; the retired V2.1
        // edges/edgeWidth fields were REMOVED — the foam chamber walls are the
        // wall visualization now) ---
        float foamOpacity;       // per-face opacity of the foam composite (solid foam)
        uint  foamLayers;        // max composited chamber faces per ray (<= 8, clamped host-side)
        float foamWall;          // foam-xray wall-crossing spike gain
        uint  pad5;
    };

    // SimParams and the grid/march helpers (binHash, coarseHash,
    // coarseNeighbourhoodClear, lightVisibility) come from the shared
    // `common` chunk — this library binds the SIM's own params buffer, so
    // gridBits*/gridCoarseBits*/roi/lightSoft all match by construction
    // (the former verbatim copies are single-sourced there). The per-pixel
    // shadow rays call lightVisibility with lightSoft = 0 (hard per-sample
    // occlusion); the cone jitter provides the penumbra.

    // Deterministic per-pixel/per-sample AA jitter (PCG-style integer hash).
    static inline uint jitterHash(uint x) {
        uint state = x * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
    static inline float2 sampleJitter(uint px, uint py, uint s, uint seed) {
        uint h = jitterHash(px * 1973u ^ py * 9277u ^ s * 26699u ^ seed * 7919u);
        return float2((float)(h & 0xFFFFu) / 65535.0f - 0.5f,
                      (float)((h >> 16u) & 0xFFFFu) / 65535.0f - 0.5f);
    }

    // V2.1: jittered direction inside the cone of half-angle `soft` around
    // the light direction L (small-angle: sin ~ tan ~ soft). soft = 0 gives
    // L exactly. u = 2 uniform samples in [0, 1).
    static inline float3 coneJitter(float3 L, float soft, float2 u) {
        if (soft <= 1e-6f) return L;
        float3 helper = (fabs(L.y) < 0.99f) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
        float3 t1 = normalize(cross(L, helper));
        float3 t2 = cross(L, t1);
        return normalize(L + soft * (u.x * t1 + u.y * t2));
    }

    // V2.1b foam-solid: one composited chamber face along a ray. t = the
    // chamber's ENTRY parameter; idx = the chamber's cell; wallJ = the ring
    // neighbour whose power bisector produced the entry bound (a flat WALL
    // entry — see foamChamberInterval), or 0xFFFFFFFF for a curved CAP entry
    // (the union boundary; always a cap — see the visibility fact in `common`).
    struct FoamFace {
        float t;
        uint  idx;
        uint  wallJ;
    };

    // Insert into the top-K nearest-entry list, kept sorted ascending by t
    // (K = maxFaces <= 8 — the composite's layer budget). Called per candidate
    // AFTER the exactly-once anchor check, so a chamber enters the list at
    // most once per ray.
    static void foamFaceInsert(thread FoamFace* faces, thread uint* faceCount,
                               uint maxFaces, float t, uint idx, uint wallJ) {
        if (*faceCount < maxFaces) {
            uint p = (*faceCount)++;
            while (p > 0u && faces[p - 1u].t > t) { faces[p] = faces[p - 1u]; --p; }
            faces[p] = FoamFace{ t, idx, wallJ };
        } else if (t < faces[maxFaces - 1u].t) {
            uint p = maxFaces - 1u;
            while (p > 0u && faces[p - 1u].t > t) { faces[p] = faces[p - 1u]; --p; }
            faces[p] = FoamFace{ t, idx, wallJ };
        }
    }

    // Traversal over the hashed grid along the segment [t0, t1] of the ray
    // ro + t*rd (see the header comment for the correctness argument).
    // wantNearest: writes the nearest sphere entry parameter + its cell index
    //   (re-testing a candidate from several of the neighbourhoods is
    //   idempotent — min). Sphere test: dLine2 <= rad².
    // wantDensity: accumulates per-sphere profiles with the entry-anchor
    //   dedup (each sphere counted at exactly one visited bin — the bin whose
    //   [tCur, tExitC) range contains the clamped foot tf). The density
    //   filter is the PROFILE SUPPORT (rad·profileSupport), NOT the sphere
    //   test: the profile is C¹ at the support edge, so a cell's contribution
    //   vanishes continuously there — filtering at rad² instead would make
    //   skirt cells (perp distance in (rad, support]) jump discontinuously
    //   between 0 and ~0.75·rad on 1-ulp dLine2 differences (measured: 0.26
    //   density flicker per affected ray). The support filter needs a wider
    //   neighbourhood: the closest segment point can be support away from
    //   the centre, i.e. profileSupport bins per axis for rad < h — the
    //   caller passes nbhdK = wantDensity ? ceil(profileSupport) : 1.
    // V2.1b style 3 (foam-xray): the density foot clamps to the cell's CHAMBER
    //   interval (foamChamberInterval — needs the link arrays) instead of the
    //   scene segment, so the density follows the power-diagram chambers
    //   (skirt cells contribute nothing), and every plane-raised chamber
    //   entry — a WALL CROSSING — adds a thin obliquity-weighted density
    //   spike (rp.foamWall): internal membranes in the X-ray. Style 1 keeps
    //   the M4 accumulation bit-identical.
    // V2.1b wantFoam (style 2): per candidate the CHAMBER entry (foam CSG,
    //   needs the link arrays) is inserted into the top-K faces list — the
    //   exactly-once anchor is the entry parameter e itself (the entry point
    //   — cap or wall — always lies within rad of the centre, so the centre's
    //   bin is in the anchor bin's 27-bin neighbourhood and e lies in exactly
    //   one bin range of the partition).
    static void traverseGrid(float3 ro, float3 rd, float t0, float t1, uint count,
                             device const float3* positions,
                             device const float* radii,
                             device const uint* links,
                             device const uint* linkCounts,
                             device const uint* binStarts,
                             device const uint* particleIndices,
                             constant SimParams& sp,
                             float profileSupport, uint nbhdK,
                             constant RenderParams& rp,
                             device const float3* tint,
                             bool wantNearest, thread float* nearest, thread uint* nearestIdx,
                             bool wantDensity, thread float3* density,
                             bool wantFoam, thread FoamFace* faces, thread uint* faceCount,
                             uint maxFaces)
    {
        if (count == 0u || !(t1 > t0)) return;
        const float h = sp.roi;
        float3 p0 = ro + rd * t0;
        int3 c = int3((int)floor(p0.x / h), (int)floor(p0.y / h), (int)floor(p0.z / h));
        int3 st = int3(rd.x > 0.0f ? 1 : -1, rd.y > 0.0f ? 1 : -1, rd.z > 0.0f ? 1 : -1);
        // +inf where rd == 0: that axis never advances (IEEE division).
        float3 tDelta = fabs(float3(h) / rd);
        float3 tMax;
        tMax.x = rd.x != 0.0f ? t0 + ((rd.x > 0.0f ? (float)(c.x + 1) * h : (float)c.x * h) - p0.x) / rd.x : INFINITY;
        tMax.y = rd.y != 0.0f ? t0 + ((rd.y > 0.0f ? (float)(c.y + 1) * h : (float)c.y * h) - p0.y) / rd.y : INFINITY;
        tMax.z = rd.z != 0.0f ? t0 + ((rd.z > 0.0f ? (float)(c.z + 1) * h : (float)c.z * h) - p0.z) / rd.z : INFINITY;

        float tCur = t0;
        // Safety bound: the segment can never cross more bins than
        // (scene diameter / h) + diagonal slack; 8192 is orders above that
        // for any structure the sim can produce. Guards against a GPU hang
        // on pathological input (a hung kernel stalls the whole machine).
        for (uint iter = 0; iter < 8192u && tCur < t1; ++iter) {
            float tExit = min(min(tMax.x, tMax.y), tMax.z);
            float tExitC = min(tExit, t1);   // this bin's range: [tCur, tExitC)

            for (int dz = -(int)nbhdK; dz <= (int)nbhdK; ++dz) {
                for (int dy = -(int)nbhdK; dy <= (int)nbhdK; ++dy) {
                    for (int dx = -(int)nbhdK; dx <= (int)nbhdK; ++dx) {
                        uint hh = binHash(c + int3(dx, dy, dz), sp);
                        uint s = binStarts[hh];
                        uint e = binStarts[hh + 1];
                        for (uint idx = s; idx < e; ++idx) {
                            uint r = particleIndices[idx];
                            float rad = radii[r];
                            // Strict split of responsibilities with
                            // bigCellPass: the grid path proves correctness
                            // only for rad < h (crossing point within nbhdK
                            // bins of the centre bin); bigger cells are the
                            // supplement pass's exclusive set. Testing them
                            // here too would double-count their density.
                            if (rad <= 0.0f || rad >= h) continue;
                            float support = rad * profileSupport;
                            float3 C = float3(positions[r]);
                            float3 toC = C - ro;
                            float tc = dot(toC, rd);
                            float dLine2 = dot(toC, toC) - tc * tc;
                            float r2test = wantDensity ? support * support : rad * rad;
                            if (dLine2 > r2test) continue;   // outside this pass's domain
                            float thc = sqrt(max(rad * rad - dLine2, 0.0f));
                            float ta = tc - thc;
                            float tb = tc + thc;
                            if (tb <= t0 || ta >= t1) continue; // outside segment
                            if (wantNearest && dLine2 <= rad * rad) {
                                if (ta > t0 && ta < *nearest) {
                                    *nearest = ta;
                                    *nearestIdx = r;
                                }
                            }
                            if (wantDensity) {
                                // Count exactly once: at the visited bin whose
                                // [tCur, tExitC) range contains the clamped
                                // foot tf (well-defined for every cell —
                                // skirt cells included — and strictly inside
                                // [t0, t1] thanks to the bounding inflation).
                                // V2.1b foam-xray (style 3): the foot clamps
                                // to the CHAMBER interval — no chamber along
                                // the ray, no contribution — and every
                                // plane-raised entry (a WALL CROSSING) adds
                                // the membrane spike below. The chamber-
                                // clamped foot is per-candidate, so the
                                // anchor stays exactly-once (one bin range
                                // contains it, and the candidate is walked
                                // at that bin because its entry point lies
                                // within rad of the centre — the same proof
                                // as the sphere-test walk).
                                float tf;
                                uint wallJ = 0xFFFFFFFFu;
                                if (rp.style == 3u) {
                                    if (dLine2 > rad * rad) continue;   // no sphere -> no chamber
                                    float e, x;
                                    if (!foamChamberInterval(ro, rd, C, rad, tc, dLine2, r,
                                                             positions, radii,
                                                             links, linkCounts, sp.maxValence,
                                                             &e, &x, &wallJ)) continue;
                                    tf = clamp(tc, e, x);
                                } else {
                                    tf = clamp(tc, t0, t1);
                                }
                                if (tf >= tCur && tf < tExitC) {
                                    // Distance from the centre to the SEGMENT
                                    // (clamped foot): rim spheres beyond the
                                    // segment ends fall off naturally.
                                    float3 dv = C - (ro + rd * tf);
                                    float d2 = dot(dv, dv);
                                    if (d2 < support * support) {
                                        float q = 1.0f - d2 / (support * support);
                                        float contrib = rad * q * q;
                                        // V2.1 coloured X-ray: per-channel
                                        // tinted density. Grayscale (0) keeps
                                        // the M4 accumulation bit-identical
                                        // (multiply by exactly (1,1,1)).
                                        float3 w3 = (rp.xrayColored != 0u) ? float3(tint[r]) : float3(1.0f, 1.0f, 1.0f);
                                        *density += w3 * contrib;
                                        // V2.1b foam-xray wall crossing: thin-
                                        // film density with the grazing
                                        // obliquity factor 1/|rd·n̂| — a sheet's
                                        // projected optical depth grows as it is
                                        // seen edge-on, which is what draws the
                                        // internal membrane LINES (clamped at 4x
                                        // against near-tangent rays). Coloured
                                        // X-ray tints the membrane with the two
                                        // cells' average (the wall is shared).
                                        if (rp.style == 3u && wallJ != 0xFFFFFFFFu) {
                                            float3 nh = foamWallNormal(r, wallJ, positions);
                                            float obliq = min(1.0f / max(fabs(dot(rd, nh)), 0.25f), 4.0f);
                                            float3 wSpike = (rp.xrayColored != 0u)
                                                ? 0.5f * (float3(tint[r]) + float3(tint[wallJ]))
                                                : float3(1.0f, 1.0f, 1.0f);
                                            *density += wSpike * (rp.foamWall * rad * obliq);
                                        }
                                    }
                                }
                            }
                            // V2.1b foam-solid: the chamber's ENTRY face joins
                            // the top-K composite list — exactly once per ray
                            // at the bin whose range contains the entry
                            // parameter (see the function comment).
                            if (wantFoam && dLine2 <= rad * rad) {
                                float e, x;
                                uint wallJ;
                                if (foamChamberInterval(ro, rd, C, rad, tc, dLine2, r,
                                                        positions, radii,
                                                        links, linkCounts, sp.maxValence,
                                                        &e, &x, &wallJ)
                                    && e >= tCur && e < tExitC) {
                                    foamFaceInsert(faces, faceCount, maxFaces, e, r, wallJ);
                                }
                            }
                        }
                    }
                }
            }

            if (tExit >= t1) break;
            // Advance to the next bin (ties broken consistently; ranges stay
            // a partition of [t0, t1] either way).
            if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
                c.x += st.x; tCur = tMax.x; tMax.x += tDelta.x;
            } else if (tMax.y <= tMax.z) {
                c.y += st.y; tCur = tMax.y; tMax.y += tDelta.y;
            } else {
                c.z += st.z; tCur = tMax.z; tMax.z += tDelta.z;
            }
        }
    }

    // Big-cell supplement: cells with radius >= roi are outside the grid
    // traversal's correctness domain (see the header comment); every ray
    // tests them ONCE here (they are few — stretched cells only). Same
    // formulas as the grid path; the two sets are disjoint, so density adds
    // without double counting and the nearest hit is the min over both.
    // V2.1b: the foam-xray (style 3) chamber treatment matches traverseGrid;
    // the foam-solid (style 2) entries join the same top-K list (big cells
    // are enumerated exactly once per ray by construction — no anchor needed).
    static void bigCellPass(float3 ro, float3 rd, float t0, float t1,
                            device const uint* bigCells, uint bigCount,
                            device const float3* positions,
                            device const float* radii,
                            device const uint* links,
                            device const uint* linkCounts,
                            device const uint* binStarts,
                            device const uint* particleIndices,
                            constant SimParams& sp,
                            float profileSupport,
                            constant RenderParams& rp,
                            device const float3* tint,
                            bool wantNearest, thread float* nearest, thread uint* nearestIdx,
                            bool wantDensity, thread float3* density,
                            bool wantFoam, thread FoamFace* faces, thread uint* faceCount,
                            uint maxFaces)
    {
        for (uint bi = 0; bi < bigCount; ++bi) {
            uint r = bigCells[bi];
            float rad = radii[r];
            if (rad <= 0.0f) continue;
            float support = rad * profileSupport;
            float3 C = float3(positions[r]);
            float3 toC = C - ro;
            float tc = dot(toC, rd);
            float dLine2 = dot(toC, toC) - tc * tc;
            // Density filters at the profile support (C1 edge, see
            // traverseGrid); the nearest hit at the sphere test.
            float r2test = wantDensity ? support * support : rad * rad;
            if (dLine2 > r2test) continue;
            if (wantNearest && dLine2 <= rad * rad) {
                float thc = sqrt(max(rad * rad - dLine2, 0.0f));
                float ta = tc - thc;
                float tb = tc + thc;
                if (tb > t0 && ta < t1 && ta > t0 && ta < *nearest) {
                    *nearest = ta;
                    *nearestIdx = r;
                }
            }
            if (wantFoam && dLine2 <= rad * rad) {
                float e, x;
                uint wallJ;
                if (foamChamberInterval(ro, rd, C, rad, tc, dLine2, r,
                                        positions, radii,
                                        links, linkCounts, sp.maxValence,
                                        &e, &x, &wallJ) && e < t1) {
                    foamFaceInsert(faces, faceCount, maxFaces, e, r, wallJ);
                }
            }
            if (wantDensity) {
                float tf;
                uint wallJ = 0xFFFFFFFFu;
                if (rp.style == 3u) {
                    if (dLine2 > rad * rad) continue;   // no sphere -> no chamber
                    float e, x;
                    if (!foamChamberInterval(ro, rd, C, rad, tc, dLine2, r,
                                             positions, radii,
                                             links, linkCounts, sp.maxValence,
                                             &e, &x, &wallJ)) continue;
                    tf = clamp(tc, e, x);
                } else {
                    tf = clamp(tc, t0, t1);
                }
                float3 dv = C - (ro + rd * tf);
                float d2 = dot(dv, dv);
                if (d2 < support * support) {
                    float q = 1.0f - d2 / (support * support);
                    float contrib = rad * q * q;
                    float3 w3 = (rp.xrayColored != 0u) ? float3(tint[r]) : float3(1.0f, 1.0f, 1.0f);
                    *density += w3 * contrib;
                    if (rp.style == 3u && wallJ != 0xFFFFFFFFu) {
                        float3 nh = foamWallNormal(r, wallJ, positions);
                        float obliq = min(1.0f / max(fabs(dot(rd, nh)), 0.25f), 4.0f);
                        float3 wSpike = (rp.xrayColored != 0u)
                            ? 0.5f * (float3(tint[r]) + float3(tint[wallJ]))
                            : float3(1.0f, 1.0f, 1.0f);
                        *density += wSpike * (rp.foamWall * rad * obliq);
                    }
                }
            }
        }
    }

    // Analytic solid-angle ambient occlusion at surface point Ph with normal
    // N (formula in the header comment). Occluders = all spheres within
    // aoRadius of Ph, gathered from the grid with a cube walk
    // (2*(aoRadius/h) + 1 bins per axis).
    // V2.1b: selfA/selfB are EXCLUDED explicitly. The solid/cap callers pass
    // (0xFFFFFFFF, 0xFFFFFFFF) — the cell's own sphere already contributes
    // exactly zero at a cap point (cosT <= 0 by construction), so behaviour
    // is bit-identical to the pre-V2.1b form. Foam WALL callers pass the two
    // cells sharing the wall: the wall point lies INSIDE both spheres, and
    // without the exclusion they would read as two huge occluders sitting
    // behind the surface (the wall is their shared boundary — the material
    // behind a surface does not occlude it).
    static float ambientOcclusion(float3 Ph, float3 N, float aoRadius, float aoStrength,
                                  device const float3* positions,
                                  device const float* radii,
                                  device const uint* binStarts,
                                  device const uint* particleIndices,
                                  constant SimParams& sp,
                                  uint selfA, uint selfB)
    {
        const float h = sp.roi;
        float r2max = aoRadius * aoRadius;
        int3 lo = int3((int)floor((Ph.x - aoRadius) / h),
                       (int)floor((Ph.y - aoRadius) / h),
                       (int)floor((Ph.z - aoRadius) / h));
        int3 hi = int3((int)floor((Ph.x + aoRadius) / h),
                       (int)floor((Ph.y + aoRadius) / h),
                       (int)floor((Ph.z + aoRadius) / h));
        float occ = 0.0f;
        for (int z = lo.z; z <= hi.z; ++z) {
            for (int y = lo.y; y <= hi.y; ++y) {
                for (int x = lo.x; x <= hi.x; ++x) {
                    uint hh = binHash(int3(x, y, z), sp);
                    uint s = binStarts[hh];
                    uint e = binStarts[hh + 1];
                    for (uint idx = s; idx < e; ++idx) {
                        uint r = particleIndices[idx];
                        if (r == selfA || r == selfB) continue;
                        float rad = radii[r];
                        if (rad <= 0.0f) continue;
                        float3 dv = float3(positions[r]) - Ph;
                        float d2 = dot(dv, dv);
                        if (d2 < 1e-12f || d2 > r2max) continue;
                        float invD = rsqrt(d2);
                        float cosT = dot(N, dv) * invD;
                        if (cosT <= 0.0f) continue;   // below the tangent plane
                        occ += (rad * rad * invD * invD) * cosT;   // sin²α·cosθ
                    }
                }
            }
        }
        return clamp(1.0f - aoStrength * occ, 0.0f, 1.0f);
    }

    // V2.1b: surface shading shared by the solid style and the foam faces
    // (caps AND walls). Extracted VERBATIM from the M4/V2.1 solid block —
    // the arithmetic order is unchanged, so the solid goldens stay
    // byte-identical (ambientOcclusion's new exclusion params receive
    // (hit, none) there: the hit cell contributed exactly zero before —
    // cosT <= 0 at its own cap point — and the shadow march's second
    // self-index never matches).
    //
    //   Ph    surface point; N shading normal (sphere normal for caps; the
    //         canonical wall normal flipped toward the camera for walls)
    //   tintV/litV  the face's tint and illumination fraction: the cell's own
    //         for caps and the solid style; the two cells' AVERAGE for walls
    //         (the wall is shared — both sides must shade it identically)
    //   selfB second cell excluded from the AO cube walk and the shadow march
    //         (walls: the wall partner — the wall point lies inside both
    //         chambers' spheres; 0xFFFFFFFF = none)
    //
    // Directional mode (M5 unified light): vis = mix(per-cell fraction,
    // per-pixel ray average, shadow strength); shade = 0.2 + 0.8*ndl*vis (the
    // M5 ambient-floor form). AO stays the OMNIDIRECTIONAL self-shadowing
    // multiplier (paper §5/§6 style 1); the shadow term is the DIRECTIONAL
    // occlusion inside the Lambertian response — the standard ambient+direct
    // split: col = tint·(aoFloor + (1−aoFloor)·ao)·(0.2 + 0.8·ndl·vis).
    // Unbiased mode (directionalLight == 0) reads neither litFraction nor
    // shadow rays: shade = 1 exactly (M4 goldens byte-identical).
    static float3 shadeSurface(float3 Ph, float3 N, float3 tintV, float litV,
                               uint selfIdx, uint selfB,
                               uint px, uint py,
                               device const float3* positions,
                               device const float* radii,
                               device const uint* binStarts,
                               device const uint* particleIndices,
                               device const uint* occupancy,
                               constant SimParams& sp,
                               constant RenderParams& rp)
    {
        float ao = ambientOcclusion(Ph, N, rp.aoRadius, rp.aoStrength,
                                    positions, radii, binStarts,
                                    particleIndices, sp, selfIdx, selfB);
        float shade = 1.0f;
        if (rp.directionalLight != 0u) {
            float3 L = normalize(rp.lightDir.xyz);
            float ndl = max(dot(N, L), 0.0f);
            float vis = litV;
            if (rp.shadows > 0.0f && rp.shadowSamples > 0u) {
                // Per-pixel penumbra: hard occlusion along jittered
                // directions inside the light's angular cone. Sample stream
                // starts BEYOND the AA sample indices (s + rp.samples) so the
                // two jitters are decorrelated but both deterministic.
                float shadowT1 = length(Ph) + length(rp.boundingCentre.xyz)
                               + 2.0f * rp.boundingRadius + 2.0f * sp.roi;
                float visAccum = 0.0f;
                for (uint ss = 0; ss < rp.shadowSamples; ++ss) {
                    float2 sj = sampleJitter(px, py, rp.samples + ss, rp.renderSeed);
                    float3 srd = coneJitter(L, sp.lightSoft, sj);
                    visAccum += (lightVisibility(Ph, srd, 1e-3f, shadowT1, selfIdx,
                                                 selfB, 0.0f, positions, radii,
                                                 binStarts, particleIndices,
                                                 occupancy, sp) > 0.0f) ? 1.0f : 0.0f;
                }
                float visPixel = visAccum / float(rp.shadowSamples);
                vis = mix(vis, visPixel, clamp(rp.shadows, 0.0f, 1.0f));
            }
            shade = 0.2f + 0.8f * ndl * vis;
        }
        return tintV * (rp.aoFloor + (1.0f - rp.aoFloor) * ao) * shade;
    }

    // One thread per pixel of the current tile.
    //
    // Buffer table:
    //   texture(0) outTex (write; bgra8Unorm, tile-sized)
    //   0 positions (const)        1 radii (const)
    //   2 binStarts (const)        3 particleIndices (const)
    //   4 cellCounter (const)      5 RenderParams
    //   6 SimParams (roi + grid hash bits + lightSoft; the sim's own params buffer)
    //   7 bigCells (const; supplement indices, rad >= roi)
    //   8 tint (const; M5 per-cell albedo from the cellTint kernel)
    //   9 litFraction (const; V2.1 per-cell illumination fraction from illuminate)
    //  10 occupancy (const; M6 coarse skip map — the shadow marches skip on it)
    //  11 links (const; V2.1b foam chamber CSG — the cells' rings)
    //  12 linkCounts (const; V2.1b)
    kernel void raytraceTile(texture2d<float, access::write> outTex [[texture(0)]],
                             device const float3* positions [[buffer(0)]],
                             device const float*   radii [[buffer(1)]],
                             device const uint*   binStarts [[buffer(2)]],
                             device const uint*   particleIndices [[buffer(3)]],
                             device const uint*   cellCounter [[buffer(4)]],
                             constant RenderParams& rp [[buffer(5)]],
                             constant SimParams& sp [[buffer(6)]],
                             device const uint*   bigCells [[buffer(7)]],
                             device const float3* tint [[buffer(8)]],
                             device const float*  litFraction [[buffer(9)]],
                             device const uint*   occupancy [[buffer(10)]],
                             device const uint*   links [[buffer(11)]],
                             device const uint*   linkCounts [[buffer(12)]],
                             uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= rp.tileW || gid.y >= rp.tileH) return;
        const uint count = cellCounter[0];
        const uint px = rp.tileX + gid.x;
        const uint py = rp.tileY + gid.y;
        const float aspect = (float)rp.imageW / (float)rp.imageH;
        const bool solid = (rp.style == 0u);
        const bool foam = (rp.style == 2u);
        const bool density = (rp.style == 1u || rp.style == 3u);

        float3 accum = float3(0.0);
        float3 densAccum = float3(0.0);
        float dummyF = 0.0f;      // wantNearest dummy (float*, traverseGrid contract)
        float3 dummyF3 = float3(0.0);   // wantDensity dummy
        uint  dummyU = 0u;
        FoamFace dummyFaces[1];   // wantFoam dummy (non-foam styles never touch it)
        uint  dummyFaceCount = 0u;

        for (uint s = 0; s < rp.samples; ++s) {
            float2 j = sampleJitter(px, py, s, rp.renderSeed);
            // Same camera conventions as the raster path: NDC +y up at the
            // top of the image (row 0), view space looks down -Z.
            float2 ndc;
            ndc.x = (2.0f * ((float)px + j.x) / (float)rp.imageW - 1.0f) * rp.tanHalfFov * aspect;
            ndc.y = (1.0f - 2.0f * ((float)py + j.y) / (float)rp.imageH) * rp.tanHalfFov;
            float3 dirV = normalize(float3(ndc, -1.0f));
            // world dir = dirV.x*right + dirV.y*up + dirV.z*(backward)
            float3 rd = normalize(dirV.x * rp.camRight.xyz
                                + dirV.y * rp.camUp.xyz
                                - dirV.z * rp.camFwd.xyz);
            float3 ro = rp.camEye.xyz;

            // Clip the ray to the scene's bounding sphere (inflated by the
            // max cell radius, so every sphere entry is inside [t0, t1]).
            float3 b = rp.boundingCentre.xyz - ro;
            float tc = dot(b, rd);
            float d2 = dot(b, b) - tc * tc;
            float br = rp.boundingRadius;
            if (d2 > br * br) {
                if (solid || foam) accum += rp.background.xyz;
                continue;   // xray: nothing along the ray
            }
            float thc = sqrt(br * br - d2);
            float t0 = max(tc - thc, 0.0f);
            float t1 = tc + thc;

            if (solid) {
                float nearest = INFINITY;
                uint hitIdx = 0u;
                traverseGrid(ro, rd, t0, t1, count, positions, radii, links, linkCounts,
                             binStarts, particleIndices, sp, rp.profileSupport, 1u, rp, tint,
                             true, &nearest, &hitIdx,
                             false, &dummyF3,
                             false, dummyFaces, &dummyFaceCount, 1u);
                bigCellPass(ro, rd, t0, t1, bigCells, rp.bigCount, positions, radii,
                            links, linkCounts, binStarts, particleIndices, sp,
                            rp.profileSupport, rp, tint,
                            true, &nearest, &hitIdx, false, &dummyF3,
                            false, dummyFaces, &dummyFaceCount, 1u);
                if (nearest < INFINITY) {
                    float3 C = float3(positions[hitIdx]);
                    float rad = radii[hitIdx];
                    float3 Ph = ro + rd * nearest;
                    float3 N = (Ph - C) / max(rad, 1e-9f);   // (hit - centre)/radius
                    // M5 unified light + V2.1 directional shading, factored
                    // VERBATIM into shadeSurface (V2.1b) so the solid and the
                    // foam faces share one implementation: AO = the
                    // omnidirectional self-shadowing (paper §5/§6 style 1);
                    // the shadow term = the DIRECTIONAL occlusion inside the
                    // Lambertian response (per-cell illumination fraction
                    // refined by jittered cone shadow rays). Per-cell tint
                    // (neutral = the pre-M5 albedo, so unbiased renders are
                    // byte-identical to M4).
                    accum += shadeSurface(Ph, N, float3(tint[hitIdx]), litFraction[hitIdx],
                                          hitIdx, 0xFFFFFFFFu, px, py,
                                          positions, radii, binStarts, particleIndices,
                                          occupancy, sp, rp);
                } else {
                    accum += rp.background.xyz;
                }
            } else if (foam) {
                // V2.1b foam (style 2): composite the nearest CHAMBER entry
                // faces front to back with per-face opacity. The union
                // boundary is always a cap; every wall crossed by the ray is
                // the following chamber's entry (see the visibility fact in
                // `common`), so the flat walls show THROUGH the translucent
                // front caps — walls behind walls, the soap-froth depth.
                FoamFace faces[8];
                uint faceCount = 0u;
                const uint maxFaces = min(rp.foamLayers, 8u);
                traverseGrid(ro, rd, t0, t1, count, positions, radii, links, linkCounts,
                             binStarts, particleIndices, sp, rp.profileSupport, 1u, rp, tint,
                             false, &dummyF, &dummyU,
                             false, &dummyF3,
                             true, faces, &faceCount, maxFaces);
                bigCellPass(ro, rd, t0, t1, bigCells, rp.bigCount, positions, radii,
                            links, linkCounts, binStarts, particleIndices, sp,
                            rp.profileSupport, rp, tint,
                            false, &dummyF, &dummyU, false, &dummyF3,
                            true, faces, &faceCount, maxFaces);
                float T = 1.0f;
                for (uint f = 0; f < faceCount; ++f) {
                    float3 col;
                    const uint idx = faces[f].idx;
                    const float3 C = float3(positions[idx]);
                    const float3 Ph = ro + rd * faces[f].t;
                    if (faces[f].wallJ == 0xFFFFFFFFu) {
                        // Curved cap: the solid shading with the sphere normal.
                        float rad = radii[idx];
                        float3 N = (Ph - C) / max(rad, 1e-9f);
                        col = shadeSurface(Ph, N, float3(tint[idx]), litFraction[idx],
                                           idx, 0xFFFFFFFFu, px, py,
                                           positions, radii, binStarts, particleIndices,
                                           occupancy, sp, rp);
                    } else {
                        // Flat wall: the canonical bisector normal flipped
                        // toward the camera; tint and illumination are the two
                        // cells' AVERAGE — the wall is shared, so both sides
                        // shade it identically.
                        float3 nh = foamWallNormal(idx, faces[f].wallJ, positions);
                        float3 N = dot(nh, rd) > 0.0f ? -nh : nh;
                        float3 tintV = 0.5f * (float3(tint[idx]) + float3(tint[faces[f].wallJ]));
                        float litV = 0.5f * (litFraction[idx] + litFraction[faces[f].wallJ]);
                        col = shadeSurface(Ph, N, tintV, litV, idx, faces[f].wallJ, px, py,
                                           positions, radii, binStarts, particleIndices,
                                           occupancy, sp, rp);
                    }
                    accum += T * rp.foamOpacity * col;
                    T *= 1.0f - rp.foamOpacity;
                    if (T < 0.02f) break;   // deeper layers contribute < 2%
                }
                accum += T * rp.background.xyz;   // gaps + the attenuated background
            } else if (density) {
                float3 dens = float3(0.0);
                traverseGrid(ro, rd, t0, t1, count, positions, radii, links, linkCounts,
                             binStarts, particleIndices, sp, rp.profileSupport,
                             (uint)ceil(rp.profileSupport), rp, tint,
                             false, &dummyF, &dummyU,
                             true, &dens,
                             false, dummyFaces, &dummyFaceCount, 1u);
                bigCellPass(ro, rd, t0, t1, bigCells, rp.bigCount, positions, radii,
                            links, linkCounts, binStarts, particleIndices, sp,
                            rp.profileSupport, rp, tint,
                            false, &dummyF, &dummyU, true, &dens,
                            false, dummyFaces, &dummyFaceCount, 1u);
                densAccum += dens;
            }
        }

        float4 out;
        if (density) {
            // V2.1: per-channel tone map (grayscale path accumulates the same
            // scalar density in all three channels — bit-identical to M4).
            float3 dens = densAccum / float(max(rp.samples, 1u)) * rp.densityScale;
            float3 L = 1.0f - exp(-dens * rp.xrayExposure);
            L = pow(L, 1.0f / max(rp.xrayGamma, 1e-3f));
            out = float4(L, 1.0f);
        } else {
            // solid AND foam: straight colour average over the AA samples.
            out = float4(accum / float(max(rp.samples, 1u)), 1.0f);
        }
        outTex.write(out, gid);
    }

    // ---- CGDBG probe: recompute ONE pixel with full instrumentation ----
    // Dumps the primary hit, the AO cube walk bounds/candidate count, the
    // occ sum, the first 8 AO candidates (idx, d, cosT), the ray, the jitter,
    // and raw RenderParams/hash echoes. One thread.
    //
    // Buffer table:
    //   0 out (write, 96 x float)   1 positions   2 radii   3 binStarts
    //   4 particleIndices   5 cellCounter   6 RenderParams   7 SimParams
    //   8 bigCells (supplement)   9 probe pixel (2 x uint)   10 tint (const)
    //  11 links   12 linkCounts (V2.1b: traverseGrid's foam signature)
    kernel void raytraceProbe(device float* out [[buffer(0)]],
                              device const float3* positions [[buffer(1)]],
                              device const float* radii [[buffer(2)]],
                              device const uint* binStarts [[buffer(3)]],
                              device const uint* particleIndices [[buffer(4)]],
                              device const uint* cellCounter [[buffer(5)]],
                              constant RenderParams& rp [[buffer(6)]],
                              constant SimParams& sp [[buffer(7)]],
                              device const uint* bigCells [[buffer(8)]],
                              device const uint* probePixel [[buffer(9)]],
                              device const float3* tint [[buffer(10)]],
                              device const uint* links [[buffer(11)]],
                              device const uint* linkCounts [[buffer(12)]],
                              uint gid [[thread_position_in_grid]])
    {
        if (gid != 0u) return;
        for (uint i = 0; i < 96u; ++i) out[i] = -1.0f;
        const uint count = cellCounter[0];
        const uint px = probePixel[0];
        const uint py = probePixel[1];
        const float aspect = (float)rp.imageW / (float)rp.imageH;
        float2 j = sampleJitter(px, py, 0u, rp.renderSeed);
        float2 ndc;
        ndc.x = (2.0f * ((float)px + j.x) / (float)rp.imageW - 1.0f) * rp.tanHalfFov * aspect;
        ndc.y = (1.0f - 2.0f * ((float)py + j.y) / (float)rp.imageH) * rp.tanHalfFov;
        float3 dirV = normalize(float3(ndc, -1.0f));
        float3 rd = normalize(dirV.x * rp.camRight.xyz + dirV.y * rp.camUp.xyz - dirV.z * rp.camFwd.xyz);
        float3 ro = rp.camEye.xyz;

        float3 b = rp.boundingCentre.xyz - ro;
        float tc = dot(b, rd);
        float d2 = dot(b, b) - tc * tc;
        float thc = sqrt(rp.boundingRadius * rp.boundingRadius - d2);
        float t0 = max(tc - thc, 0.0f);
        float t1 = tc + thc;
        out[0] = t0; out[1] = t1;

        float nearest = INFINITY;
        uint hitIdx = 0u;
        float3 dens = float3(0.0);
        FoamFace dummyFaces[1];
        uint dummyFaceCount = 0u;
        traverseGrid(ro, rd, t0, t1, count, positions, radii, links, linkCounts,
                     binStarts, particleIndices, sp, rp.profileSupport,
                     (uint)ceil(rp.profileSupport), rp, tint,
                     true, &nearest, &hitIdx, true, &dens,
                     false, dummyFaces, &dummyFaceCount, 1u);
        bigCellPass(ro, rd, t0, t1, bigCells, rp.bigCount, positions, radii,
                    links, linkCounts, binStarts, particleIndices, sp,
                    rp.profileSupport, rp, tint,
                    true, &nearest, &hitIdx, true, &dens,
                    false, dummyFaces, &dummyFaceCount, 1u);
        out[2] = nearest;
        out[3] = (float)hitIdx;
        out[4] = dens.x;   // xray accumulation on the same ray (bonus diagnostic)
        if (nearest >= INFINITY) return;
        float3 C = float3(positions[hitIdx]);
        float rad = radii[hitIdx];
        float3 Ph = ro + rd * nearest;
        float3 N = (Ph - C) / max(rad, 1e-9f);
        out[5] = Ph.x; out[6] = Ph.y; out[7] = Ph.z;
        out[8] = N.x; out[9] = N.y; out[10] = N.z;

        // Instrumented AO cube walk (same formula as ambientOcclusion).
        const float h = sp.roi;
        float r2max = rp.aoRadius * rp.aoRadius;
        int3 lo = int3((int)floor((Ph.x - rp.aoRadius) / h),
                       (int)floor((Ph.y - rp.aoRadius) / h),
                       (int)floor((Ph.z - rp.aoRadius) / h));
        int3 hi = int3((int)floor((Ph.x + rp.aoRadius) / h),
                       (int)floor((Ph.y + rp.aoRadius) / h),
                       (int)floor((Ph.z + rp.aoRadius) / h));
        out[11] = (float)lo.x; out[12] = (float)lo.y; out[13] = (float)lo.z;
        out[14] = (float)hi.x; out[15] = (float)hi.y; out[16] = (float)hi.z;
        float occ = 0.0f;
        uint visited = 0u;
        uint kept = 0u;
        for (int z = lo.z; z <= hi.z; ++z) {
            for (int y = lo.y; y <= hi.y; ++y) {
                for (int x = lo.x; x <= hi.x; ++x) {
                    uint hh = binHash(int3(x, y, z), sp);
                    uint s = binStarts[hh];
                    uint e = binStarts[hh + 1];
                    for (uint idx = s; idx < e; ++idx) {
                        uint r = particleIndices[idx];
                        float rad2 = radii[r];
                        if (rad2 <= 0.0f) continue;
                        float3 dv = float3(positions[r]) - Ph;
                        float dd2 = dot(dv, dv);
                        if (dd2 < 1e-12f || dd2 > r2max) continue;
                        ++visited;
                        float invD = rsqrt(dd2);
                        float cosT = dot(N, dv) * invD;
                        if (cosT <= 0.0f) continue;
                        occ += (rad2 * rad2 * invD * invD) * cosT;
                        if (kept < 8u) {
                            out[17 + kept * 4u + 0u] = (float)r;
                            out[17 + kept * 4u + 1u] = sqrt(dd2);
                            out[17 + kept * 4u + 2u] = cosT;
                            out[17 + kept * 4u + 3u] = rad2;
                        }
                        ++kept;
                    }
                }
            }
        }
        out[49] = (float)visited;
        out[50] = (float)kept;
        out[51] = occ;
        out[52] = ro.x; out[53] = ro.y; out[54] = ro.z;
        out[55] = rd.x; out[56] = rd.y; out[57] = rd.z;
        out[58] = j.x; out[59] = j.y;
        out[60] = (float)px; out[61] = (float)py;
        // Raw RenderParams echo (bit-cast floats; layout ground truth).
        out[62] = as_type<float>(rp.renderSeed);
        out[63] = rp.tanHalfFov;
        // Hash ground truth: combined input + raw hash + a second helper call
        // (compiler sanity check: inline vs helper must agree).
        uint combined = px * 1973u ^ py * 9277u ^ 0u * 26699u ^ rp.renderSeed * 7919u;
        out[64] = as_type<float>(combined);
        out[65] = as_type<float>(jitterHash(combined));
        float2 j2 = sampleJitter(px, py, 0u, rp.renderSeed);
        out[66] = j2.x; out[67] = j2.y;
        // Bisection: brute-force density over ALL cells (no grid, no DDA,
        // no anchor — each cell enumerated once). Compare with out[4] (the
        // traversed density) and the CPU brute force.
        float densBF = 0.0f;
        for (uint i = 0; i < count; ++i) {
            float rad = radii[i];
            if (rad <= 0.0f) continue;
            float support = rad * rp.profileSupport;
            float3 C = float3(positions[i]);
            float3 toC = C - ro;
            float tcl = dot(toC, rd);
            float dLine2 = dot(toC, toC) - tcl * tcl;
            if (dLine2 > support * support) continue;
            float tf = clamp(tcl, t0, t1);
            float3 dv = C - (ro + rd * tf);
            float dd2 = dot(dv, dv);
            if (dd2 < support * support) {
                float q = 1.0f - dd2 / (support * support);
                densBF += rad * q * q;
            }
        }
        out[68] = densBF;
        // Per-cell ground truth for up to 8 host-selected cells (the CPU
        // verify's top contributors): rad, dLine2, contribution.
        for (uint k = 0; k < 8u; ++k) {
            uint i = probePixel[2 + k];
            if (i == 0xFFFFFFFFu || i >= count) { out[72 + k * 3 + 0] = -2.0f; continue; }
            float rad = radii[i];
            float3 C = float3(positions[i]);
            float3 toC = C - ro;
            float tcl = dot(toC, rd);
            float dLine2 = dot(toC, toC) - tcl * tcl;
            float contrib = 0.0f;
            if (rad > 0.0f && dLine2 <= rad * rad) {
                float th = sqrt(rad * rad - dLine2);
                float ta = tcl - th;
                float tb = tcl + th;
                if (!(tb <= t0 || ta >= t1)) {
                    float tf = clamp(tcl, t0, t1);
                    float3 dv = C - (ro + rd * tf);
                    float dd2 = dot(dv, dv);
                    float support = rad * rp.profileSupport;
                    if (dd2 < support * support) {
                        float q = 1.0f - dd2 / (support * support);
                        contrib = rad * q * q;
                    }
                }
            }
            out[72 + k * 3 + 0] = rad;
            out[72 + k * 3 + 1] = dLine2;
            out[72 + k * 3 + 2] = contrib;
        }
    }
    """

    // MARK: - Library sources (compositions; the names are the makeLibrary contract)

    static let sim = common + simKernels
    static let render = common + renderKernels
    static let raytrace = common + raytraceKernels
}

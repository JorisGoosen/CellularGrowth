//  Shaders.swift
//  Metal shader sources (MSL), compiled at runtime with device.makeLibrary(source:).
//  Kept as string constants so both `swift run` and the .app bundle work
//  without SPM resource plumbing.

import Foundation

enum Shaders {

    // MARK: - Simulation kernels (M1 forces; M2 adds food + cell division)

    static let sim = """
    #include <metal_stdlib>
    using namespace metal;

    // NOTE: layout must mirror Swift struct SimParams exactly (36 x 4 bytes = 144).
    // New fields are APPENDED at the end on both sides (AGENTS.md SimParams rule).
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
    };

    // Positions/normals use float3 (16-byte stride), matching Swift SIMD3<Float>
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
    // Buffer table:
    //   0 rngState (read+write)   1 food (write)   2 cellCounter   3 params
    //   4 nutrient (const, current buffer; light mode)
    //   5 chemU (const, current buffer; rde mode)   6 chemV (const; rde mode)
    kernel void foodStep(device uint64_t* rngState    [[buffer(0)]],
                         device float*   food         [[buffer(1)]],
                         device const uint* cellCounter [[buffer(2)]],
                         constant SimParams& params     [[buffer(3)]],
                         device const float* nutrient   [[buffer(4)]],
                         device const float* chemU      [[buffer(5)]],
                         device const float* chemV      [[buffer(6)]],
                         uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float drive = 1.0f;
        if (params.growthMode == 1u) {
            drive = params.lightBase + nutrient[gid];
        } else if (params.growthMode == 2u) {
            drive = params.rdeBase + params.rdeGain * chemV[gid];
        }
        food[gid] += params.foodIncrement * drive * rand01(&rngState[gid]);
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

    // Segment [t0, t1] occlusion test through the spatial hash (Amanatides–Woo
    // 3D-DDA — the M4 renderer's traverseGrid pattern, reduced to a boolean).
    // Correctness (same argument as the renderer): a sphere the segment crosses
    // inside a centerline bin has its centre within rad < h of that crossing,
    // so it lies in the bin's 27-bin packed-hash neighbourhood. Occlusion
    // radius = the renderer's per-cell radius (mean link distance x
    // radiusScale), CLAMPED below roi to stay inside the grid path's proof
    // (radii >= roi exist only on stretched cells, typically << 1% of a grown
    // state — they under-shadow slightly; documented in the M5 report).
    //
    // M6 coarse occupancy skip (the M5 report's two-level fix): rays from the
    // dark hemisphere cross the structure's EMPTY interior bin by bin, and
    // each empty bin costs 27 binStarts lookups into the 16 MB table. Before
    // the fine walk, the 27 coarse occupancy bits around the current 8³-bin
    // block are tested; if all clear, nothing can block inside that block
    // (proof in coarseNeighbourhoodClear) and the DDA jumps straight to the
    // block's exit — ~500 binStarts lookups collapse to 27 bit tests on a
    // 1 KB bitfield (L1/L2-resident). Skips are only ever taken over regions
    // provably free of cells, so the occlusion verdict is bit-identical to
    // the fine-only walk.
    static bool lightOccluded(float3 ro, float3 rd, float t0, float t1, uint selfIdx,
                              device const float3* positions,
                              device const float* radii,
                              device const uint* binStarts,
                              device const uint* particleIndices,
                              device const uint* occupancy,
                              constant SimParams& sp)
    {
        if (!(t1 > t0)) return false;
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
                if (tExitB >= t1) return false;   // segment ends inside the clear block
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
                            if (r == selfIdx) continue;
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
                            if (tb > t0 && ta < t1) return true;   // blocks the light
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
        return false;
    }

    // One thread per cell: illumination march + nutrient production. Runs on
    // the FROZEN pre-step state (positions + radii + the grid built from them
    // this step), before integrateForces — the march must match the grid
    // exactly, and the nutrient is consumed later in the same step by
    // foodStep (post-diffuse buffer). SCHEDULED every lightInterval steps
    // (host-side; production is pulsed lightInterval× so the average
    // production rate — and hence the nutrient steady state — is unchanged);
    // decay + diffusion run EVERY step in nutrientDiffuse.
    // M5 unified light: the per-cell lit/unlit verdict is also STORED
    // (litFlags) — the ray tracer's directional solid style shades with the
    // SAME per-cell illumination the growth used (CellSim.
    // encodeIlluminationRefresh recomputes it for a loaded state).
    //
    // Buffer table:
    //   0 positions (const)       1 radii (const)        2 binStarts (const)
    //   3 particleIndices (const) 4 nutrient (read+write) 5 bound2 (const)
    //   6 cellCounter             7 params               8 litFlags (write)
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
                           device uint*          litFlags [[buffer(8)]],
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

        bool blocked = lightOccluded(P, L, 1e-3f, t1, gid, positions, radii,
                                     binStarts, particleIndices, occupancy, params);
        if (!blocked) {
            n += params.lightNutrient * (float)params.lightInterval;   // pulsed production
        }
        litFlags[gid] = blocked ? 0u : 1u;
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
    // Buffer table:
    //   0 links (const)   1 linkCounts (const)   2 chemUCur (const)
    //   3 chemVCur (const)   4 chemUNext (write)   5 chemVNext (write)
    //   6 cellCounter   7 params
    kernel void rdeStep(device const uint* links      [[buffer(0)]],
                        device const uint*   linkCounts [[buffer(1)]],
                        device const float*  chemUCur   [[buffer(2)]],
                        device const float*  chemVCur   [[buffer(3)]],
                        device float*        chemUNext  [[buffer(4)]],
                        device float*        chemVNext  [[buffer(5)]],
                        device const uint*   cellCounter [[buffer(6)]],
                        constant SimParams&  params     [[buffer(7)]],
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
        float uv2 = u * v * v;
        float du = params.rdeDu * (uAvg - u) - uv2 + params.rdeF * (1.0f - u);
        float dv = params.rdeDv * (vAvg - v) + uv2 - (params.rdeF + params.rdeK) * v;
        chemUNext[gid] = clamp(u + du, 0.0f, 2.0f);
        chemVNext[gid] = clamp(v + dv, 0.0f, 2.0f);
    }

    // M5: per-cell RENDER colour. In rde mode the chemicals tint the cells:
    // base GREEN, shifted toward BLUE proportional to u and toward RED
    // proportional to v (documented mapping below); in uniform/light mode the
    // neutral albedo is written — bit-identical to the pre-M5 hardcoded
    // colours (the impostor fragment's float3(0.87, 0.85, 0.82) and
    // RayRenderSettings.albedo), so the M3/M4 goldens are unaffected.
    //
    // Mapping (linear blend of three base colours):
    //   green (0.15, 0.65, 0.20), blue (0.15, 0.25, 0.85), red (0.85, 0.15, 0.15)
    //   wBlue = 0.30 * clamp(u, 0, 1)   — u's background is 1, so the field
    //                                     carries a teal-green cast; u-depleted
    //                                     spot interiors lose the blue cast
    //   wRed  = 1.60 * clamp(v, 0, 1)   — v's dynamic range is ~0.4 (vs u's
    //                                     1.0); the 1.6 scale makes spots
    //                                     (v ≈ 0.35–0.45) read as clear red
    //                                     patches (wRed ≈ 0.56–0.72)
    //   wGreen = max(1 − wBlue − wRed, 0) — low-concentration cells stay green
    // Runs every step, AFTER the division block (M6: the daughters born this
    // step inherit the parent's u/v inside divisionExecute, so tint must be
    // recomputed from the post-division chemicals or last-step daughters
    // render with a stale/zero tint).
    //
    // Buffer table:
    //   0 chemU (const)   1 chemV (const)   2 tint (write)   3 cellCounter   4 params
    kernel void cellTint(device const float* chemU    [[buffer(0)]],
                         device const float* chemV    [[buffer(1)]],
                         device float3*      tint     [[buffer(2)]],
                         device const uint*  cellCounter [[buffer(3)]],
                         constant SimParams& params   [[buffer(4)]],
                         uint gid [[thread_position_in_grid]])
    {
        if (gid >= cellCounter[0]) return;
        float3 col = float3(0.87, 0.85, 0.82);   // neutral albedo (pre-M5 value)
        if (params.growthMode == 2u) {
            float uC = clamp(chemU[gid], 0.0f, 1.0f);
            float vC = clamp(chemV[gid], 0.0f, 1.0f);
            float wBlue = 0.30f * uC;
            float wRed  = 1.60f * vC;
            float wGreen = max(1.0f - wBlue - wRed, 0.0f);
            col = wGreen * float3(0.15, 0.65, 0.20)
                + wBlue  * float3(0.15, 0.25, 0.85)
                + wRed   * float3(0.85, 0.15, 0.15);
        }
        tint[gid] = col;
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
    // Buffer table:
    //   0 food (const)          1 links (const)     2 linkCounts (const)
    //   3 splitFlags (write)    4 cellCounter       5 params
    kernel void divisionDecide(device const float* food      [[buffer(0)]],
                               device const uint*   links          [[buffer(1)]],
                               device const uint*   linkCounts     [[buffer(2)]],
                               device uint*         splitFlags     [[buffer(3)]],
                               device const uint*   cellCounter    [[buffer(4)]],
                               constant SimParams&  params         [[buffer(5)]],
                               uint gid [[thread_position_in_grid]])
    {
        uint count = cellCounter[0];
        if (gid >= count) { splitFlags[gid] = 0u; return; }
        if (params.divisionEnabled == 0u) { splitFlags[gid] = 0u; return; }
        if (food[gid] <= params.splitThreshold) { splitFlags[gid] = 0u; return; }

        uint n = linkCounts[gid];
        if (n < 3u || n > params.maxValence) { splitFlags[gid] = 0u; return; }

        uint base = gid * params.maxValence;
        for (uint k = 0; k < n; ++k) {
            uint j = links[base + k];
            if (j == gid) { splitFlags[gid] = 0u; return; }   // corrupt ring: defer
            if (food[j] > params.splitThreshold && j < gid) { splitFlags[gid] = 0u; return; }
            uint jn = linkCounts[j];
            if (jn > params.maxValence) { splitFlags[gid] = 0u; return; }
            uint jbase = j * params.maxValence;
            for (uint k2 = 0; k2 < jn; ++k2) {
                uint l = links[jbase + k2];
                if (l != gid && food[l] > params.splitThreshold && l < gid) {
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
    // writes: own + daughter slot in nutrient/chemU/chemV) are disjoint. Reads
    // in this kernel touch only cells within ring-distance 1, which no
    // claimant writes. The daughter slot is >= the step-start count, disjoint
    // from the positions integrateForces wrote into this same buffer.
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

    // MARK: - Cell renderer (instanced sphere impostors)

    static let render = """
    #include <metal_stdlib>
    using namespace metal;

    // Layout must mirror Swift struct CameraUniforms.
    struct Camera {
        float4x4 view;
        float4x4 proj;
        float4 right;    // world-space camera right (unit)
        float4 up;       // world-space camera up (unit)
        float4 negFwd;   // world-space direction from target toward the camera
        float4 lightDir; // world-space unit vector pointing TOWARD the light
        float4 xrayPad;  // M6: .x = interactive X-ray exposure gain (solid path never reads it)
    };

    struct VOut {
        float4 clip    [[position]];
        float2 uv;       // [-1,1] billboard coordinates
        float3 centerV;  // view-space sphere center
        float3 right;
        float3 up;
        float3 negFwd;
        float  radius;
        float3 tint;     // M5: per-cell base colour (cellTint kernel; neutral albedo unless rde mode)
    };

    // One camera-facing quad per cell; the fragment shader shades it as a sphere.
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

    fragment FOut cellFragment(VOut in [[stage_in]],
                               constant Camera& cam [[buffer(0)]])
    {
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
        col *= 1.0 - 0.35 * rim;

        // True sphere-surface point (view space) -> exact fragment depth.
        float4 clip = cam.proj * float4(in.centerV + in.radius * nV, 1.0);

        FOut o;
        o.color = float4(col, 1.0);
        o.depth = clip.z / clip.w;
        return o;
    }
    """

    // MARK: - M4 offline ray tracer (compute kernels; raster path untouched)

    static let raytrace = """
    #include <metal_stdlib>
    using namespace metal;

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
    };

    // Same struct layout as Shaders.sim SimParams — bound to the SAME params
    // buffer (only roi + the grid bit fields are read here). Keep in sync;
    // appended-only per AGENTS.md.
    struct SimParams {
        float linkRestLength;
        float springFactor;
        float planarFactor;
        float bulgeFactor;
        float radiusScale;
        uint  cellCount;
        uint  maxValence;
        float foodIncrement;
        float splitThreshold;
        uint  divisionEnabled;
        uint  rngSeed;
        uint  maxCells;
        float roi;
        float repulsionStrength;
        uint  gridBitsX;
        uint  gridBitsY;
        uint  gridBitsZ;
        // --- M5 (appended; must mirror Shaders.sim exactly) ---
        uint  growthMode;
        float lightDirX;
        float lightDirY;
        float lightDirZ;
        float lightDiffusionRate;
        float lightNutrient;
        float lightDecay;
        float lightBase;
        uint  lightInterval;
        float rdeF;
        float rdeK;
        float rdeDu;
        float rdeDv;
        float rdeBase;
        float rdeGain;
        // --- M6 (appended; must mirror Shaders.sim exactly) ---
        uint  gridCoarseBitsX;
        uint  gridCoarseBitsY;
        uint  gridCoarseBitsZ;
        uint  lightCoarseSkip;
    };

    // Bit-packed bin hash — IDENTICAL to Shaders.sim binHash (same grid).
    static inline uint binHash(int3 c, constant SimParams& p) {
        uint maskX = (1u << p.gridBitsX) - 1u;
        uint maskY = (1u << p.gridBitsY) - 1u;
        uint maskZ = (1u << p.gridBitsZ) - 1u;
        return ((uint)c.x & maskX)
             | (((uint)c.y & maskY) << p.gridBitsX)
             | (((uint)c.z & maskZ) << (p.gridBitsX + p.gridBitsY));
    }

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
    static void traverseGrid(float3 ro, float3 rd, float t0, float t1, uint count,
                             device const float3* positions,
                             device const float* radii,
                             device const uint* binStarts,
                             device const uint* particleIndices,
                             constant SimParams& sp,
                             float profileSupport, uint nbhdK,
                             bool wantNearest, thread float* nearest, thread uint* nearestIdx,
                             bool wantDensity, thread float* density)
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
                                float tf = clamp(tc, t0, t1);
                                if (tf >= tCur && tf < tExitC) {
                                    // Distance from the centre to the SEGMENT
                                    // (clamped foot): rim spheres beyond the
                                    // segment ends fall off naturally.
                                    float3 dv = C - (ro + rd * tf);
                                    float d2 = dot(dv, dv);
                                    if (d2 < support * support) {
                                        float q = 1.0f - d2 / (support * support);
                                        *density += rad * q * q;
                                    }
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
    static void bigCellPass(float3 ro, float3 rd, float t0, float t1,
                            device const uint* bigCells, uint bigCount,
                            device const float3* positions,
                            device const float* radii,
                            float profileSupport,
                            bool wantNearest, thread float* nearest, thread uint* nearestIdx,
                            bool wantDensity, thread float* density)
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
            if (wantDensity) {
                float tf = clamp(tc, t0, t1);
                float3 dv = C - (ro + rd * tf);
                float d2 = dot(dv, dv);
                if (d2 < support * support) {
                    float q = 1.0f - d2 / (support * support);
                    *density += rad * q * q;
                }
            }
        }
    }

    // Analytic solid-angle ambient occlusion at surface point Ph with normal
    // N (formula in the header comment). Occluders = all spheres within
    // aoRadius of Ph, gathered from the grid with a cube walk
    // (2*(aoRadius/h) + 1 bins per axis).
    static float ambientOcclusion(float3 Ph, float3 N, float aoRadius, float aoStrength,
                                  device const float3* positions,
                                  device const float* radii,
                                  device const uint* binStarts,
                                  device const uint* particleIndices,
                                  constant SimParams& sp)
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

    // One thread per pixel of the current tile.
    //
    // Buffer table:
    //   texture(0) outTex (write; bgra8Unorm, tile-sized)
    //   0 positions (const)        1 radii (const)
    //   2 binStarts (const)        3 particleIndices (const)
    //   4 cellCounter (const)      5 RenderParams
    //   6 SimParams (roi + grid hash bits; the sim's own params buffer)
    //   7 bigCells (const; supplement indices, rad >= roi)
    //   8 tint (const; M5 per-cell albedo from the cellTint kernel)
    //   9 litFlags (const; M5 per-cell illumination verdict from illuminate)
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
                             device const uint*   litFlags [[buffer(9)]],
                             uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= rp.tileW || gid.y >= rp.tileH) return;
        const uint count = cellCounter[0];
        const uint px = rp.tileX + gid.x;
        const uint py = rp.tileY + gid.y;
        const float aspect = (float)rp.imageW / (float)rp.imageH;
        const bool solid = (rp.style == 0u);

        float3 accum = float3(0.0);
        float densAccum = 0.0f;
        float dummyF = 0.0f;
        uint  dummyU = 0u;

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
                if (solid) accum += rp.background.xyz;
                continue;   // xray: nothing along the ray
            }
            float thc = sqrt(br * br - d2);
            float t0 = max(tc - thc, 0.0f);
            float t1 = tc + thc;

            if (solid) {
                float nearest = INFINITY;
                uint hitIdx = 0u;
                traverseGrid(ro, rd, t0, t1, count, positions, radii, binStarts,
                             particleIndices, sp, rp.profileSupport, 1u,
                             true, &nearest, &hitIdx,
                             false, &dummyF);
                bigCellPass(ro, rd, t0, t1, bigCells, rp.bigCount, positions, radii,
                            rp.profileSupport, true, &nearest, &hitIdx, false, &dummyF);
                if (nearest < INFINITY) {
                    float3 C = float3(positions[hitIdx]);
                    float rad = radii[hitIdx];
                    float3 Ph = ro + rd * nearest;
                    float3 N = (Ph - C) / max(rad, 1e-9f);   // (hit - centre)/radius
                    float ao = ambientOcclusion(Ph, N, rp.aoRadius, rp.aoStrength,
                                                positions, radii, binStarts,
                                                particleIndices, sp);
                    // M5 unified light: per-cell tint (neutral = the pre-M5
                    // albedo, so unbiased renders are byte-identical to M4).
                    // Directional mode shades with the SAME per-cell
                    // illumination verdict the growth used (litFlags — no
                    // separate shadow march; the ambient floor keeps shaded
                    // form readable while lit surfaces read bright).
                    float shade = 1.0f;
                    if (rp.directionalLight != 0u) {
                        float3 L = normalize(rp.lightDir.xyz);
                        float ndl = max(dot(N, L), 0.0f);
                        shade = 0.2f + 0.8f * ndl * (litFlags[hitIdx] != 0u ? 1.0f : 0.0f);
                    }
                    accum += float3(tint[hitIdx])
                           * (rp.aoFloor + (1.0f - rp.aoFloor) * ao) * shade;
                } else {
                    accum += rp.background.xyz;
                }
            } else {
                float dens = 0.0f;
                traverseGrid(ro, rd, t0, t1, count, positions, radii, binStarts,
                             particleIndices, sp, rp.profileSupport,
                             (uint)ceil(rp.profileSupport),
                             false, &dummyF, &dummyU,
                             true, &dens);
                bigCellPass(ro, rd, t0, t1, bigCells, rp.bigCount, positions, radii,
                            rp.profileSupport, false, &dummyF, &dummyU, true, &dens);
                densAccum += dens;
            }
        }

        float4 out;
        if (solid) {
            out = float4(accum / float(max(rp.samples, 1u)), 1.0f);
        } else {
            float dens = densAccum / float(max(rp.samples, 1u)) * rp.densityScale;
            float L = 1.0f - exp(-dens * rp.xrayExposure);
            L = pow(L, 1.0f / max(rp.xrayGamma, 1e-3f));
            out = float4(L, L, L, 1.0f);
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
    //   8 bigCells (supplement)   9 probe pixel (2 x uint)
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
        float dens = 0.0f;
        traverseGrid(ro, rd, t0, t1, count, positions, radii, binStarts,
                     particleIndices, sp, rp.profileSupport,
                     (uint)ceil(rp.profileSupport),
                     true, &nearest, &hitIdx, true, &dens);
        bigCellPass(ro, rd, t0, t1, bigCells, rp.bigCount, positions, radii,
                    rp.profileSupport, true, &nearest, &hitIdx, true, &dens);
        out[2] = nearest;
        out[3] = (float)hitIdx;
        out[4] = dens;   // xray accumulation on the same ray (bonus diagnostic)
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
}

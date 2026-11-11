# AGENTS.md

Working agreements for every agent (human or subagent) working in this repo.
Read this file BEFORE changing anything.

## Project

Reimplementation of Andy Lomas, *"Cellular Forms: an Artistic Exploration of
Morphogenesis"* (AISB50) as a macOS Swift/Metal app for Apple Silicon
(target machine: M1 Pro, 16 GB unified memory).

- **Authoritative spec**: `docs/reference/lomas_cellular_forms_extraction.md`
  — verbatim paper transcription, all formulas in LaTeX with notation notes,
  and the formula-verification section (including the `bulgeDist` typo:
  printed `|L_r|²` must be implemented as `|L_r − P|²`).
- **Page renders**: `docs/reference/pages/` (8 pages + 2 hi-res formula pages).
- **NEVER modify `docs/reference/**`** — it is the immutable source of truth.

## Hard rules

1. A milestone is DONE only when every validation gate below passes AND the
   milestone report is delivered.
2. All simulation data lives in preallocated Metal buffers
   (`storageModeShared`). No per-step allocations. Growth must stay within
   `maxCells` (2M default; 16 GB machine ⇒ keep final runs ≤ ~30M cells).
3. Positions are double-buffered: within one step all kernels read the current
   buffer and write the next one; buffers swap after the step.
4. Shaders are MSL string constants in `Sources/CellularGrowth/Shaders.swift`.
   Do NOT move them to SPM resources (breaks `swift run` + .app bundling).
5. No momentum in the integrator (paper §4): positions move a *fraction*
   toward computed target positions. Any new force must be an additive
   over-damped offset toward a target. Justification: highly viscous medium;
   division injects energy every step, inertia would destabilize.
6. Comments and identifiers in English. Keep the existing file layout.
7. `docs/benchlog.md` is append-only history. README's roadmap table reflects
   live milestone status.
8. The repo is a git repository. The **orchestrator commits** after each
   verified milestone (and at meaningful checkpoints); subagents do NOT run
   git commit — leave the tree as they finished it for verification.

## Commands

```bash
swift build                                        # dev build
swift run CellularGrowth                           # GUI app
scripts/make_app.sh && open CellularGrowth.app     # double-clickable bundle

# headless stats (validation scale)
swift run -c release CellularGrowth --headless --steps 3000 --subdiv 3 --jitter 0.25

# offscreen screenshot — then READ the PNG (agents can read images)
swift run -c release CellularGrowth --screenshot /tmp/x.png --steps 500 --subdiv 4 --jitter 0.2

# benchmark sweep (lightweight timers; see Bench protocol)
swift run -c release CellularGrowth --bench

# invariant selftest (exists from M2 onward)
swift run -c release CellularGrowth --selftest

# CPU-lockstep validation of the GPU step semantics (M3+)
swift run -c release CellularGrowth --lockstep

# golden-image regression (M3+): capture/check deterministic renders
swift run -c release CellularGrowth --golden-capture NAME
swift run -c release CellularGrowth --golden-check NAME

# debug instrumentation (paramEcho + hash fingerprint + probe cb.error checks)
CELLDBG=1 swift run -c release CellularGrowth --lockstep
```

Headless flags: `--steps N`, `--subdiv N`, `--jitter F`, `--spring F`,
`--planar F`, `--bulge F`, `--screenshot PATH`, `--no-div`, `--food F`,
`--threshold F`, `--seed N`, `--max-cells N`, `--roi F`, `--repulsion F`,
`--bench-growth`, `--render PATH`, `--style solid|xray`, `--width N`,
`--height N`, `--tiled`, `--samples N`, `--state-save PATH`,
`--state-load PATH`. M5: `--growth uniform|light|rde`, `--light-dir "x,y,z"`,
`--light-diffuse F`, `--light-nutrient F`, `--light-decay F`, `--light-base F`,
`--light-interval N`, `--rde-F F`, `--rde-k F`, `--rde-du F`, `--rde-dv F`,
`--rde-base F`, `--rde-gain F`, `--rde-probe`, `--rde-dump DIR`,
`--render-light auto|growth|unbiased`. M6: `--preset
ball|brain|plant|coral|degenerate-demo`, `--anim DIR`, `--anim-steps N`,
`--anim-frames M`, `--stereo`, `--interocular F`,
`--screenshot-style solid|xray`. V2.1: `--edges F`, `--edge-width F`,
`--shadows F`, `--xray-color`, `--light-soft F`.

Scratch-artefact policy: screenshots, test renders and debug dumps go under
`/private/tmp/opencode/cellulargrowth/` (pre-approved scratch) — NOT generic
`/tmp`. Permanent artefacts (goldens, state snapshots, bench log) live in-repo
(`docs/golden/`, `docs/benchlog.md`).

## Architecture

| File | Role |
|---|---|
| `Core/CellSim.swift` | buffers, pipeline states, `step(_:)` encoding, stats |
| `Core/Icosphere.swift` | initial ball; links in **cyclic ring order** (CCW seen from outside) |
| `Core/HeadlessRunner.swift` | CLI modes: headless / screenshot / bench / selftest |
| `Render/Renderer.swift` | instanced sphere impostors; shared by GUI + offscreen |
| `App/*` | NSApplication, SwiftUI controls, MTKView orbit/zoom input |
| `Shaders.swift` | MSL sources, compiled at runtime |

**Ring order is load-bearing**: `computeNormals` derives the outward normal
from consecutive ring cross products; cell division (M2+) must preserve the
cyclic CCW ring structure. The initial ball starts ~5–6 valence, cap is
`CellSim.maxValence = 16`.

Per-step kernel pipeline (extend in place, keep order):
`normals → [grid rebuild: count, scanLocal, scanBlocks, scanOffset, scatter] →
[repulsionAccumulate] → [M5 chemicals: boundMax + illuminate + clearBound +
nutrientDiffuse | rdeStep] → integrateForces → food → division (decide,
execute, finalize) → cellTint → radii`.

State snapshots: **CGST v3** is the snapshot format (full state incl. chemicals,
RNG, raw SimParams); v1/v2 files load read-only-legacy. `docs/golden/*.state`
files written from M6 onward are v3.

## Data layout & Metal conventions

- Swift `SIMD3<Float>`: size 12, **alignment 16, stride 16**. MSL `device
  float3*` also strides 16. Always copy with
  `MemoryLayout<SIMD3<Float>>.stride` and declare MSL buffers as `float3*`.
  **Never** use `packed_float3` for these buffers.
- `SimParams` is mirrored between Swift and MSL. Add new fields ONLY at the
  end, on both sides, and update the layout comment in both.
- Each MSL kernel gets a comment listing its buffer-index table; the Swift
  `setBuffer` calls must match exactly.
- Compute dispatch: `dispatchThreads`, 64-thread threadgroups, kernels
  bounds-check `gid >= cellCount`. When a step changes the cell count
  (division), either dispatch an over-estimate and early-exit on a
  **device-side counter**, or split the step into two command buffers with a
  count readback — document which you chose.
- RNG: per-cell state in a buffer (PCG32 or xorshift), fixed seed ⇒
  deterministic runs (needed for the CPU-lockstep validation in M3+).

## Known pitfalls (each bit us once — do not re-trip)

1. `SIMD3<Float>` stride 16 vs `packed_float3` stride 12 ⇒ positions read as
   shifted garbage for every cell except cell 0.
2. Link arrays must be stored **strided** per cell (`cell * maxValence + k`),
   never packed/flatMapped.
3. Icosahedron base vertices must be normalized to the unit sphere (otherwise
   12 spike cells with ~5× rest-length links).
4. Offscreen color textures must be `.bgra8Unorm` (pipeline pixel format);
   `.rgba8Unorm` silently renders nothing (no CB error!).
5. `CGContext.makeImage()` snapshots the *current* buffer — copy the bytes in
   BEFORE calling it, or the PNG is blank/white.
6. `Renderer.encodeScene` must update the camera buffer itself; every call
   site needs it (offscreen path once rendered an empty frame).
7. MTKView continuous rendering = `view.isPaused = false` (there is no
   `isContinuous` property).
8. The paper's `bulgeDist` prints `|L_r|²` — implement `|L_r − P|²` (proof in
   the extraction doc). Clamp ONLY the √ radicand to ≥ 0 and KEEP the sign of
   `dotN` — do NOT clamp the per-link push at ≥ 0: a one-way clamp turns the
   sheet's scale into a ratchet (stretched patches never relax; found in M2).
   The faithful form restores the inward pull on stretched/bumped regions;
   rest-state behaviour is identical (x_r = |dotN| + dotN = 0).
9. A kernel written for exactly ONE threadgroup (single `shared[64]`, one
   barrier set, one final writer) must be dispatched with exactly one
   threadgroup — never `ceil(T/64)`: `threadgroups_per_grid` cannot double as
   a data count, it races N copies of the algorithm (the M3 "exactly 3×
   collision field" bug). Derive counts inside the kernel from `SimParams`.
10. New CLI flags must be added to BOTH `HeadlessRunner.shouldRun()` AND the
    `run()` dispatch — otherwise the app silently launches the GUI event loop
    and hangs with no output (`--bench-growth`, found in M3).
11. Swift `Set`/`Dictionary` iteration order is per-process randomized — never
    use it to seed deterministic data (Icosphere ring rotation, found in M6).
12. A light-march skip map must be live during the post-scatter march —
    "self-cleaning" counters consumed by the scatter are zero before the march
    reads them (found in M6).

## Validation gates (per milestone)

1. `swift build` clean.
2. `--selftest` PASS (from M2 on), invariants checked on a *growing* run:
   - links bidirectional: `j ∈ ring(i) ⟺ i ∈ ring(j)`;
   - closed triangulated sphere: **Σ valence = 6·(V − 2)** — follows from
     χ = V − E + F = 2 with E = Σval/2, F = Σval/3 on a triangulation;
   - valence ≤ maxValence; no NaN/Inf anywhere;
   - mean link length within ~2% of `linkRestLength` after settling;
   - max |P| bounded (no explosion).
3. `--headless` stats sane (no NaN, stable mean link length).
4. ≥ 1 `--screenshot` PNG **visually inspected by the executing agent**,
   compared against the target paper figure, observation written down.
5. `--bench` numbers appended to `docs/benchlog.md` with a milestone tag.
6. README roadmap status row updated.

## Bench protocol (lightweight timers)

`--bench` sweeps subdivision levels (⇒ cell counts), runs a fixed number of
steps per level (jitter 0), and reports per level: cells, ms/step (CPU wall),
ms/step (GPU, from `commandBuffer.gpuStartTime/gpuEndTime`), steps/s.
One warmup batch of 10 steps per level, batches of 10 steps per commit.
Append results to `docs/benchlog.md` as:

```markdown
## <milestone tag> — <date> — <device>
| cells | ms/step cpu | ms/step gpu | steps/s |
```

This is how we track how complexity progresses across milestones. Note: with
M3+, `--bench`/`--bench-growth` also print a per-level grid-memory line —
append it under the table (grid memory is preallocated for `T_max = 2²²`).

## Milestone workflow (subagents)

Every milestone is executed by a **fresh subagent**, started sequentially.
The subagent must: read AGENTS.md + the relevant extraction-doc sections +
current code; implement; run all validation gates; and report back in this
exact shape:

1. **Files changed** — paths + one-line what/why.
2. **Validation** — commands run + key output tails; screenshot paths +
   what they show.
3. **Bench table** (also appended to benchlog).
4. **Design choices** where the paper is silent (also as code comments).
5. **Open issues / risks** for the next milestone.

Live roadmap (status also kept in README):

- M1 ✅ forces (spring/planar/bulge), stable ball — paper Fig. 3
- M2 food + cell division (topology change) — paper Fig. 7; **expected**
  observation: without repulsion, prolonged growth degenerates (paper Fig. 5)
- M3 spatial-hash repulsion — paper Fig. 6 (coherent forms)
- M4 rendering v2: ambient occlusion + X-ray density accumulation
- M5 growth modes: light-based nutrient (+ diffusion), then RDE (Gray-Scott)
- M6 app polish: state save/load, animation export, stereo pairs, presets

# Bench log

Format per AGENTS.md bench protocol: cells | ms/step cpu | ms/step gpu | steps/s.
Release build, jitter 0, fixed steps per level, batches of 10, warmup 10.

## M1.5 baseline (forces only: normals, spring/planar/bulge, radii) — 2026-11-09 — Apple M1 Pro

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 | 0.112 | 0.088 | 8 894 |
| 10 242 | 0.150 | 0.126 | 6 649 |
| 40 962 | 0.348 | 0.322 | 2 876 |
| 163 842 | 0.918 | 0.889 | 1 089 |
| 655 362 | 3.588 | 3.555 | 279 |

Observations: clean O(N) scaling (~5.4 µs/step per 1k cells GPU). Extrapolated:
~50M cells ≈ 270 ms/step for forces only. Dispatch overhead dominates below
~10k cells. The heavy addition in M3 (repulsion) is expected to dominate this
baseline by a large factor; these numbers are the reference to measure it against.

## M2 forces-only (division compiled in, disabled) — 2026-11-09 — Apple M1 Pro

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 | 0.083 | 0.057 | 12 022 |
| 10 242 | 0.173 | 0.148 | 5 780 |
| 40 962 | 0.263 | 0.239 | 3 797 |
| 163 842 | 0.889 | 0.859 | 1 125 |
| 655 362 | 3.569 | 3.533 | 280 |

Observations: parity with the M1.5 baseline (same force kernels, now bounded on
the device cell counter). With division disabled the four division/food kernels
are skipped entirely, so the M1 cost structure is unchanged.

## M2 growth (division ON: food 0.01, threshold 1.0, seed 12345, 50 steps/level) — 2026-11-09 — Apple M1 Pro

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 → 3 288 | 0.163 | 0.131 | 6 118 |
| 10 242 → 13 241 | 0.145 | 0.117 | 6 886 |
| 40 962 → 53 178 | 0.448 | 0.420 | 2 230 |
| 163 842 → 212 709 | 1.342 | 1.307 | 745 |
| 655 362 → 851 452 | 5.220 | 5.185 | 192 |

Observations: cells grow ~28% during each 50-step run, so per-step costs average
over the growing count. The division machinery (foodStep + decide + execute +
finalize) costs ~1.5x a forces-only step at equal counts — cheap compared to
what the M3 spatial-hash repulsion is expected to add. The decide kernel's
ring-distance-2 guard short-circuits on the first lower-indexed candidate, so
its cost stays low even when most cells are above the split threshold (observed
when growth saturates at the maxCells cap).

## M3 repulsion (division OFF: grid rebuild + collisionOffset every step, roi 1.6, strength 0.1) — 2026-11-10 — Apple M1 Pro

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 | 0.218 | 0.184 | 4 580 |
| 10 242 | 0.241 | 0.213 | 4 145 |
| 40 962 | 0.721 | 0.689 | 1 386 |
| 163 842 | 2.509 | 2.470 | 399 |
| 655 362 | 9.928 | 9.891 | 101 |

Grid memory per level (binCounts + binStarts + particleIndices + collisionOffsets,
T = 2^(bx+by+bz) from the live count, buffers preallocated for T_max = 2^22):

- 2 562 cells: T=8 192 → 116 780 B
- 10 242 cells: T=32 768 → 466 988 B
- 40 962 cells: T=131 072 → 1 867 820 B
- 163 842 cells: T=524 288 → 7 471 148 B
- 655 362 cells: T=2 097 152 → 29 884 460 B

Observations: repulsion ≈ 2.8x the forces-only per-cell cost (9.93 vs 3.57 ms
at 655k cells; ~15 µs/step per 1k cells GPU vs 5.4 forces-only). Still clean
O(N): the five grid passes are O(N + T) and the 27-bin query is O(neighbours)
with bin size = roi. Grid memory is O(T_max) preallocated (~34 MB worst case at
2M cells), never reallocated.

## M3 growth (division ON: food 0.01, threshold 1.0, seed 12345, 50 steps/level, repulsion roi 1.6 strength 0.1) — 2026-11-10 — Apple M1 Pro

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 → 3 288 | 0.311 | 0.266 | 3 219 |
| 10 242 → 13 241 | 0.512 | 0.461 | 1 953 |
| 40 962 → 53 179 | 1.848 | 1.799 | 541 |
| 163 842 → 212 708 | 8.770 | 8.729 | 114 |
| 655 362 → 851 452 | 36.474 | 36.432 | 27 |

Grid memory at level ends: T=8 192/131 300 B, T=32 768/526 968 B,
T=131 072/2 112 160 B, T=524 288/8 448 468 B, T=2 097 152/33 806 260 B
(counts + starts + indices + collisionOffsets at the final count).

Observations: growing runs cost ~3.7x the static repulsion bench at equal
counts (36.5 vs 9.9 ms at ~655–851k cells): dividing cells pack densely, so
the average query sees many more candidates within roi than the pristine
icosphere (whose second-neighbour distance ~1.7 rest exceeds roi). The grid
rebuild itself is cheap; the 27-bin candidate walk dominates. No GPU hang; the
single-threadgroup scanBlocks pass (root-cause fix of the 3x field bug) stays
well under the step budget at T = 2^21 (B = 32 768, chunk 512).

## M4 render (offline ray tracer, 1600x1200 samples=1, release) — 2026-11-10 — Apple M1 Pro

Render wall time for the final state (grid built once for the render; grown
states carry division-stretched cells, radii up to ~1.7 > roi 1.6, so the
255-cell big-cell supplement pass runs). The X-ray style walks the DDA with
the K=2 neighbourhood (profile support 2 x radius needs the wider grid
domain), hence ~3x the solid cost.

| cells | style | render wall s | render gpu s |
|---|---|---|---|
| 91 142 (grown, seed 12345) | solid | 0.23 | 0.22 |
| 91 142 (grown) | xray | 0.58 | 0.58 |
| 254 305 (grown, 4000 steps) | solid | 0.46 | 0.45 |
| 254 305 (grown) | xray | 1.25 | 1.24 |
| 655 362 (static subdiv 8) | solid | 0.72 | 0.71 |
| 655 362 (static subdiv 8) | xray | 2.42 | 2.41 |
| 2 621 442 (static subdiv 9) | solid | 9.19 | 9.17 |
| 2 621 442 (static subdiv 9) | xray | 32.41 | 32.39 |

8600 x 8600 tiled renders (81 tiles of 1024², tile texture 4 MB, grown
91 142-cell state): solid 7.53 s wall / 7.39 s gpu; xray 20.56 s wall /
20.42 s gpu. CPU-side PNG assembly of the 296 MB byte buffer adds ~1.5 s.

Observations: solid-style cost scales with the number of centreline bins
(structure diameter / roi) x neighbourhood candidates — superlinear in cell
count once the hash table caps at T = 2^22 (subdiv 9: 14x the 655k time for
4x the cells; binStarts at 16 MB no longer fits L2). The X-ray adds the K=2
walk (125 bins visited per centreline bin) and the full-segment traversal,
so it is 3-4x solid throughout. Both styles render the 8600² artefact
resolution in seconds at 100k cells — practical for print-sized output.

## M5 growth modes (division ON: food 0.01, threshold 1.0, seed 12345, 50 steps/level, repulsion roi 1.6 strength 0.1) — 2026-11-10 — Apple M1 Pro

Per growth mode (`--bench-growth --growth <mode>`). Light mode marches the
illumination every `lightInterval` = 8 steps (pulsed production; steady state
unchanged — see the M5 report). All three tables from the same session/binary.

**uniform (control)**
| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 → 3 288 | 0.316 | 0.278 | 3 169 |
| 10 242 → 13 241 | 0.333 | 0.302 | 3 006 |
| 40 962 → 53 179 | 1.098 | 1.065 | 911 |
| 163 842 → 212 709 | 4.700 | 4.667 | 213 |
| 655 362 → 851 454 | 24.603 | 24.564 | 41 |

**light (interval 8)**
| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 → 2 886 | 0.393 | 0.357 | 2 546 |
| 10 242 → 11 662 | 0.509 | 0.468 | 1 966 |
| 40 962 → 46 617 | 2.149 | 2.108 | 465 |
| 163 842 → 186 550 | 12.241 | 12.194 | 82 |
| 655 362 → 745 651 | 88.938 | 88.890 | 11 |

**rde (Gray-Scott F 0.037, k 0.06, Du 0.35, Dv 0.15)**
| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 → 2 975 | 0.298 | 0.263 | 3 355 |
| 10 242 → 11 901 | 0.392 | 0.361 | 2 549 |
| 40 962 → 47 619 | 0.974 | 0.939 | 1 027 |
| 163 842 → 190 249 | 4.950 | 4.906 | 202 |
| 655 362 → 761 457 | 22.354 | 22.310 | 45 |

Grid memory at level ends (same shape for all modes; T grows with the live
count): T=8 192/123–131 kB, T=32 768/494–527 kB, T=131 072/1.98–2.11 MB,
T=524 288/7.9–8.45 MB, T=2 097 152/31.6–33.8 MB (counts + starts + indices +
collisionOffsets at the final count).

Observations:
- RDE ≈ uniform per-step cost (one ring walk per cell, explicit Euler dt=1);
  slightly cheaper at equal levels because patterned growth reaches fewer
  cells in 50 steps (761k vs 851k).
- Light mode ≈ uniform at validation scale (march amortized over 8 steps:
  0.39 vs 0.32 ms at 2.5k cells). At 655k cells the cost is dominated by the
  DARK hemisphere's rays crossing the smooth ball's EMPTY interior (~250 bins
  x 27 binStarts lookups per cell): measured 835 ms/step with interval 1
  (the first light bench run) vs 88.9 ms/step at interval 8. The structural
  fix (coarse occupancy skip: 8^3-fine-bin blocks, 27 bit-lookups per coarse
  DDA step instead of 512x27) is sketched in the M5 report as future work.
- uniform at 655k measured 24.6 ms/step vs M3-growth's 36.5 — same config;
  session/thermal variance (the M5 comparison is internally consistent).
- Chemical buffers add 6 x 4 B/cell preallocated (nutrient A/B, u A/B,
  v A/B = 48 MB at the 2M cap) + 1 x 4 B litFlags; no per-step allocations.

## M6 light-march occupancy skip (interleaved A/B, division ON: food 0.01, seed 12345, interval 1, 12 steps/level) — 2026-11-11 — Apple M1 Pro

The M5 open issue: the illumination march's empty-interior traversal dominates
light-mode steps at scale (dark-hemisphere rays cross the hollow interior bin
by bin, each empty bin costing 27 binStarts lookups into the 16 MB table).
M6 fix: a coarse occupancy bitfield (one bit per hashed 8³-fine-bin block,
FIXED 7-bit packed hash per axis, 2^21 slots = 256 KB, blit-cleared at every
grid rebuild); the march tests the 27 coarse bits around the current 8³-bin
block and, when all clear, jumps the DDA straight to the block's exit (proof
that a skipped region cannot contain a blocker is in
Shaders.swift/coarseNeighbourhoodClear). Skip correctness validated by an
exact A/B: a --no-div light-mode run saves BYTE-IDENTICAL state files with the
skip on (kill switch lightCoarseSkip/CG_NO_COARSE_SKIP) and off — the entire
nutrient trajectory is unchanged; only the kill-switch byte in the saved
SimParams differs.

The machine's thermal state drifted >2x across this session (M5's 835 ms and
the first M6 "before" at 460 ms are the same code), so the comparison below is
INTERLEAVED (on/off alternating in one session, two passes) — the ratio is the
measurement, not the absolute numbers:

| cells | ms/step skip ON | ms/step skip OFF | speedup |
|---|---|---|---|
| 655 362 → 675 925 | 215.2 / 221.8 | 771.6 / 906.8 | 3.6–4.1x |

Default light interval (8), same session (skip on, the shipped configuration):

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 → 2 890 | 0.467 | 0.426 | 2 139 |
| 10 242 → 11 648 | 0.592 | 0.553 | 1 688 |
| 40 962 → 46 622 | 1.973 | 1.930 | 507 |
| 163 842 → 186 533 | 7.587 | 7.540 | 132 |
| 655 362 → 745 705 | 35.230 | 35.174 | 28 |

vs the M5 interval-8 table (88.9 ms/step at 655k): ~2.5x. The two failed
designs are documented so they are not re-tried: (1) coarse hash bits derived
from the fine table size (max(bits−3, 2)) saturate by coordinate-wrap
collisions at high cell counts — the skip never fires; (2) "self-cleaning"
occupancy counts mirrored in gridScatter are zero by the time the march runs
(the scatter consumes them) — the skip fires EVERYWHERE (every cell lit). Both
were caught by the same A/B: (1) showed no speedup, (2) showed a massive
lit/dark inversion. The shipped design needs the map LIVE during the march:
blit-cleared per rebuild, set in gridCount, sized independently of the fine
table.

## M6 render/animation/stereo timings (release, this session — thermally noisy, treat as upper bounds) — 2026-11-11 — Apple M1 Pro

- Offline ray tracer (grown states, --state-load): 800x600 solid 0.03–0.04 s;
  800x600 xray 0.08 s (18 666 cells, 5 big cells).
- Stereo pair (--stereo, 2 eyes + composite): 640x480 solid 0.06 s wall /
  0.06 s gpu (18 666 cells).
- Animation export (--anim, 20 frames at 150-step intervals, light growth
  subdiv 4 → 60 000 cells, 800x600): total wall 10.7–11.3 s of which ray
  render 1.1–1.3 s gpu, rest = growth (2 850 steps) + PNG encode + H.264
  assembly (AVFoundation, 30 fps). X-ray style anim (3 frames, 640x480):
  0.48 s total.
- Static --bench (uniform, division off) this session measured ~3x the M3
  numbers at 655k (31.9 vs 9.9 ms/step) — session/thermal variance, consistent
  with the M5 observation; the interleaved A/B above is the trusted comparison.

## V2.1 render costs (release, 1600x1200, samples 1, --state-load replays; one session) — 2026-11-11 — Apple M1 Pro

Edge layer / coloured X-ray / per-pixel shadow costs, GPU seconds per render
(`commandBuffer.gpuStartTime/gpuEndTime` summed over tiles):

| cells | solid | solid + edges 1 | xray | xray + color | dir. shadows 0 | dir. shadows 1 (4 rays) |
|---|---|---|---|---|---|---|
| 100 000 (m4 grown, folded) | 0.21 | 0.20 | 0.53 | 0.54 | 0.19 | 0.41 |
| 655 362 (settled ball) | 0.62 | 0.62 | 2.14 | 2.15 | 0.61 | 1.77 |

- **Edge layer**: no measurable cost (within run-to-run noise, <= 5%). The
  per-hit 27-bin nearest-centre query is dwarfed by the primary DDA
  traversal (one extra neighbourhood scan per hit pixel vs ~10-50 for the
  ray itself). In the X-ray the per-CONTRIBUTING-SPHERE query (10-50 per
  ray) is likewise invisible against the traversal's own neighbourhood
  scans. The edge layer is effectively free in both styles.
- **Coloured X-ray**: free — the per-channel tint multiply replaces the
  scalar accumulation 1:1 (no extra queries or marches); tone map is 3x
  exp/pow instead of 1x.
- **Per-pixel shadow rays**: +0.22 s (100k) / +1.16 s (650k) for 4 jittered
  cone marches per HIT pixel (~55% of the frame at this framing) => ~60 ns
  per shadow march per pixel at 100k cells, ~300 ns at 655k (longer segments
  through the larger structure; the M6 coarse-occupancy skip applies to
  these marches too). Directional solid total cost ~2.1x (100k) to ~2.9x
  (650k) over shadows-off. `--shadows 0` restores the per-cell-fraction-only
  shading (same cost as the unbiased path + the litFraction read).

## V2.1 fractional light cost (light mode, division OFF, interleaved A/B, 2 passes each; interval-1 sweep and interval-8 sweep) — 2026-11-11 — Apple M1 Pro

Fractional per-cell illumination (lightSoft 0.35) vs the legacy binary
verdict (lightSoft 0). The cost is the LOST EARLY EXIT: the binary march
stops at the first blocker; the fractional march continues until the
visibility product drops below 1e-3 (full-coverage blockers still exit
immediately). Interleaved passes agree tightly (thermal noise shows between
sweeps, not within):

| cells | ms/step interval 1 binary | interval 1 fractional | ratio | ms/step interval 8 binary | interval 8 fractional | ratio |
|---|---|---|---|---|---|---|
| 2 562 | 1.09 / 1.05 | 1.23 / 1.16 | 1.13x | 0.42 | 0.60 | 1.4x |
| 10 242 | 1.35 / 1.35 | 1.87 / 1.87 | 1.39x | 0.56 | 0.58 | 1.0x |
| 40 962 | 4.42 / 4.25 | 8.82 / 8.73 | 2.06x | 1.27 | 1.94 | 1.5x |
| 163 842 | 13.42 / 12.73 | 30.29 / 30.25 | 2.34x | 4.08 / 4.10 | 6.56 / 6.49 | 1.59x |
| 655 362 | 62.0 / 62.3 | 140.2 / 140.6 | 2.26x | 18.0 / 18.2 | 28.2 / 41.6* | ~1.6x |

*the 655k interval-8 fractional pass 2 drifted (thermal); pass 1 (28.2) with
the interleaved binary passes gives ~1.6x, consistent with the 164k level.
At the SHIPPED default (interval 8) the fractional light costs ~1.5-1.6x in
light mode; at interval 1 it is ~2.3x. Non-light modes are untouched (the
march does not run).

## V2.1 light-gradient A/B (growth coupling, 1500 steps, subdiv 4, jitter 0.25, light mode) — 2026-11-11 — Apple M1 Pro

Facing (N.L > 0.3) vs away (N.L < -0.3) mean nutrient after 1500 steps:

| lightSoft | facing | away | ratio |
|---|---|---|---|
| 0 (binary, legacy) | 0.502 | 0.047 | 10.7 |
| 0.35 (V2.1 default) | 0.895 | 0.416 | 2.15 |

The fractional light compresses the gradient BY DESIGN: partially shaded
cells receive a partial nutrient flux instead of a hard 0 (and the lit-side
mean rises because marginally-blocked cells now count as partially lit).
Selftest phase C measures the same field at 2.07-2.09. The phase C gate was
therefore RECALIBRATED (raised, not lowered): > 1.2 -> > 1.5 — the old gate
was calibrated for the binary ~10x ratio and would pass a badly degraded
gradient at the new operating point; 1.5 keeps a ~30% margin under the
measured 2.07-2.15 while still asserting the gradient exists. Documented in
the selftest source next to the gate.

## V2.2 tint modes (food heatmap + mitosis pulse) — 2026-11-11 — Apple M1 Pro

Cost of the V2.2 additions, measured as a four-way `--bench --bench-steps 50`
sweep (uniform mode, division off — the per-step pipeline shape both modes
share): the V2.1 binary (git 00e58ff worktree) as the pre-V2.2 baseline, then
the V2.2 binary with tint auto (= baseline pipeline + the always-on
pulseDecayStep pass), tint food, tint pulse. ms/step GPU
(`commandBuffer.gpuStartTime/gpuEndTime`), CPU wall in parentheses:

| cells | V2.1 baseline | V2.2 auto | V2.2 food | V2.2 pulse |
|---|---|---|---|---|
| 2 562 | 0.156 (0.188) | 0.334 (0.374) | 0.193 (0.224) | 0.159 (0.192) |
| 10 242 | 0.225 (0.256) | 0.302 (0.338) | 0.226 (0.258) | 0.263 (0.296) |
| 40 962 | 0.746 (0.785) | 0.800 (0.839) | 0.791 (0.831) | 0.793 (0.831) |
| 163 842 | 2.540 (2.582) | 2.616 (2.660) | 2.596 (2.640) | 2.626 (2.667) |
| 655 362 | 10.292 (10.335) | 10.296 (10.339) | 10.315 (10.361) | 10.290 (10.337) |

The 2 562 / 10 242 rows are dispatch-and-clock dominated and swing ~2x
between identical sweeps (the same V2.2 binary produced 0.19–0.37 ms/step at
2 562 across the three sweeps above) — read them as noise. At the levels
where the measurement is meaningful the four sweeps agree within ~1–3%
(163 842) and ~0.05–0.2% (655 362): the pulse-decay pass (one extra kernel,
one FMA per cell) and the heatmap/pulse tinting (two extra buffer reads + a
few flops per cellTint pass) cost ~NIL, as designed. Per-step grid memory at
the sweep levels (unchanged from M3/V2.1 — the pulse buffer adds 4 B/cell
outside the grid): T=8192 116 780 B … T=2097152 29 884 460 B for the 655 362
level.

V2.2 mitosis-pulse wavefront metric (checkpoint host readback, tint pulse,
subdiv 4, jitter 0.25, 2000 steps): ring R² of the pulse field
(var(ring average)/var — ~1/valence ≈ 0.17 for an uncorrelated field):

| t | light mode | uniform mode |
|---|---|---|
| 500 | 0.303 | 0.302 |
| 1000 | 0.321 | 0.180 |
| 1500 | 0.331 | 0.209 |
| 2000 | 0.362 | 0.217 |

In uniform growth the recent-division field decorrelates to the iid baseline
(scattered flicker); in light mode it stays clustered and RISES with growth —
divisions remain organized in the lit band that moves with the growing form
(the paper §7 "rhythmic pulses and waves of growth propagating across
surfaces"). Active fraction (pulse > 0.05) at t=2000: 14.1% light vs 6.0%
uniform.

## V2.1b foam rendering (power-diagram chambers) — 2026-11-12 — Apple M1 Pro

Render cost of the foam styles vs their base styles (release, 1600x1200,
samples 1, `--state-load` replays; wall ~= gpu seconds). Foam-solid replaces
the per-candidate sphere test with the interval CSG (<= 16 power-bisector
plane clips per candidate) plus the top-K face insertions; foam-xray keeps
the density walk and adds the chamber clip + wall spikes per candidate.
radiusScale 0.7 everywhere (the foam contract: walls need overlap; the
states' own radii were recomputed with it).

| state | cells | solid | foam | foam/solid | xray | foam-xray | fx/xray |
|---|---|---|---|---|---|---|---|
| grown A (v21b-foam-grown.state) | 83 739 | 0.35 s | 1.57 s | 4.5x | 0.92 s | 1.13 s | 1.2x |
| grown B (bench, 6200 steps) | 221 868 | 0.65 s | 3.45 s | 5.3x | 1.70 s | 2.13 s | 1.25x |
| ball subdiv 8 (static) | 655 362 | 0.83 s | 3.01 s | 3.6x | — | — | — |

The foam-solid ratio (3.6-5.3x) is the CSG's <= 16 plane clips per candidate
against the solid path's single sphere test, amplified by candidate density
along the ray (grown states pack more candidates per ray than the smooth
ball). Interactive impostor foam: whole `--screenshot` run (process start +
state load + one render + PNG; the delta isolates the foam cost incl. the
one-off back-to-front sort) — solid 0.12 s vs foam 0.11 s at 83 739 cells;
solid 0.75 s vs foam 0.78 s at 655 362 cells (the sort of 655k indices is in
the delta: ~tens of ms). The impostor foam's per-fragment cost (<= 16 plane
tests on the cell's own quad) is ~nil against rasterization.

foamWall tuning (foam-xray spike gain, grown A, 800x600): 1.5 saturates the
film curve (membrane web washed out), 0.15 is faint; default set to 0.25.

Retirement note: the V2.1 painted edge layer (`--edges`/`--edge-width`) was
removed in this milestone (the user rejected the painted-on look); the
v21-solid-edges golden (png+meta+state) was deleted; the remaining 11
pre-V2.1b goldens PASS byte-identical (the edge code was a no-op at their
recorded edges=0), and 3 new foam goldens (v21b-foam-ball, v21b-foam-grown
snapshot replay, v21b-foam-imp-ball) capture+check PASS.

## V2.3 cell types + two-phase growth — 2026-11-13 — Apple M1 Pro

Cost of the V2.3 additions, `--bench --bench-steps 50` (uniform mode,
division off — the only per-step change outside the division block is one
extra const buffer read in cellTint):

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 | 0.284 | 0.249 | 3520 |
| 10 242 | 0.332 | 0.300 | 3014 |
| 40 962 | 0.765 | 0.723 | 1307 |
| 163 842 | 2.473 | 2.421 | 404 |
| 655 362 | 9.657 | 9.610 | 104 |

vs the V2.2 binary's tint-auto sweep (benchlog V2.2): within the small-level
noise (the 2 562 / 10 242 rows swing ~2x between identical sweeps) and ~1–7%
at the meaningful levels — the V2.3 per-step additions cost ~NIL with
division off. Grid memory unchanged (the type/counter buffers are 4 B/cell
+ 4 B total, outside the grid).

Decide/foodStep type-check cost, `--bench-growth --bench-steps 50` (division
on). Uniform = the historical rule; typed = the paddenstoel typing (t1 −0.4,
t2 0.9, phaseX 2700, lamelRate 0.5, dalRate 0.25) — in phase 1 the stem
divides only at its band edge and the 65% band stays dormant, so the typed
runs grow much slower; that IS the rule matrix working, not overhead. The
overhead question is the per-step cost at MATCHED counts:

| cells (uniform) | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 -> 3 288 | 0.346 | 0.308 | 2887 |
| 10 242 -> 13 241 | 0.410 | 0.374 | 2442 |
| 40 962 -> 53 179 | 1.092 | 1.047 | 916 |
| 163 842 -> 212 709 | 4.797 | 4.747 | 208 |
| 655 362 -> 851 455 | 23.168 | 23.114 | 43 |

| cells (typed) | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 -> 2 620 | 0.350 | 0.312 | 2855 |
| 10 242 -> 10 436 | 0.367 | 0.333 | 2724 |
| 40 962 -> 41 657 | 0.884 | 0.840 | 1131 |
| 163 842 -> 166 470 | 2.694 | 2.648 | 371 |
| 655 362 -> 662 688 | 16.997 | 16.943 | 59 |

At the identical starting count (level 4, pristine ball) the typed per-step
GPU cost is +1.3% (0.312 vs 0.308) — the decide matrix's type reads and the
foodStep rate classification are ~nil, as designed (a type-0 cell
short-circuits on one buffer read; typed cells scan their own ring once per
decide candidate). The larger typed rows are count-confounded (662k vs 851k
cells), not cost-comparable.

Typed-growth checkpoint table (paddenstoel, subdiv 3, 642 -> 4 536 cells,
4000 steps; per-type counts, division counter, phase):

| t | stem | lamel | dal | cap | Σ | divisions | phase |
|---|---|---|---|---|---|---|---|
| 0 | 191 | 209 | 209 | 33 | 642 | 0 | 1 |
| 500 | 571 | 210 | 209 | 648 | 1 638 | 996 | 1 |
| 1000 | 865 | 226 | 217 | 2 059 | 3 367 | 2 725 | 2 |
| 1500 | 865 | 320 | 257 | 2 059 | 3 501 | 2 859 | 2 |
| 2000 | 865 | 450 | 298 | 2 059 | 3 672 | 3 030 | 2 |
| 2500 | 865 | 680 | 351 | 2 059 | 3 955 | 3 313 | 2 |
| 3000 | 865 | 820 | 439 | 2 059 | 4 183 | 3 541 | 2 |
| 3500 | 865 | 899 | 566 | 2 059 | 4 389 | 3 747 | 2 |
| 4000 | 865 | ~950 | ~620 | 2 059 | 4 536 | ~3 950 | 2 |

Readings: (a) the phase switch lands between t=500 and t=1000 (divCounter
crosses 2700) and the frozen-side counts flip exactly there — stem/cap
freeze in phase 2, lamel+dal freeze in phase 1 (the +9/+8 at t=1000 are the
first phase-2 steps: dormant band cells carry their reset food, uniform
[0, splitThreshold), so the top of the distribution crosses the threshold
within ~20 steps of the switch — prompt, not synchronized); (b) the lamel:dal
count ratio converges to ~1.6:1 under lamelRate:dalRate = 2:1 (the edge/interior
link mix dilutes the raw rate ratio); (c) phase-2 growth does NOT stall on
valence pressure at this scale: at-cap (valence 16) cells stay at 17–39
(0.5–1.2%) through t=4000 and the division rate holds at ~200–280 per 500
steps — the band's own divisions relieve its valence, and the dormant
stem/cap boundary cells ratchet slowly (stem: 0→1 at-cap by t=3500). The
PHASE-1 stem front DOES saturate (the t2=10 isolation probe: growth froze
after 723 divisions, 14 band cells at valence 16) — dormant band cells are
valence sinks; switching to phase 2 (band divisions) relieves it.

## V2.4 — 2026-11-13 — Apple M1 Pro

The differentiation phase: one extra kernel (typeTransition) + one extra
cellTint pass per step while the phase is active. Two bench questions:
(a) the differentiation-PASS cost at matched counts (static ball, rde mode,
division off, phase ON vs OFF — the phase arm adds typeTransition + the
extra tint), and (b) regional growth vs uniform (per-step cost at matched
starting counts; the growth RATE difference is the feature, not overhead —
stem regions grow at drive rdeBase 0.25 vs uniform 1.0, cap regions at
~1.1, so the typed runs grow slower in total).

(a) Differentiation-pass cost, `--bench --growth rde --bench-steps 50`
(division off). Two runs per arm (run-to-run GPU-clock noise dominates the
small levels — the first-level rows swapped signs between runs; the
top-level signal is stable):

| cells | phase OFF gpu ms/step (run1/run2) | phase ON gpu ms/step (run1/run2) |
|---|---|---|
| 2 562 | 0.289 / 0.407 | 0.347 / 0.295 |
| 10 242 | 0.371 / 0.299 | 0.334 / 0.319 |
| 40 962 | 1.071 / 1.014 | 1.045 / 1.047 |
| 163 842 | 3.515 / 3.481 | 3.558 / 3.602 |
| 655 362 | 13.827 / 13.795 | 14.091 / 14.126 |

At the stable top levels the phase adds +1.9–2.4% (655 362 cells) and
+1.2–3.5% (163 842) — two full-size passes (typeTransition reads chemV +
writes cellType; the second cellTint re-tints from the new types) on a
~13.8 ms step. ~nil as designed; below ~50k cells the measurement is
noise-dominated.

(b) Regional growth vs uniform, `--bench-growth --bench-steps 50` (the
regional arm: `--growth rde --differentiate-at 1 --differentiate-until 21`
— a 20-step differentiation phase types the ball from the seeded field, then
the timed window is regional growth; typeMode region). The cells column is
count-confounded BY DESIGN (the regional rate difference is the point —
cf. the V2.3 typed rows): the stem majority (v frozen ~0, drive 0.25) grows
~4x slower than uniform; cap regions (v ~0.35, drive ~1.1) slightly faster.

| cells (uniform) | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 -> 3 288 | 0.451 | 0.405 | 2215 |
| 10 242 -> 13 241 | 0.502 | 0.465 | 1993 |
| 40 962 -> 53 179 | 1.583 | 1.537 | 632 |
| 163 842 -> 212 709 | 7.635 | 7.587 | 131 |
| 655 362 -> 851 455 | 31.950 | 31.899 | 31 |

| cells (regional) | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 -> 2 805 | 0.366 | 0.327 | 2731 |
| 10 242 -> 11 253 | 0.510 | 0.470 | 1960 |
| 40 962 -> 45 126 | 1.426 | 1.381 | 701 |
| 163 842 -> 180 316 | 6.704 | 6.656 | 149 |
| 655 362 -> 721 922 | 28.029 | 27.980 | 36 |

At the identical starting count (level 4) the regional arm is FASTER per
step (0.327 vs 0.405 gpu) because it executes fewer divisions in the window
(243 vs 726) — the region-mode rule matrix itself is a single branch
(typeEligible returns true for every non-zero type; foodStep's rate is 1.0),
i.e. regional growth costs ~nil over uniform at matched counts. The
selftest phase F per-capita table is the rate-difference measurement: over
1000 growth steps, stem ×6.14 vs cap ×94.90 (frozen v 0 vs ~0.35 under
rdeGain 2.5 → ~5x drive ratio, amplified by the exponential division
process).

grid memory rows (T_max = 2^22 preallocated; live T per level):

```
   grid T=8192: binCounts 32768B + binStarts 32772B + particleIndices 10248B + collisionOffsets 40992B = 116780B
   grid T=32768: binCounts 131072B + binStarts 131076B + particleIndices 40968B + collisionOffsets 163872B = 466988B
   grid T=131072: binCounts 524288B + binStarts 524292B + particleIndices 163848B + collisionOffsets 655392B = 1867820B
   grid T=524288: binCounts 2097152B + binStarts 2097156B + particleIndices 655368B + collisionOffsets 2621472B = 7471148B
   grid T=2097152: binCounts 8388608B + binStarts 8388612B + particleIndices 2621448B + collisionOffsets 10485792B = 29884460B
```

## V2.5 starvation death — 2026-11-13 — Apple M1 Pro

Starvation death (the inverse of division, paper §8): a cell whose food
ledger stays below `deathThreshold` for `deathSteps` consecutive steps dies;
the ring is re-spliced Euler-exactly; dead slots stay allocated. The bench
work below started as an open item from the interrupted first V2.5 session:
the 655k forces-only bench was BIMODAL with death armed (~10 vs ~16 ms/step,
interleaved, unexplained). Full investigation record:

### 1. The bimodality — what it was and was not

- NOT the death kernels' executed work: with `--death-steps 1000000` no
  death can execute within the window, the decide pass early-returns per
  cell, and the arm still measured 15–20 ms/step.
- NOT food-init randomness (hypothesis (a) of the interrupted session): the
  GPU food init is seeded via `rngSeed`; and with the death window huge, the
  starving set cannot change any executed work anyway.
- NOT CPU encode starvation: `CG_BENCH_SPLIT=1` measured encode at
  0.013–0.016 ms/step at every level — the gpu window (gpuStartTime..
  gpuEndTime) itself stretched.
- NOT the footprint: `--max-cells 700000` (buffers ~175 MB instead of
  ~500 MB) changed nothing measurable.
- REAL, code-linked (bisect arms, one variable at a time): deleting the
  splice body from `deathExecute` returned the arm to baseline (10.8);
  keeping the body with every per-thread array shrunk to 4 entries was
  STILL slow; a scalar-only body was fast. The statically present splice
  body (ring staging + ear clipping + ring rewrites) alone cost ~+7 ms/step
  at 655k cells — the heavy register footprint collapses co-residency, the
  ~20k threadgroups of a full-width dispatch turn launch-latency bound, GPU
  utilization sags and the whole step inflates ~1.7x. The in-run RATCHET
  (batches 10.9 → 13.9 → 17.7, never recovering within a level) and the
  rare fully-fast runs (1 of ~16) are the DVFS interplay on top; transient
  system-wide windows (even the baseline hit 34–37 ms/step for a few
  minutes — the machine ran with 4.2 GB swap in use, Firefox/Slack/editor
  resident) add the wide spread the first session saw.
- FIX: hot/splice split. `deathExecute` is now guards + claim QUEUEING
  (atomic append); the ear-clip machinery lives in `deathSplice` at a fixed
  narrow width (16 384 threads, whole threadgroups) striding over the
  queue. Queue order is nondeterministic but provably irrelevant: claims
  are pairwise ring-distance >= 3 (decide's independent set), so splices
  touch disjoint neighbourhoods. Verified EMPIRICALLY: two identical seeded
  runs produce BYTE-IDENTICAL CGST v3 states (201 = 201 deaths).

### 2. Determinism chase (side findings, two real bugs)

- `--headless` ran the GUI's UNSEEDED jitter (`SimViewModel.randomInBall`)
  although the file header documents "All headless jitter is seeded" —
  two identical headless invocations started from different positions
  (states differed at step 0). FIXED in V2.5 (headless now uses
  `seededJitter`; `--screenshot` keeps its documented historical unseeded
  behaviour; `--render` was already seeded). This also explains the first
  session's 218-vs-208 death-count mismatch between "identical" runs.
- Grid-scatter arrival order (per-bin atomic placement) makes repulsion
  sums float-order-nondeterministic: with the seed fix, death counts now
  match run-to-run (209 = 209) but positions still differ at ulp level in
  single components. Pre-existing, absorbed by the lockstep (1e-6 mean) and
  golden (pixel-tolerance) gates; death sets are stable.
- Division-ON runs remain macroscopically nondeterministic (9296 vs 9298
  cells after 500 steps): daughter slots come from an atomic fetch_add
  race, and both division's and death's min-index tie-breaking is
  index-sensitive. Pre-existing M2 design, NOT a V2.5 regression; the
  paper does not require bit-reproducibility of slot indices. Follow-up if
  reproducible growth runs are ever needed: deterministic slot handout
  (e.g. claim-index-ordered allocation).

### 3. Death-block cost arm (forces-only, division OFF, death armed with
`--death-steps 1000000` — starvation + decide + queue pass run every step,
no deaths execute; the delta vs the unarmed baseline is the block cost at
matched counts). Two interleaved A/B sweep pairs, same session/binary
(run1/run2):

| cells | baseline gpu ms/step (run1/run2) | death-armed gpu ms/step (run1/run2) |
|---|---|---|
| 2 562 | 0.287 / 0.265 | 0.297 / 0.251 |
| 10 242 | 0.309 / 0.285 | 0.406 / 0.399 |
| 40 962 | 0.941 / 0.943 | 1.076 / 1.084 |
| 163 842 | 3.175 / 3.173 | 3.454 / 3.459 |
| 655 362 | 12.697 / 12.776 | 13.231 / 13.253 |

Delta at the stable levels: +0.11 (10 242), +0.14 (40 962), +0.29
(163 842), +0.48 (655 362) ms/step = +3.7% at the top level. The model
fits: a ~0.1 ms fixed launch floor (4 extra compute encoders + 1 blit per
step) plus linear traffic (starving read/write + claims write + claims
read + the second tint pass ~ 34 B/cell ≈ 22 MB/step at 655k ≈ 0.4 ms at
effective DRAM bandwidth). Re-tint pass ≈ 0.2 ms of the top-level delta
(bisect).

### 4. Measurement protocol (applied above; use for future cost arms)

- Run cost arms as INTERLEAVED A/B pairs (off/death alternating), never as
  isolated sweeps — the absolute floor drifted 10.5 → 12.8 ms/step at 655k
  across this session with system load; only deltas are meaningful.
- `CG_BENCH_BATCHES=1` prints every batch's gpu window + median/min: a
  level whose batches disagree by >2x is contention-tainted (ratchet or
  external interference); prefer the min batch as the exclusive-GPU
  estimate and re-run.
- `CG_BENCH_SPLIT=1` prints encode/wait/gpu-window per level: encode
  ~0.015 ms/step always; wait >> gpu-window would mean CPU starvation
  (never observed); gpu-window >> fast-mode gpu-window means scheduler
  interference, not kernel cost.

Baseline forces-only sweep (division off), full prescribed columns
(death-armed sweep = the table above):

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 | 0.301 | 0.265 | 3321 |
| 10 242 | 0.320 | 0.285 | 3122 |
| 40 962 | 0.985 | 0.943 | 1015 |
| 163 842 | 3.228 | 3.173 | 310 |
| 655 362 | 12.826 | 12.776 | 78 |

Death-armed (window 1e6 — the block cost arm):

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 | 0.295 | 0.251 | 3390 |
| 10 242 | 0.443 | 0.399 | 2256 |
| 40 962 | 1.132 | 1.084 | 884 |
| 163 842 | 3.512 | 3.459 | 285 |
| 655 362 | 13.310 | 13.253 | 75 |

grid memory rows (T_max = 2^22 preallocated; live T per level; identical
for both arms — the death buffers add 8 B/cell + one 4 B counter, all
preallocated at maxCells):

```
   grid T=8192: binCounts 32768B + binStarts 32772B + particleIndices 10248B + collisionOffsets 40992B = 116780B
   grid T=32768: binCounts 131072B + binStarts 131076B + particleIndices 40968B + collisionOffsets 163872B = 466988B
   grid T=131072: binCounts 524288B + binStarts 524292B + particleIndices 163848B + collisionOffsets 655392B = 1867820B
   grid T=524288: binCounts 2097152B + binStarts 2097156B + particleIndices 655368B + collisionOffsets 2621472B = 7471148B
   grid T=2097152: binCounts 8388608B + binStarts 8388612B + particleIndices 2621448B + collisionOffsets 10485792B = 29884460B
```

### 5. Follow-ups surfaced by this investigation

- `divisionExecute` still carries the same-shape risks at full width: a
  same-address `atomic_load` guard (plus a second one mislabelled "plain
  load" in the capacity pre-check) and the heavy inline splice body. In the
  M5/V2.4 growth benches the division block costs ~+12–14 ms/step at 655k
  over forces-only; part of that may be the same occupancy/launch-latency
  effect. NOT touched in V2.5 (M2 core; needs its own validation cycle) —
  the hot/splice split pattern is proven now and transfers directly.
- `boundMax` does a same-address `atomic_fetch_max` at full width (light
  mode, every `lightInterval` steps) — same atomic class, ~1% amortized at
  655k against the march's cost; a two-stage reduction would remove it.
- Dead-slot compaction (reindex + swap-removal with ring rewrite) — the
  documented fragmentation trade-off, still open.
- GUI death sliders (death parameters are CLI/preset-side for now).

## V2.6 camera choreography + video export — 2026-11-13 — Apple M1 Pro

Sim step UNCHANGED by construction: V2.6 touches no step-routine code, no
shader, no SimParams field, no buffer — the camera choreography is pure
host-side viewer state (target/yaw + a two-pass live-centroid read per
frame). One confirmation sweep (same prescribed config as the V2.5 tables:
50 steps/level, jitter 0, division off, repulsion on) — the floor matches
the V2.5 baseline within session noise (655k: 12.77 vs 12.83 ms/step; only
deltas are meaningful, absolute floors drift with system load):

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 | 0.327 | 0.287 | 3060 |
| 10 242 | 0.343 | 0.305 | 2920 |
| 40 962 | 0.959 | 0.921 | 1043 |
| 163 842 | 3.218 | 3.170 | 311 |
| 655 362 | 12.765 | 12.717 | 78 |

grid memory rows (T_max = 2^22 preallocated; live T per level; identical to
V2.5 — no new buffers):

```
   grid T=8192: binCounts 32768B + binStarts 32772B + particleIndices 10248B + collisionOffsets 40992B = 116780B
   grid T=32768: binCounts 131072B + binStarts 131076B + particleIndices 40968B + collisionOffsets 163872B = 466988B
   grid T=131072: binCounts 524288B + binStarts 524292B + particleIndices 163848B + collisionOffsets 655392B = 1867820B
   grid T=524288: binCounts 2097152B + binStarts 2097156B + particleIndices 655368B + collisionOffsets 2621472B = 7471148B
   grid T=2097152: binCounts 8388608B + binStarts 8388612B + particleIndices 2621448B + collisionOffsets 10485792B = 29884460B
```

Video-export costs (reference config: `--anim` 8 frames × 150 steps, subdiv
3, jitter 0.25, solid, 1600×1200, H.264 .mp4 @30 fps; the state GROWS
642 → 185 797 cells across the frames, so the per-frame cost covers that
range):

- render (offline ray tracer, solid): 2.95 s GPU total ≈ **0.37 s/frame**
  (frames 0.04–0.71 s, tracking the cell count)
- movie encode (AVFoundation, BGRA→yuv420p): **0.30 s / 8 frames ≈ 38 ms/frame**
- camera choreography (live-centroid + framing + orbit, host): **0.11 ms/frame**
  at up to ~186k cells (two O(live) pointer passes, no allocation) — ~0.4 ms
  extrapolated at 655k; the GUI follows every rendered frame, ~1% of its
  frame budget at 60 fps
- small-config cross-check (800×600, ~9k cells): ~0.05 s/frame render,
  0.17–0.38 s encode per 8-frame movie; follow vs legacy arms differ by
  <0.1 ms/frame in choreography (the framing pass runs in both arms)

GUI launch check: app runs (12 s alive, no crash) with the new Camera +
Video export groups; the button itself was not click-tested (no screen
recording permission in the agent context) — it calls the same
`exportAnimation` core the CLI runs validated above.

## V2.7 interactive video export + death sliders — 2026-11-14 — Apple M1 Pro

GUI-only milestone: NO step-routine, shader, SimParams or buffer change —
the sim step is unchanged by construction. Confirmation sweep, prescribed
config (50 steps/level, jitter 0, division off, repulsion roi 1.6 / 0.1,
growth uniform food 0.01 / threshold 1.0, death off), `CG_BENCH_BATCHES=1`.
System load ~6.9 at sweep time (multi-user machine) but the batches are
tight (min ≈ median at every level) — clean exclusive-GPU numbers, matching
the fast-mode floor from the V2.5 bimodality investigation (655k: 9.72 here
vs 9.65–10.86 there; the V2.6 12.77 sweep ran inside the loaded window):

| cells | ms/step cpu | ms/step gpu | steps/s |
|---|---|---|---|
| 2 562 | 0.359 | 0.315 | 2787 |
| 10 242 | 0.361 | 0.327 | 2767 |
| 40 962 | 0.798 | 0.755 | 1254 |
| 163 842 | 2.513 | 2.462 | 398 |
| 655 362 | 9.774 | 9.720 | 102 |

Per-batch gpu windows at 655 362: 9.753 9.727 9.736 9.698 9.687 | median
9.727 min 9.687. Grid memory rows identical to V2.5/V2.6 (no new buffers).

V2.7-specific costs (probes by the milestone agent):
- `CellSim.cloneState` (the byte-exact export clone): **8–10 ms at ~200k
  live cells** — one-off per export, not per frame/step.
- Export frame/encode costs are the V2.6 numbers (shared `exportAnimation`
  core, unchanged): ~0.37 s/frame render at 1600×1200 offline solid,
  ~38 ms/frame AVFoundation encode; the GUI adds preview scaling only.
- Export-sim capacity estimate keeps the clone's maxCells at expected
  growth — a second full 2M buffer set (~400 MB) is only paid when the
  estimate demands it.

Validation (agent + independent orchestrator re-run): build debug+release,
`--selftest` 7 phases PASS, `--lockstep` PASS (mean |ΔP| 1.1e-6),
14/14 goldens byte-PASS, CLI `--anim` regression → valid H.264 .mp4
(ffprobe), abort probe → valid partial movie, capacity-watchdog probe,
clone/continuation byte-exactness probes, GUI launch check. Open point:
the GUI export button itself was not click-tested (no screen-recording
permission in the agent context).

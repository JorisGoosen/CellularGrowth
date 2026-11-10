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

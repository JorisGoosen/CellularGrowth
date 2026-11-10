# CellularGrowth

Reimplementation of Andy Lomas' **"Cellular Forms: an Artistic Exploration of
Morphogenesis"** (AISB50) on Metal / Apple Silicon.

The exact paper extraction (formulas verified against hi-res page renders,
including the `bulgeDist` typo analysis) lives in
[`docs/reference/lomas_cellular_forms_extraction.md`](docs/reference/lomas_cellular_forms_extraction.md),
with page screenshots in `docs/reference/pages/`.

## Status — M4: offline ray-traced rendering (ambient occlusion + X-ray)

- M1–M3 base unchanged (forces, food + division, spatial-hash repulsion,
  `--selftest`/`--lockstep`/golden machinery). The interactive raster
  renderer and both M3 goldens are untouched.
- **Offline ray tracer** (M4, `Render/RayRenderer.swift` + the `raytrace`
  MSL in `Shaders.swift`): compute-kernel ray tracing of the cells-as-spheres
  model (paper §5) through the simulation's OWN spatial hash (bin size = roi),
  built once for the final state (`CellSim.encodeGridRebuild`). Amanatides–Woo
  3D-DDA along each ray; per visited bin the K-bin neighbourhood is queried
  (K = 1 for hits, K = ceil(profileSupport) for density — the traversal-proof
  needs the crossing point within K bins of the cell's centre bin). Cells with
  radius >= roi (division-stretched cells) leave the grid's correctness domain
  and are brute-forced per ray from a small supplement list.
- **Solid style** (paper §6 style 1, Fig. 8): nearest sphere hit, Lambertian
  reflectance under a uniform omnidirectional diffuse light — with an
  isotropic light field the Lambertian irradiance integral reduces to a pure
  visibility quantity, implemented as ANALYTIC solid-angle ambient occlusion
  (per neighbour: sin^2(alpha)·max(N·C,0), sin alpha = rad/d, summed within
  aoRadius; noise-free and ~30x cheaper than hemisphere ray sampling). Tuned
  (aoRadius 9.6, aoStrength 0.35, aoFloor 0.18) to the pale sculptural
  Fig. 8 look on a paper-white background.
- **X-ray style** (paper §6 style 2, Fig. 9): per-ray accumulated density —
  every sphere within profileSupport·radius of the ray segment contributes
  its smooth radial profile; each sphere is counted exactly once via the
  entry-anchor rule (the visited bin whose [tEnter,tExit) contains the
  clamped foot — the per-bin ranges partition the ray, so the anchor selects
  exactly one bin). Film-curve tone map 1−exp(−x·exposure) to grayscale.
  Matches the Fig. 9 microscopy look (ghostly translucent envelope, wispy
  folded sheets, bright knots).
- **Validation**: the render kernels are checked against a CPU brute force
  (CGDBG=1: 64 pixels re-computed over ALL cells — validates the DDA, the
  anchor dedup and the AO sum; X-ray agrees at the byte-quantization floor,
  solid within near-tangent-hit conditioning). Two grown-state goldens
  (`m4-grown-solid`, `m4-grown-xray`) store the final structure in a state
  snapshot — the growth itself is NOT run-to-run reproducible (daughter slot
  allocation is GPU-schedule-dependent; measured: 43.7% of pixels differ
  between two identical growth runs) — and `--golden-check` replays the
  render pixel-exactly (max diff 0/255).
- 8600×8600 artefact resolution renders in 7.5 s (solid) / 21 s (X-ray) via
  automatic 1024² tiling (paper §6 artefact size).

The M3 status (spatial-hash repulsion) is unchanged; see the M3 section of
git history or docs/benchlog.md for details:
forces + food + division + hashed-grid repulsion, `--selftest` invariants,
`--lockstep` CPU golden reference, deterministic raster goldens.

## Build & run

```bash
swift run CellularGrowth                      # GUI app
scripts/make_app.sh && open CellularGrowth.app  # double-clickable bundle
```

Headless validation:

```bash
# invariant selftest: control phase (strict settle gate) + growing run
# (topology exact at every checkpoint); exit code 0 only if all pass
swift run -c release CellularGrowth --selftest

# growing run: division on by default, stats printed every 500 steps
swift run -c release CellularGrowth --headless --steps 3000 --subdiv 3 --jitter 0.25

# M1-style stability check (division disabled)
swift run -c release CellularGrowth --headless --steps 3000 --subdiv 3 --jitter 0.25 --no-div

# offscreen screenshot of the state after N steps (division on by default)
swift run -c release CellularGrowth --screenshot /tmp/growth.png --steps 1000 --subdiv 4
```

Flags: `--steps N`, `--subdiv N`, `--jitter F`, `--spring F`, `--planar F`,
`--bulge F`, `--food F`, `--threshold F`, `--seed N`, `--no-div`,
`--max-cells N`, `--screenshot PATH`, `--bench`, `--bench-steps N`,
`--bench-growth`, `--selftest`, `--roi F`, `--repulsion F`, `--lockstep`,
`--golden-capture NAME`, `--golden-check NAME`, `--render PATH`,
`--style solid|xray`, `--width N`, `--height N`, `--tiled`, `--samples N`,
`--state-save PATH`, `--state-load PATH`. Diagnostic: `--div-stop N`
(disable division after step N; used to separate growth effects from
settling).

```bash
# M4: ray-traced renders of the grown state (paper Fig. 8 / Fig. 9 styles)
swift run -c release CellularGrowth --render form.png --style solid \
    --width 1600 --height 1200 --steps 2400 --subdiv 3 --max-cells 100000
swift run -c release CellularGrowth --render xray.png --style xray \
    --width 8600 --height 8600 --state-load state.bin
```

## Controls

Drag to orbit, scroll/pinch to zoom. Sliders: spring/planar/bulge factors,
radius scale, food increment, split threshold, cell division toggle, steps per
frame, jitter and subdivision (applied on Reset).

## Roadmap

| Milestone | Content | Validation |
|---|---|---|
| M1 ✅ | forces, stable ball | headless stats + screenshot vs paper Fig. 3 |
| M2 ✅ | food levels + cell division (topology change) | `--selftest` invariants + growing screenshots; expected degeneration without repulsion (paper Fig. 5) |
| M3 ✅ | spatial-hash grid + repulsion | `--lockstep` probe/trajectory gates + goldens; Fig. 5 (degeneration without) vs Fig. 6 (coherent with) screenshots |
| M4 ✅ | offline ray tracer: solid + analytic AO, X-ray density accumulation | CPU brute-force pixel verification (CGDBG), grown-state snapshot goldens (pixel-exact replay), paper-gate comparison vs Fig. 8/9, 8600² tiled render |
| M5 | growth modes: light-based nutrient (+ diffusion), then RDE (Gray-Scott) | paper §6 styles |
| M6 | app polish: state save/load, animation export, stereo pairs, presets | — |

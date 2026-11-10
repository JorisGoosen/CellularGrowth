# CellularGrowth

Reimplementation of Andy Lomas' **"Cellular Forms: an Artistic Exploration of
Morphogenesis"** (AISB50) on Metal / Apple Silicon.

The exact paper extraction (formulas verified against hi-res page renders,
including the `bulgeDist` typo analysis) lives in
[`docs/reference/lomas_cellular_forms_extraction.md`](docs/reference/lomas_cellular_forms_extraction.md),
with page screenshots in `docs/reference/pages/`.

## Status — M6 complete: app polish (all milestones ✅)

**M1–M5 base** (details in git history / docs/benchlog.md): over-damped
spring/planar/bulge forces (paper §4, Fig. 3); food + cell division with
race-free claim/execute topology change; hashed sparse-grid repulsion
(`--roi`, `--repulsion`; without it forms degenerate — paper Fig. 5 vs 6);
offline ray tracer (solid + analytic ambient occlusion = paper Fig. 8 look;
X-ray density accumulation = Fig. 9 look; CPU brute-force pixel verification;
8600² tiled renders); three growth modes `--growth uniform|light|rde`
(uniform random / light-created nutrient with link diffusion + decay — low
diffusion grows plant-like forms toward the light, high diffusion organ-like /
Gray-Scott RDE on the cell graph driving patterned growth, `--rde-probe`
validates pattern formation); ONE unified light direction shared by growth
illumination and rendering (`--render-light auto|growth|unbiased`); RDE
chemical tinting in both render paths.

M6 additions:
- **Interactive X-ray** (paper §6 style 2, realtime): the GUI gains a
  Solid / X-ray render-style switch. X-ray renders every cell as an additive
  density sprite (same radial profile as the offline accumulation,
  `rad·(1−(d/(rad·S))²)²` with S = 2) with additive blending and no depth —
  dense knots glow on a black background, chemical tint included. Documented
  approximation: the film curve is nonlinear in the *accumulated* density;
  fixed-function additive blending sums per-sprite terms in 8-bit sRGB and the
  clamp is the tone map (`xrayGain` slider, default 0.3). The default Solid
  path is pixel-identical to M5 (all goldens pass without recapture).
- **State save/load unified (CGST v3)**: one writer/reader for CLI, goldens
  and GUI. v3 persists EVERYTHING — positions, links, linkCounts, food,
  nutrient, u, v, tint, rngState + the full SimParams — so a saved state
  *continues* exactly where it left off. v1 (M4) and v2 (M5) files still load
  (v2 restores the growth config and reseeds chemicals; v1 keeps current
  params). Round-trip validation: grow → save → load → render is
  byte-identical to the pre-save render, and save→load→save reproduces the
  file bit-for-bit. GUI: Save State…/Load State… (NSSavePanel/NSOpenPanel,
  default directory ~/Documents); CLI `--state-save`/`--state-load` now
  round-trip chemicals everywhere (including `--screenshot`).
- **Animation export** (paper §6): `--anim DIR --anim-steps N --anim-frames M`
  renders equally spaced frames with the offline ray tracer (style/size/
  samples/`--render-light` configurable) into numbered PNGs and assembles an
  H.264 .mov via AVFoundation (30 fps; if the encoder fails the PNG sequence
  is still written and an ffmpeg one-liner is printed). Camera re-frames per
  frame so the growing structure stays in view.
- **Stereo pairs** (paper §6): `--stereo` renders left/right eyes with the
  camera shifted ±interocular/2 along the camera right axis (parallel-axis
  stereo, no toe-in) and composites a side-by-side PNG; separation is a
  fraction of the scene scale (`--interocular F`, default 0.03).
- **Presets**: `ball`, `brain`, `plant`, `coral`, `degenerate-demo` — full
  parameter bundles (Core/Presets.swift, one source of truth) with documented
  values; GUI dropdown and CLI `--preset NAME` (explicit flags still win).
- **Light-march occupancy skip** (M5 open issue): coarse 8³-bin occupancy
  bitfield + DDA block jumps make light mode ~4x faster at interval 1 and
  ~2.5x at the default interval 8 (655k cells; interleaved A/B in
  docs/benchlog.md). Skip proven trajectory-exact by a byte-identical A/B.
- **Determinism fix (found by the M6 A/B)**: Icosphere ring ordering used
  Swift `Set` iteration order (randomized per process) as the angle-sort
  reference — every ring's cyclic rotation differed run to run. Now
  deterministic (min-index reference); the M3 goldens still pass within their
  recorded tolerance, no recapture needed.
- **Division/tint fix (found by the round-trip validation)**: `cellTint` now
  runs after the division block, so same-step daughters no longer render with
  a stale (zeroed = black) tint.
- **Validation**: `--selftest` (4 phases) PASS, `--lockstep` PASS, all six
  goldens PASS (m3 ×2 unchanged, m4 ×2 unchanged, m6 impostor-X-ray ball +
  grown added), `--headless` sane.

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
`--state-save PATH`, `--state-load PATH`. M5: `--growth uniform|light|rde`,
`--light-dir "x,y,z"`, `--light-diffuse F`, `--light-nutrient F`,
`--light-decay F`, `--light-base F`, `--light-interval N`, `--rde-F F`,
`--rde-k F`, `--rde-du F`, `--rde-dv F`, `--rde-base F`, `--rde-gain F`,
`--rde-probe`, `--rde-dump DIR`, `--render-light auto|growth|unbiased`.
M6: `--preset ball|brain|plant|coral|degenerate-demo`, `--anim DIR
--anim-steps N --anim-frames M`, `--stereo`, `--interocular F`,
`--screenshot-style solid|xray`. Diagnostic: `--div-stop N` (disable division
after step N); env `CG_NO_COARSE_SKIP=1` disables the light-march occupancy
skip (A/B kill switch).

```bash
# M4: ray-traced renders of the grown state (paper Fig. 8 / Fig. 9 styles)
swift run -c release CellularGrowth --render form.png --style solid \
    --width 1600 --height 1200 --steps 2400 --subdiv 3 --max-cells 100000
swift run -c release CellularGrowth --render xray.png --style xray \
    --width 8600 --height 8600 --state-load state.bin
```

## Controls

Drag to orbit, scroll/pinch to zoom. Panel: preset dropdown (ball / brain /
plant / coral / degenerate-demo), Solid/X-ray render-style switch (+ X-ray
gain), Save State…/Load State…, growth mode + light/RDE parameter groups,
repulsion, forces, run controls. Jitter and subdivision apply on Reset.

## Roadmap

| Milestone | Content | Validation |
|---|---|---|
| M1 ✅ | forces, stable ball | headless stats + screenshot vs paper Fig. 3 |
| M2 ✅ | food levels + cell division (topology change) | `--selftest` invariants + growing screenshots; expected degeneration without repulsion (paper Fig. 5) |
| M3 ✅ | spatial-hash grid + repulsion | `--lockstep` probe/trajectory gates + goldens; Fig. 5 (degeneration without) vs Fig. 6 (coherent with) screenshots |
| M4 ✅ | offline ray tracer: solid + analytic AO, X-ray density accumulation | CPU brute-force pixel verification (CGDBG), grown-state snapshot goldens (pixel-exact replay), paper-gate comparison vs Fig. 8/9, 8600² tiled render |
| M5 ✅ | growth modes: light-based nutrient (+ diffusion), Gray-Scott RDE; unified light + chemical tinting | `--rde-probe` pattern-formation gates (variance/R²/stability), `--selftest` phases C/D, directional-vs-unbiased render side-by-side, per-mode bench |
| M6 ✅ | app polish: interactive X-ray, CGST v3 state save/load, animation export, stereo pairs, presets, light-march occupancy skip (~4x) | `--selftest` 4 phases, `--lockstep`, 6/6 goldens (no recapture), state round-trip byte-identical, 20-frame animation + H.264, stereo parallax check, interleaved occupancy A/B (benchlog) |

# CellularGrowth

Reimplementation of Andy Lomas' **"Cellular Forms: an Artistic Exploration of
Morphogenesis"** (AISB50) on Metal / Apple Silicon.

The exact paper extraction (formulas verified against hi-res page renders,
including the `bulgeDist` typo analysis) lives in
[`docs/reference/lomas_cellular_forms_extraction.md`](docs/reference/lomas_cellular_forms_extraction.md),
with page screenshots in `docs/reference/pages/`.

## Status — V2.1 complete: edge layer, coloured X-ray, soft shadows + fractional light

**M1–M6 base** (details in git history / docs/benchlog.md): over-damped
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
chemical tinting in both render paths; interactive X-ray, CGST v3 state
save/load, animation export, stereo pairs, presets, light-march occupancy
skip.

V2.1 additions:
- **Edge layer** (`--edges F`, `--edge-width F`, default off): a wall-darkening
  blend parameter (NOT a separate style) in all three render paths. At every
  query point the darkening ramps over `edgeWidth × radius` from the wall
  plane (nearest other cell centre == own centre distance) into the cell
  interior: ray-tracer solid = grid nearest-centre query at the primary hit;
  ray-tracer X-ray = per contributing sphere at the density-profile foot
  (faint cell outlines woven into the density cloud — the microscopy look);
  interactive impostors = the cell's own ring neighbours (≤ 16 fragment
  reads). `solid + edges 1` gives the Voronoi-wall look the paper mentions
  as an early experiment (§8: "a variation on Voronoi cells"); `X-ray +
  edges` reads like stained microscopy.
- **Coloured X-ray** (`--xray-color`, ray tracer): per-channel tinted density
  accumulation — contribution = the cell's tint (cellTint: RDE chemistry
  colours) × its density term, tone mapped per channel. RDE spots glow warm
  through the teal u-field. The grayscale path is bit-identical to M4.
- **Soft shadows + fractional light**: the growth illumination is now a
  FRACTION per cell — the visible fraction of a soft light disk (angular
  radius `--light-soft F` radians, default 0.35 ≈ 20°; 0 = the legacy binary
  verdict) instead of a lit/unlit flag. One DDA march, same candidate set:
  each centerline-crossing blocker occludes a linear-ramp fraction of the
  disk; independent blockers multiply. Consumers: the growth coupling
  (smoother lit/dark gradient), the impostor fragment (per-cell shadow
  factor), and the ray tracer's directional solid style, where per-pixel
  shadow rays (`--shadows F`, 4 jittered cone samples, `--shadows 0` restores
  the per-cell-only shading) refine it into a penumbra. Shadows exist ONLY
  under the directional light — unbiased renders never trace shadow rays
  (all M3/M4/M6 goldens byte-identical, verified). AO stays the
  omnidirectional term: `col = tint·(aoFloor + (1−aoFloor)·ao)·(0.2 +
  0.8·N·L·vis)`. The selftest phase C gradient gate was recalibrated for the
  softer gradient (raised 1.2 → 1.5; measured ratio 2.07–2.15, binary was
  ~10.7 — see the selftest comment).
- **Validation**: `swift build` (debug+release), `--selftest` (4 phases,
  recalibrated C gate) PASS, `--lockstep` PASS, all 6 previous goldens PASS
  byte-identical (no recapture), 3 new goldens (directional solid with
  shadows, coloured X-ray of a grown RDE state, solid with edges=1) capture +
  check PASS.

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
`--screenshot-style solid|xray`. V2.1: `--edges F` (edge-layer strength,
0 = off), `--edge-width F` (wall width in cell radii, default 0.12),
`--shadows F` (per-pixel shadow strength, directional solid only, default 1),
`--xray-color` (per-channel tinted X-ray), `--light-soft F` (light angular
radius in radians, default 0.35, 0 = legacy binary illumination). Diagnostic:
`--div-stop N` (disable division after step N); env `CG_NO_COARSE_SKIP=1`
disables the light-march occupancy skip (A/B kill switch); env
`CG_SHADOW_SAMPLES=N` overrides the 4 default shadow rays per pixel (bench
knob — goldens record 4).

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
| V2.1 ✅ | render v2: edge-blend layer in all three paths (`--edges`), coloured X-ray (`--xray-color`), soft shadows + fractional per-cell light (`--shadows`, `--light-soft`) | `--selftest` 4 phases (recalibrated C gate), `--lockstep`, 6/6 legacy goldens byte-identical (no recapture), 3 new goldens capture+check, render-cost bench (edges ~free, shadows ~2–3x directional solid, coloured X-ray ~free), fractional-light A/B (benchlog) |

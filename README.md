# CellularGrowth

Reimplementation of Andy Lomas' **"Cellular Forms: an Artistic Exploration of
Morphogenesis"** (AISB50) on Metal / Apple Silicon.

The exact paper extraction (formulas verified against hi-res page renders,
including the `bulgeDist` typo analysis) lives in
[`docs/reference/lomas_cellular_forms_extraction.md`](docs/reference/lomas_cellular_forms_extraction.md),
with page screenshots in `docs/reference/pages/`.

## Status — M5: growth modes (light nutrient, Gray-Scott RDE) + unified light

- M1–M4 base unchanged. Uniform random growth (paper §4 method 1) is the
  default and is bit-identical to the pre-M5 semantics (the RNG stream is
  consumed exactly once per cell per step in every mode; `--lockstep` and all
  goldens still pass).
- **Three food/growth modes** (`--growth uniform|light|rde`, paper §4):
  - **light** (method 3): per-cell illumination — a ray is marched from each
    cell TOWARD the light through the simulation's own spatial hash
    (Amanatides–Woo 3D-DDA, same traversal pattern as the M4 renderer); a cell
    with no blocking sphere is lit and produces nutrient
    (`lightNutrient`, pulsed every `--light-interval` steps with proportional
    amplitude — the shadow pattern evolves on ~100-step scales, so the march
    is amortized; decay + diffusion stay per-step). Nutrient diffuses along
    links (Jacobi) and decays first-order (`--light-decay`); the food
    increment is ∝ (lightBase + nutrient). LOW diffusion (0.02) + single
    direction → plant-like elongated growth toward the light; HIGH diffusion
    (0.3) → organ-like folded forms (paper §7).
  - **rde** (method 2): Gray-Scott à la Turk on the cell graph — chemicals
    u, v stored per cell, diffusing along links (graph Laplacian
    avg(ring) − x), explicit Euler dt=1, defaults F=0.037, k=0.06,
    Du=0.35, Dv=0.15 (halved from the classic mixing weights so the pattern
    wavelength localizes on small domains — measured at `--rde-probe`).
    The v chemical drives the food increment (∝ rdeBase + rdeGain·v).
    `--rde-probe` validates pattern formation on a static ball (variance
    amplification, substrate consumption, spatial organization R², stability).
  - **Chemical tinting**: in rde mode cells render tinted — base green,
    toward blue ∝ u, toward red ∝ v (weights 0.30/1.60 on clamped u/v; the
    neutral pre-M5 albedo elsewhere) — in BOTH render paths (per-cell tint
    buffer filled by the `cellTint` kernel every step).
- **Unified light**: ONE light direction (SimParams lightDirX/Y/Z, default
  the pre-M5 impostor direction (0.55, 0.75, 0.4)) drives BOTH the growth
  illumination AND rendering. The interactive impostor path shades with it;
  the offline solid style gains `--render-light auto|growth|unbiased` —
  auto selects the directional shading in light-growth mode (shading with the
  per-cell illumination verdicts the growth itself computed — lit cells bright,
  shadowed dark), unbiased (the paper's omnidirectional look, default) keeps
  all goldens byte-identical. X-ray is density-based (no lighting).
- **State snapshots v2**: the CGST format now carries the growth config
  (mode + light/rde coupling), so `--state-load` renders/continues with the
  mode that grew the state; v1 files (M4 goldens) still load.
- **Validation**: `--selftest` gains phases C (light growth: invariants +
  lit/dark nutrient gradient gate) and D (rde growth); `--lockstep` unchanged
  (forces growthMode 0 + division off — the new kernels write only the
  chemical buffers, never positions/links); all goldens pass byte-identically.

The M4 status (offline ray tracer) is unchanged; see the M4 section of
git history or docs/benchlog.md for details: solid + analytic AO, X-ray
density accumulation, CPU brute-force pixel verification, grown-state
snapshot goldens, 8600² tiled renders.

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
| M5 ✅ | growth modes: light-based nutrient (+ diffusion), Gray-Scott RDE; unified light + chemical tinting | `--rde-probe` pattern-formation gates (variance/R²/stability), `--selftest` phases C/D, directional-vs-unbiased render side-by-side, per-mode bench |
| M6 | app polish: state save/load, animation export, stereo pairs, presets | — |

# CellularGrowth

Reimplementation of Andy Lomas' **"Cellular Forms: an Artistic Exploration of
Morphogenesis"** (AISB50) on Metal / Apple Silicon.

The exact paper extraction (formulas verified against hi-res page renders,
including the `bulgeDist` typo analysis) lives in
[`docs/reference/lomas_cellular_forms_extraction.md`](docs/reference/lomas_cellular_forms_extraction.md),
with page screenshots in `docs/reference/pages/`.

## Status — V2.1b complete: foam rendering (real flat walls, power-diagram chambers)

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

V2.1/V2.2 additions (soft shadows + fractional per-cell light `--light-soft`/
`--shadows`, coloured X-ray `--xray-color`, food-heatmap and mitosis-pulse
tint modes `--tint food|pulse`) are described in git history and the V2.1/V2.2
entries of docs/benchlog.md. The V2.1 painted **edge layer** (`--edges`,
`--edge-width`) was RETIRED in V2.1b: a darkening blend painted near walls
reads as drawn outlines, not geometry — the foam walls below are the wall
visualization now (the v21-solid-edges golden was retired with it).

V2.1b additions — **foam rendering** (`--style foam`, `--style foam-xray`):
- Each cell renders as its **power-diagram chamber**:
  `sphere(c_i, r_i) ∩ ⋂_{j∈ring(i)} {x : |x−c_i|²−r_i² ≤ |x−c_j|²−r_j²}` —
  the sphere truncated by the POWER-BISECTOR half-spaces toward its ring
  neighbours. Power bisectors are PLANES, so walls between pressed cells are
  truly flat and sit fairly between unequal cells (unlike chord-midpoint
  Voronoi). The chamber math lives once in the shared `common` shader chunk
  (`foamChamberInterval`/`foamWallNormal`); both renderers and the shadow
  march use it.
- **Walls are visible because foam is translucent**: the union of chambers
  equals the union of spheres exactly (tiling proof in Shaders.swift), so an
  opaque render would show only curved caps — the renderers composite the
  chamber ENTRY faces front-to-back with a per-face opacity
  (`CG_FOAM_OPACITY`, default 0.5): the flat walls show through the curved
  front caps — walls behind walls, the soap-froth depth. Every wall crossed
  by a ray is provably the following chamber's entry, so each wall is shaded
  exactly once, with the two cells' AVERAGE tint and illumination (the ray
  tracer and the impostors shade the shared wall identically — no
  z-fighting: the bisector planes are derived from the lower cell index, so
  both sides compute bit-identical planes).
- **Ray tracer** (`--style foam`): per candidate the ray-sphere interval is
  clipped against the ≤16 ring bisectors (interval CSG), the nearest K
  chamber entries (K = `CG_FOAM_LAYERS`, default 6) are composited with the
  existing AO + directional soft shadows (the shadow march skips BOTH cells
  of a wall). `--style foam-xray`: the X-ray density follows the CHAMBER
  intervals (skirt cells vanish) plus an obliquity-weighted spike at every
  wall crossing (`CG_FOAM_WALL`, default 0.25) — internal membranes in the
  microscopy look.
- **Interactive impostors**: the fragment intersects the pixel's exact view
  ray with the cell's chamber and shades the first face (cap or wall);
  premultiplied over-blending with the instances sorted back-to-front
  (throttled CPU sort: every frame ≤ 20k cells, every 30th above). Replaces
  the retired edge layer.
- **Radius scale contract**: a wall has real area only when neighbouring
  spheres OVERLAP — foam mode recommends `--radius-scale 0.7` (the GUI hint
  says the same); at the 0.5 default the spheres merely touch and the walls
  degenerate to points (the foam then reads as the plain sphere union).
- **Validation**: `swift build` (debug+release), `--selftest` (4 phases)
  PASS, `--lockstep` PASS, all 11 surviving goldens byte-PASS (the
  v21-solid-edges golden retired — its config is no longer renderable), 3
  new goldens (ray-traced foam ball, ray-traced foam grown state with
  byte-exact snapshot replay, impostor foam ball) capture + check PASS;
  cross-renderer consistency (the same 84k grown state through the impostor
  foam and the ray-traced foam reads as the same object); foam cost bench
  (benchlog V2.1b: ray foam ≈ 3.6–5.3× solid, foam-xray ≈ 1.2× xray,
  impostor foam ≈ solid cost + a one-off sort).

V2.1 additions (the edge layer described here was RETIRED in V2.1b — see the
status header; the other two are live features):
- ~~**Edge layer** (`--edges F`, `--edge-width F`)~~ — RETIRED in V2.1b: a
  darkening blend painted near walls read as drawn outlines, not geometry
  (the user rejected it). The foam walls (--style foam) are the wall
  visualization now; the flags and the v21-solid-edges golden were removed
  (the meta format keeps the keys so the older goldens' config hashes stay
  stable).
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

V2.2 additions (both feed the shared cellTint pipeline — solid impostors,
ray tracer AND the coloured X-ray):
- **Food/nutrient heatmap** (`--tint food`, `--heat-scale F`): one scalar per
  cell — `food + nutrient` in light mode, `food` otherwise — normalized by
  `heatScale` (0 = auto: `splitThreshold + lightNutrient/lightDecay` in light
  mode, `splitThreshold` otherwise) over a cool-to-warm ramp (deep blue →
  green → yellow → red). Shows WHERE the growth substance sits: on a
  light-grown form the lit canopy is hot, the shaded underside cold.
- **Mitosis pulse** (`--tint pulse`, `--pulse-decay F`): every division sets
  a per-cell pulse to 1.0 (parent + daughter, new `pulse` buffer, persisted
  in CGST v3 so pulse renders replay byte-exact); `pulseDecayStep` multiplies
  it by `--pulse-decay` (default 0.95 → half-life ~13.5 steps, visible ~45)
  each step; cellTint blends a magenta flash over the base tint. This
  visualizes the paper §7 observation ("rhythmic pulses and waves of growth
  propagating across surfaces"): in light mode the flashes cluster in the
  lit band and advance with the growth front (pulse-field ring R² 0.33–0.36
  and rising at t=2000), while uniform growth decorrelates to the
  uncorrelated baseline (~0.18–0.22 — scattered flicker). Measured in
  benchlog V2.2.
- **Render-time hygiene fix**: the directional render's illumination refresh
  no longer leaks nutrient production into the simulation state (the food
  heatmap reads nutrient — the v22-food-heat golden replay caught the
  resulting tint drift; rendering is now strictly side-effect-free).
- **Validation**: `swift build` (debug+release), `--selftest` (4 phases)
  PASS, `--lockstep` PASS, all 9 previous goldens PASS, 3 new goldens
  (food heatmap on a light-grown state, mitosis pulse on the same state —
  byte-exact snapshot replay with the persisted pulse buffer — and coloured
  X-ray of the heatmap state) capture + check PASS; CGST v3 round trip
  byte-identical with the new pulse block; tint-mode bench ~nil cost
  (benchlog).

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
`--screenshot-style solid|xray|foam`. V2.1: `--shadows F` (per-pixel shadow
strength, directional solid only, default 1), `--xray-color` (per-channel
tinted X-ray), `--light-soft F` (light angular radius in radians, default
0.35, 0 = legacy binary illumination). The V2.1 `--edges`/`--edge-width`
flags were RETIRED in V2.1b (the painted edge layer is gone — the foam walls
replaced it). V2.2: `--tint none|chemical|food|pulse` (cellTint colour mode;
default auto = chemical colours in RDE mode, neutral otherwise — the
pre-V2.2 behaviour; `food` = food/nutrient heatmap, `pulse` = magenta
division flashes; an explicit `--tint` overrides the mode a loaded state
carries), `--heat-scale F` (heatmap normalization, 0 = auto: splitThreshold +
lightNutrient/lightDecay in light mode, splitThreshold otherwise),
`--pulse-decay F` (per-step pulse multiplier, default 0.95). V2.1b:
`--style solid|xray|foam|foam-xray` (foam = the power-diagram chamber
rendering, `--radius-scale F` sets the render radius scale — foam recommends
0.7; the 0.5 default makes walls degenerate to points). Foam look knobs via
env: `CG_FOAM_OPACITY` (per-face composite opacity, default 0.5),
`CG_FOAM_LAYERS` (max composited chamber faces per ray, default 6),
`CG_FOAM_WALL` (foam-xray wall-spike gain, default 0.25). Diagnostic:
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
plant / coral / degenerate-demo), Solid/X-ray/Foam render-style switch
(+ X-ray gain, foam opacity + the foam radius-scale hint), Save State…/Load
State…, growth mode + light/RDE parameter groups, repulsion, forces, run
controls. Jitter and subdivision apply on Reset.

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
| V2.2 ✅ | tint modes: food/nutrient heatmap (`--tint food`, `--heat-scale`), mitosis pulse visualization (`--tint pulse`, `--pulse-decay`) — division flashes trace the paper §7 growth wavefronts; CGST v3 persists the pulse buffer (byte-exact pulse replays); render-time illumination refresh made side-effect-free | `--selftest` 4 phases, `--lockstep`, 9/9 legacy goldens PASS, 3 new goldens capture+check (pulse + heatmap byte-exact snapshot replays), state round-trip byte-identical, tint-mode bench ~nil (benchlog), pulse ring-R² wavefront metric: light 0.33–0.36 vs uniform 0.18–0.22 |
| V2.1b ✅ | foam rendering: cells as power-diagram chambers (flat power-bisector walls + curved caps), translucent front-to-back composite ("walls behind walls") in the ray tracer (`--style foam`) and the interactive impostors (sorted premultiplied blend); foam X-ray with wall-crossing membranes (`--style foam-xray`); painted edge layer RETIRED (`--edges`/`--edge-width` removed, v21-solid-edges golden retired) | `--selftest` 4 phases, `--lockstep`, 11/11 surviving goldens byte-PASS, 3 new foam goldens capture+check, cross-renderer consistency (impostor-foam vs ray-foam same state), 0.5-vs-0.7 radius-scale mechanism demo (walls degenerate at 0.5), foam cost bench (ray foam 3.6–5.3× solid, impostor ≈ solid + one-off sort) |

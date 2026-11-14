# CellularGrowth

Reimplementation of Andy Lomas' **"Cellular Forms: an Artistic Exploration of
Morphogenesis"** (AISB50) on Metal / Apple Silicon.

The exact paper extraction (formulas verified against hi-res page renders,
including the `bulgeDist` typo analysis) lives in
[`docs/reference/lomas_cellular_forms_extraction.md`](docs/reference/lomas_cellular_forms_extraction.md),
with page screenshots in `docs/reference/pages/`.

## Status — V2.7 complete: interactive video export + death sliders

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

V2.3 additions — **cell types + two-phase growth** (the user's head idea; the
paper has no cell types, every value is a documented design choice):
- **Type contract**: a `cellType` uint per cell. **Type 0 = none — ALL
  existing behaviour** (the rule matrix is a no-op on all-type-0 runs: the
  14 goldens and selftest phases A–D are byte-identical). Types: 1 stem,
  2 lamel, 3 dal, 4 cap. Types persist in CGST v3 as an appended block
  (N×uint32 + the division counter, before the raw SimParams — the
  pulse-block pattern; legacy files load type 0 everywhere).
- **Phase-0 polar typing** (`--type-poles "x,y,z"`, default vertical
  `(0,1,0)`; `--type-t1 F` / `--type-t2 F`, defaults −0.4 / 0.9): at reset
  (and via the GUI re-type button) the normalized axis coordinate
  `s = dot(normalize(P − centroid), axis)` types the ball: `s < t1` → stem,
  `s > t2` → cap, between → the lamel/dal band, alternating along ONE cyclic
  walk over the band cells ordered by azimuth around the axis (an odd band
  size leaves exactly one same-type seam at the walk closure — harmless).
- **Two-phase division rule matrix** (`--phase-x N`; 0 = never): phase 1
  (divisions < N): the stem divides ONLY at the band edge (a voeg
  neighbour), the cap uniformly, the band dormant. Phase 2 (divisions ≥ N):
  ONLY the band divides — lamel at `--lamel-rate` (default 1.0) at edge
  links (stem/cap neighbours), dal at `--dal-rate` (default 0.5),
  lamel↔dal interior links at lamelRate/`--interior-divider` (10),
  lamel↔lamel never. Daughters inherit the parent's type. The phase clock
  is a device-side division counter (exactly one increment per executed
  split; the switch happens at the first step boundary after the X-th
  division — exactly once). Two non-obvious correctness points documented in
  Shaders.swift: (a) the decide pass's independent-set blocker test must
  count only MATRIX-ELIGIBLE cells — a permanently ineligible food hoarder
  (stem cells away from the band; stem/cap in phase 2) would otherwise block
  the growth front forever (measured: the front froze after 13 divisions);
  (b) dormant band cells are valence sinks — they never divide, so they
  never relieve the +1 links front divisions insert into their rings; the
  stem front saturates after a few hundred divisions (watched, documented).
- **Type colours** (`--tint type`, cellTint mode 5): stem white, lamel warm
  white, dal cool white, cap red — deliberately near-white for the three
  non-cap types so foam walls (which average adjacent tints) blend clean
  inside the band and only the cap boundary shows a strong blend (tested
  50/50 average vs a dominant-type variant; the average ships — the
  red↔white boundary reads as a natural gradient, not mud). `--tint
  type-pulse` (mode 6) adds the mitosis-pulse overlay: in phase 2 the
  flashes light up the BAND and nowhere else — the direct visual proof of
  the rule matrix (per-type pulse means: lamel/dal > 0, stem/cap 0.000).
- **Preset `paddenstoel`** (mushroom): tuned two-phase showcase — small cap
  patch (t2 0.9) so phase-1 exponential cap growth makes the dome while the
  stem edge-grows the stalk, then phase 2 (X 2700) ruffles the gill collar
  at moderate rates (lamel 0.5 / dal 0.25). Foam style + `--tint type` +
  `--radius-scale 0.7` recommended. Honest morphology note: the form reads
  as a mushroom-family silhouette (red cap canopy with overhanging margin,
  two-tone lamellen collar at the junction, white stem base) — the stem
  stays squat/flared rather than a tall cylinder, and with other
  thresholds the same rules give a ruffled valance (documented in the V2.3
  report).
- **Validation**: `swift build` (debug+release), `--selftest` (5 phases —
  the new phase E checks the typed run: invariants + Σ per-type counts = V
  at every checkpoint, phase-1 freezes lamel+dal, phase-2 freezes stem+cap,
  the phase transition observed exactly once) PASS, `--lockstep` PASS, all
  14 goldens byte-PASS, typed state round-trip (types + division counter
  persisted), decide/foodStep type-check bench ~nil at matched counts
  (benchlog V2.3).

V2.4 additions — **the differentiation phase** (RDE-driven cell-type
transitions with hysteresis; paper §8 names differentiation as the missing
piece — "cell differentiation could be explored ... re-introduce
reaction-diffusion equations to control the blending" — the mechanism below
is a documented design where the paper is silent):
- **The phase** (`--differentiate-at N [--differentiate-until M]`; GUI
  button "Differentiate now" / "Resume growth"): from step N the WHOLE
  division block is skipped — the division counter freezes AND food stops
  accumulating (growth resumes without a synchronized burst: the pre-phase
  food distribution is preserved) — while the RDE keeps stepping. `M` = 0
  (or omitted) keeps the phase until the host/GUI ends it. The flag is a
  SimParams field, so a CGST v3 state saved mid-phase resumes IN the phase
  (verified: mid-diff state round-trip prints DIFFERENTIATING with the type
  map intact).
- **typeTransition** (new kernel, encoded in the division block's slot —
  AFTER it, same family as cellTint; the transition's writes land for the
  NEXT step's decide/foodStep, race-free — every thread reads only chemV[gid]
  and writes only cellType[gid]): every UNDIFFERENTIATED cell (type 0)
  transitions `v >= --type-cap-v` (default 0.30) → cap (4), `v <=
  --type-stem-v` (default 0.10) → stem (1), else stays 0.
- **Hysteresis = type memory**: a TYPED cell ignores the RDE entirely from
  the step it transitions on (the kernel early-exits; rdeStep freezes its
  u/v — "no reaction participation"). A cell's type changes AT MOST ONCE,
  so patterns cannot flicker at a threshold boundary. (A full up/down
  hysteresis band — typed cells re-reading with offset thresholds — is a
  small extension; type memory is the minimal stable form.) Frozen values
  are still READ by untyped neighbours' stencil, so v-high regions pin the
  pattern and carry the regional growth drive.
- **Rule-matrix MODE** (`--type-mode band|region`): the V2.3 matrix assumes
  a polar band; an RDE type map has none. Region mode: every type divides
  uniformly within its own region (the patterned rate is carried by the
  frozen-v coupling in the food drive — cap regions grew ×8–13 per capita
  vs stem ×1.5–6 in validation runs), and the two-phase machinery is not
  applied. Auto: a requested differentiation phase → region, polar typing →
  band; explicit flag wins; type-0 cells behave identically in both modes.
- **Validation**: `--selftest` 6 phases (new phase F: static RDE settle →
  differentiation phase with division ENABLED but frozen — cells/divisions
  constant — → type partition 0/V → mixed, flip stream settles ≈ 0, only
  0→1/0→4 transitions ever, then regional growth resumes: cap per-capita
  growth ≫ stem) PASS, `--diff-probe` PASS (pattern R², mixed types,
  flip-settle, type map PNG), `--lockstep` PASS, 14/14 goldens byte-PASS,
  differentiation-pass bench ~2–3% at 655k cells (benchlog V2.4).

V2.5 additions — **starvation death** (cell pruning; paper §8 names it as
the inverse of division — "cell death as well as cell growth, with
appropriate topological changes to the surface when cells die"; the death
RULE below is a documented design where the paper is silent):
- **The rule** (`--death`, `--death-threshold F`, `--death-steps N`): a cell
  whose accumulated FOOD ledger stays below `deathThreshold` for
  `deathSteps` CONSECUTIVE steps dies. `0` = never. The signal is the food
  LEDGER (not food + nutrient: the nutrient steady state dwarfs the food
  scale and would read "fed" everywhere). The window is a GROWTH-RATE
  detector: a fed cell re-crosses the threshold long before the window
  closes, a shaded cell (drive ≈ `--light-base`) cannot — so the pruning
  front sits at the shade iso-line where the climb time crosses the window.
  `deathThreshold` defaults to `splitThreshold`: a cell starves while it
  cannot reach its own division threshold fast enough.
- **The splice is Euler-exact** (dV−1, dE−3, dF−2): removing a valence-k
  vertex leaves a k-gon hole bounded by the dying cell's ring cycle; it is
  closed WITHOUT new vertices by balanced EAR-CLIPPING (k−3 shortest-local
  diagonals, picked under valence caps + chord checks — a fan from one pivot
  was tried first and deadlocked: sustained pruning piled valence onto
  pivot cells up to `maxValence`, where they hoard food and block every
  higher-indexed candidate forever). Claims form the SAME ring-distance-2
  independent set as division (min index wins) — splices touch disjoint
  neighbourhoods, so claim order is irrelevant. Guards: every ring
  neighbour must have valence ≥ 4; a valence-3 death is the native
  zero-clip case; a chord-blocked hole defers the death (the cell keeps
  starving and retries — all-or-defer, no partial state).
- **Dead slots stay allocated** (`liveCount = cellCount − deathCount`; the
  counter is an allocation high-water mark): a death sets `deadFlags`,
  zeroes radius/links/starving, and every consumer skips dead slots
  (grid, repulsion, march, typing, render — radius 0 = zero-area impostor).
  Buffer fragmentation is the documented trade-off; compaction is a
  follow-up.
- **HOT/SPLICE SPLIT (V2.5 perf finding)**: the splice kernel's statically
  present body alone cost ~+7 ms/step at 655k cells even when not one
  thread executed it (bench bimodality investigation: body deleted →
  baseline speed; every array shrunk → still slow) — the heavy register
  footprint collapses co-residency and the full-width dispatch becomes
  launch-latency bound (see benchlog V2.5). deathExecute is now guards +
  claim QUEUEING only; the ear-clip machinery lives in `deathSplice` at a
  fixed narrow width striding over the queue. Queue order is
  nondeterministic but provably irrelevant (disjoint neighbourhoods) —
  verified: two identical seeded runs produce BYTE-IDENTICAL states.
- **State**: CGST v3 persists the death state (deadFlags + starving +
  deathCounter) — a loaded death-era state resumes with its dead slots
  (round-trip byte-identical).
- **Validation**: `--selftest` 7 phases (new phase G: light-grown run with
  death — V rises then FALLS, 168k deaths, all invariants hold on the LIVE
  surface with V = live cells) PASS, `--lockstep` PASS, 14/14 goldens
  PASS, CGST v3 death round-trip byte-identical, seeded two-run state
  compares byte-identical (death path deterministic), prune-demo
  before/after screenshots inspected (coral-crown pruning from below),
  death-block bench +0.5 ms/step ≈ +3.7% at 655k cells (benchlog V2.5).

V2.7 additions — **interactive video export + death sliders** (GUI-only
milestone: no shader, no SimParams field, no buffer, no step-routine change;
all 14 goldens byte-PASS unchanged):
- **Non-blocking export on a cloned sim**: "Export video…" no longer pauses
  the display link or owns the live sim. The GUI freezes the sim for a
  one-off `CellSim.cloneState(from:)` — a byte-exact copy of ALL live state
  (positions double buffer, links, food, RNG, nutrient/chemicals both sides,
  tint, pulse, cell types + division counter, death state, counters, swap
  flags AND `stepIndex`, the light-march scheduler clock) onto a PRIVATE
  sim + PRIVATE MTLCommandQueue — then resumes: the sim keeps
  stepping/rendering while the export renders offscreen on a background
  thread. GPU contention may lower the GUI fps (accepted). Why not the CGST
  v3 snapshot machinery: v3 does not persist `stepIndex`, so a v3
  round-trip would phase-shift the lightInterval march relative to the live
  sim; direct buffer copies are exact (validated: the clone-at-N state is
  byte-identical, and after +400 steps every integer/frozen state block is
  byte-identical to the control's continuation — the float deviations equal
  the sim's own run-to-run scheduling chaos, see benchlog V2.7). Play /
  Reset / state-load are disabled for the sub-second clone window (a
  mid-clone step would race the clone's buffer reads).
- **Live preview + progress**: the most recent COMPLETED export frame is
  decoded + downscaled OFF-main (pure CoreGraphics) and shown in the panel,
  with a "frame N/M — elapsed m:ss" line (frame cadence + a 0.5 s timer).
  "Show in Finder" reveals the movie (or the frames directory when the
  encode failed).
- **Stoppable**: a Stop button sets a flag polled BETWEEN frames; whatever
  rendered is still assembled into the movie — encode-after-the-loop keeps
  ONE writer path (a stop just truncates the frame list). The status marks
  it a partial export ("stopped at N/M frames"); a stop before the first
  frame completes yields no video (nothing rendered, no movie).
- **Export-sim capacity**: the clone's `maxCells` is ESTIMATED from the
  expected growth (per-capita division rate ≈ drive / (2 × splitThreshold),
  compounded over frames × steps/frame, ×2 margin, clamped to the live
  sim's maxCells) — a second full 40M buffer set (~8.4 GB) is only paid when
  the growth estimate demands it. Overshoot stalls division (the
  device-side hard cap) and is reported in the status — never a crash.
- **The .mp4 extension is forced** (`assembleMovie` normalizes the output
  URL and returns the actual written path): AVAssetWriter writes EXACTLY
  the URL it is given — the V2.6 ".m4v" report has no write path in this
  code (probe: a .m4v URL with fileType .mp4 is accepted silently and
  written verbatim, ISO-media/MP4 inside; most plausibly the user saw
  QuickTime's MPEG-4 video document type or saved through QuickTime, which
  defaults to .m4v). The on-disk name now always matches the documented
  container.
- **Death sliders (V2.5 follow-up)**: a collapsible Death group directly
  under Growth — starvation-death toggle + threshold + window sliders, live
  SimParams like every knob. Preset/state loads sync the sliders (the
  panel-shows-what-runs contract; `prune-demo` carries death on); explicit
  user input wins after a load. Frames slider now goes to 10000
  (stoppability + the preview make long exports workable — no other limits
  added); steps/frame cap raised 600 → 2000 (sized for the blocking-export
  era before V2.7). The sections this milestone touches (export, death) are
  collapsible `DisclosureGroup`s.
- **Window resize**: export start GROWS the window (never shrinks) to the
  export aspect ratio (4:3 + the 290 pt panel column) at ~80% of the screen
  height.
- **Validation**: `swift build` (debug+release), `--selftest` 7 phases PASS,
  `--lockstep` PASS, 14/14 goldens byte-PASS, CLI `--anim` regression (4
  frames → valid H.264 .mp4, ffprobe-verified), clone probes (at-N state
  byte-identical; +400-step continuation integer-state byte-identical vs
  control AND a fresh-run baseline; dividing-run form tracking within 1.4%
  + invariants PASS), abort + capacity-watchdog probes PASS, export-cost
  timings (benchlog V2.7). The export button itself was NOT click-tested
  (no screen-recording permission in the agent context — the same open
  point as V2.6).

V2.6 additions — **camera choreography + video export** (host-side viewer
state only — no sim buffer, shader or SimParams change; all 14 goldens
byte-PASS unchanged):
- **Follow centroid** (`--follow-centroid` headless, GUI default ON with a
  panel toggle + smoothing slider): the orbit camera's `target` tracks the
  LIVE cells' geometric centroid (uniform cell mass — the paper assigns no
  per-cell masses; dead slots skipped per V2.5), smoothed per rendered frame
  with exponential smoothing `t += (centroid − t)·α` (`--follow-smooth F`,
  default 0.15; snap on start/reset — a follow that starts centered).
  Purely host-side: two O(live) pointer passes over the shared positions
  buffer, no allocation, no GPU pass (measured 0.11 ms/frame at ~180k cells;
  ~0.4 ms extrapolated at 655k). Drag-orbit keeps rotating yaw/pitch around
  the CO-MOVING target by construction (eye = target + distance·dir).
  The camera distance follows only in the export (framed on the centroid,
  max live distance × 2.6); in the GUI the user keeps zoom control.
- **Circle mode** (`--orbit-rate F` headless, GUI toggle + speed slider):
  yaw advances F radians EVERY rendered frame around the (possibly
  following) target. Per-FRAME, not per-second: export frames are equally
  spaced in STEPS (the M6 convention), so a per-frame rate keeps exports
  deterministic; at the GUI's ~60 fps the 0.005 default ≈ one turn per 21 s.
- **Documented asymmetry**: follow/orbit default OFF headless, default ON in
  the GUI. The goldens and single-frame renders (`--screenshot`, `--render`)
  keep the legacy fixed-target framing unconditionally (the choreography
  applies only to the video paths: `--anim` and the GUI export); the GUI is
  not golden-covered.
- **Video export = MP4**: the M6 H.264 `.mov` container is REPLACED by
  `.mp4` (`AVAssetWriter` fileType `.mp4`; codec H.264, 30 fps, BGRA→yuv420p
  unchanged) — universal playback, one code path. The M6 animation routine
  was FACTORED into `HeadlessRunner.exportAnimation(config:)`: the CLI
  `--anim` and the GUI "Export video…" button share ONE deterministic
  offscreen implementation (never duplicated). The GUI button grows on from
  the LIVE state with the panel's style/tint knobs (style maps 1:1;
  tint needs no mapping — both renderers read the same cellTint buffer; the
  GUI X-ray gain stays an impostor-only knob — the offline exposure keeps
  its tuned default) and the current follow/orbit settings; the display link
  is paused for the duration (the export owns the sim + queue). Export
  resolution is the CLI default 1600×1200 (deterministic, window-independent).
- **Validation**: `swift build` (debug+release), `--selftest` 7 phases PASS,
  `--lockstep` PASS, 14/14 goldens byte-PASS, follow+orbit `--anim` exports
  visually inspected (form stays centered while growing 642→9 161 cells and
  the azimuth visibly rotates; light-grown form drifting toward the light is
  centered with follow ON vs. off-frame-top with follow OFF), sim-step bench
  unchanged by construction (no step-routine/shader/buffer edit; 655k
  forces-only 12.77 vs 12.83 ms/step V2.5 baseline — same floor), export
  costs measured (benchlog V2.6).

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
`CG_FOAM_WALL` (foam-xray wall-spike gain, default 0.25). V2.3:
`--type-poles "x,y,z"` (polar typing axis, default vertical `0,1,0` — the
presence of any of the three typing flags enables typing),
`--type-t1 F` / `--type-t2 F` (stem/cap thresholds on the normalized axis
coordinate; between them the lamel/dal band; defaults −0.4 / 0.9),
`--phase-x N` (phase-2 switch after N executed divisions, 0 = never),
`--tint type|type-pulse` (type colours: stem white, lamel warm white, dal
cool white, cap red; type-pulse adds the mitosis-pulse overlay),
`--lamel-rate F` / `--dal-rate F` / `--interior-divider F` (phase-2 band
growth rates, defaults 1.0 / 0.5 / 10). V2.5: `--death` (starvation death
on), `--death-threshold F` (default = splitThreshold), `--death-steps N`
(consecutive starving steps before death, default 250; 0 = never),
`--tint starve` (the red death front — blend weight = starving counter
progress toward the window). V2.6: `--follow-centroid` (anim camera follows
the live-cell centroid; GUI default ON, headless default OFF),
`--follow-smooth F` (per-frame exponential smoothing, default 0.15, 0 =
snap), `--orbit-rate F` (circle mode: yaw +F rad per rendered frame; both
apply ONLY to `--anim`/GUI exports — `--screenshot`/`--render` keep the
legacy framing), and the `--anim` movie container is now `.mp4` (H.264;
the M6 `.mov` was replaced). Diagnostic:
`--div-stop N` (disable division after step N); env `CG_NO_COARSE_SKIP=1`
disables the light-march occupancy skip (A/B kill switch); env
`CG_SHADOW_SAMPLES=N` overrides the 4 default shadow rays per pixel (bench
knob — goldens record 4). Bench diagnostics (V2.5): env `CG_BENCH_SPLIT=1`
prints per-level encode/wait/gpu-window split (exposes GPU-scheduler
interference), `CG_BENCH_BATCHES=1` prints every batch's gpu window +
median/min (a level whose batches disagree by >2x is contention-tainted;
the min batch is the best exclusive-GPU estimate).

```bash
# M4: ray-traced renders of the grown state (paper Fig. 8 / Fig. 9 styles)
swift run -c release CellularGrowth --render form.png --style solid \
    --width 1600 --height 1200 --steps 2400 --subdiv 3 --max-cells 100000
swift run -c release CellularGrowth --render xray.png --style xray \
    --width 8600 --height 8600 --state-load state.bin
```

## Controls

Drag to orbit, scroll/pinch to zoom. Panel: preset dropdown (ball / brain /
plant / coral / degenerate-demo / paddenstoel / prune-demo), Solid/X-ray/Foam
render-style switch (+ X-ray gain, foam opacity + the foam radius-scale
hint), tint picker
(Default/None/Chemical/Food heat/Mitosis pulse/Type/Type + pulse/Starving),
the V2.6 Camera group (Follow centroid toggle — ON by default, smoothing
slider, Circle toggle + orbit-speed slider) and the collapsible V2.7 Video
export group (frames slider to 10000 + steps/frame slider, "Export video…"
runs NON-BLOCKING on a byte-exact clone of the live state with its own
command queue — the sim keeps running/rendering; the most recent completed
frame previews in the panel with a frames + elapsed progress line; Stop
finishes the PARTIAL video; "Show in Finder" reveals the movie; grow-only
window resize to the export aspect at start; same deterministic offscreen
path as `--anim`, panel style/tint + current camera choreography), Save
State…/Load State…, growth mode + light/RDE parameter groups, the V2.7
collapsible Death group directly under Growth (starvation-death toggle +
threshold + window sliders — live SimParams; preset/state loads sync them,
explicit user input wins afterwards), the V2.3
Type layout group (polar typing toggle + axis/threshold sliders — dragging a
knob re-types the LIVE cells, Re-type button, phase switch X, lamel/dal
rates), repulsion, forces, run controls. Jitter and subdivision apply on
Reset.

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
| V2.3 ✅ | cell types + two-phase growth: `cellType` buffer (0 none = all existing behaviour, 1 stem, 2 lamel, 3 dal, 4 cap), phase-0 polar typing (`--type-poles/--type-t1/--type-t2`, one cyclic azimuthal walk for the lamel/dal band), type-dependent division rule matrix with a device-side phase switch (`--phase-x`): phase 1 stem edge-only + cap uniform + band dormant, phase 2 band-only (lamel/dal/interior rates); daughters inherit type; CGST v3 persists types + division counter; type colours (`--tint type|type-pulse`); preset `paddenstoel` (mushroom) | `--selftest` 5 phases (new phase E: typed run, Σ per-type = V, phase-1 freezes voeg / phase-2 freezes stem+cap, transition exactly once), `--lockstep`, 14/14 goldens byte-PASS, typed state round-trip (types + phase clock persisted), phase-2 pulse-in-the-band validation (per-type pulse means: lamel/dal > 0, stem/cap 0.000), decide/foodStep type-check bench ~nil at matched counts (benchlog V2.3), blocker-predicate fix documented (eligible-set independent set) |
| V2.4 ✅ | the differentiation phase: division frozen (`--differentiate-at N [--differentiate-until M]`, GUI "Differentiate now"), RDE keeps stepping, new `typeTransition` kernel (division-block slot, race-free) types undifferentiated cells from the chemical field (`--type-cap-v` 0.30 → cap, `--type-stem-v` 0.10 → stem, dead band stays 0); hysteresis = type memory (typed cells ignore the RDE: rdeStep freezes their u/v — no reaction participation, patterns cannot flicker, a cell's type changes at most once); rule-matrix MODE `--type-mode band\|region` (region: every type divides uniformly in its region — patterned growth via the frozen-v food coupling; band: the V2.3 matrix; type-0 contract both modes); differentiationPhase persists in CGST v3 raw params (mid-phase states resume in the phase); GUI button + threshold sliders | `--selftest` 6 phases (new phase F: static RDE settle → differentiation with division enabled-but-frozen (cells/divisions constant) → 0/V → mixed partition, flip stream settles ≈ 0 (gate ≤ 0.5% V/window), type memory (only 0→1/0→4 ever), regional growth resumes: cap per-capita ×94.9 vs stem ×6.14) PASS, `--diff-probe` PASS (pattern R² 0.68, stem 32.5%/cap 27.6%, flips 1439-burst → 0–4/window tail, type map PNG), `--lockstep` PASS, 14/14 goldens byte-PASS (typing off by default), mid-diff state round-trip resumes DIFFERENTIATING, differentiation-pass bench ~2–3% at 655k cells / regional-vs-uniform growth bench count-confounded as designed (benchlog V2.4) |
| V2.5 ✅ | starvation death — the inverse of division (paper §8): a cell below `--death-threshold` for `--death-steps` consecutive steps dies (food-LEDGER signal; the window is a growth-rate detector — the pruning front sits at the shade iso-line); Euler-exact ring splice (dV−1, dE−3, dF−2) via balanced ear-clipping under valence caps + chord checks (the fan design deadlocked on valence-hoarding pivots — documented); claims = the division independent-set rule (order-irrelevant, verified byte-identical); dead slots stay allocated (`liveCount = cellCount − deathCount`), every consumer skips them; CGST v3 persists the death state; `--tint starve` death front; preset `prune-demo` (coral-crown self-pruning); HOT/SPLICE split (guards+queue vs narrow-width splice) after the bench bimodality investigation | `--selftest` 7 phases (new phase G: grown run with death — V rises then falls, 168k deaths, invariants on the LIVE surface) PASS, `--lockstep` PASS, 14/14 goldens PASS, CGST v3 death round-trip byte-identical, seeded two-run state compares byte-identical, prune-demo before/after screenshots inspected, death-block bench +0.5 ms/step ≈ +3.7% at 655k cells, headless jitter seeding bug fixed (benchlog V2.5) |
| V2.6 ✅ | camera choreography + video export: follow-centroid (orbit-camera target tracks the LIVE cells' smoothed geometric centroid — uniform cell mass, dead slots skipped; host-side two-pass read, no GPU pass, no buffers touched; GUI default ON with toggle + smoothing slider, headless `--follow-centroid`/`--follow-smooth` opt-in), circle mode (`--orbit-rate F` rad per rendered frame around the moving target; GUI toggle + speed slider), drag-orbit rotates around the moving target by construction; video export as MP4 (M6 `.mov` REPLACED by `.mp4`, H.264/30 fps unchanged; M6 anim routine factored into `HeadlessRunner.exportAnimation` shared by CLI `--anim` and the GUI "Export video…" button — live state, panel style/tint, current choreography, display link paused during export) | `swift build` (debug+release), `--selftest` 7 phases PASS, `--lockstep` PASS, 14/14 goldens byte-PASS (headless camera legacy by default; choreography only on the video paths), follow+orbit anim exports visually inspected (form centered while growing 642→9 161 cells + azimuth rotates; light-grown drift centered with follow ON vs off-frame legacy), sim-step bench unchanged (12.77 vs 12.83 ms/step at 655k — no step-routine edit), export costs: ~0.37 s/frame GPU at 1600×1200 (~186k cells), ~38 ms/frame encode, choreography 0.11 ms/frame (benchlog V2.6) |
| V2.7 ✅ | interactive video export + death sliders (GUI-only: no shader/SimParams/buffer/step-routine edit): NON-BLOCKING export on a byte-exact cloned sim (`CellSim.cloneState` copies ALL live state incl. both double-buffer sides, chemicals, RNG, pulse, types + division counter, death state, swap flags and the stepIndex march clock — CGST v3 round-trip rejected: it drops stepIndex; own `CellSim` + own command queue, GUI keeps stepping/rendering, GPU contention accepted), live frame preview + frames/elapsed progress + "Show in Finder" in the panel, Stop = partial-but-complete video (flag polled between frames, encode-after-loop truncates the frame list), export-sim capacity ESTIMATED from expected growth (overshoot stalls division device-side and is reported, never crashes), .mp4 extension forced (AVAssetWriter writes exactly the given URL — the V2.6 ".m4v" report has no write path in this code), death sliders (V2.5 follow-up: toggle + threshold + window, preset/state-synced, explicit input wins), frames slider to 10000 + steps/frame 600→2000, collapsible export/death sections, grow-only window resize to the export aspect | `swift build` (debug+release), `--selftest` 7 phases PASS, `--lockstep` PASS, 14/14 goldens byte-PASS, CLI `--anim` regression (4 frames → valid H.264 .mp4, ffprobe: h264/30 fps), clone probes: at-N state byte-identical, +400-step continuation integer-state byte-identical vs control AND fresh-run baseline (float Δ = the sim's own scheduling chaos, equal magnitude in both pairs), dividing-run form tracking ≤1.4% + invariants PASS, abort probe (4/10 frames → partial movie), capacity-watchdog probe (stall at maxCells + warning), GUI launch check; export button NOT click-tested (agent context — open point, benchlog V2.7) |

//  ContentView.swift
//  Main window layout: Metal view + parameter panel.

import SwiftUI

struct ContentView: View {
    @ObservedObject var vm: SimViewModel
    // V2.7: the sections this milestone touches are collapsible
    // (DisclosureGroup) — the user's request. Only these two; the untouched
    // groups keep their GroupBox styling.
    @State private var exportOpen = true
    @State private var deathOpen = true
    // V2.8: same collapsible pattern for the drive groups (growth drives,
    // light, RDE). All start expanded — the pre-V2.8 panel showed everything.
    @State private var growthOpen = true
    @State private var lightOpen = true
    @State private var rdeOpen = true

    var body: some View {
        HStack(spacing: 0) {
            MetalViewContainer(vm: vm)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            ScrollView {
                controlPanel
                    .frame(width: 290)
                    .padding(12)
            }
            .background(Color(nsColor: .windowBackgroundColor))
        }
    }

    private var controlPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Cellular Growth — V2.8")
                .font(.headline)

            HStack(spacing: 8) {
                Button(vm.playing ? "Pause" : "Play") { vm.playing.toggle() }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.exportSetupActive)   // V2.7: mid-clone a step would race the clone's buffer reads
                Button("Reset") { vm.reset() }
                    .disabled(vm.exportSetupActive)
            }

            GroupBox("Preset") {
                Picker("Preset", selection: $vm.presetName) {
                    Text("—").tag(String?.none)
                    ForEach(SimPreset.all, id: \.name) { p in
                        Text(p.name).tag(String?.some(p.name))
                    }
                }
                if let name = vm.presetName, let p = SimPreset.named(name) {
                    Text(p.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            GroupBox("Render style") {
                Picker("Style", selection: $vm.renderStyle) {
                    Text("Solid").tag(Renderer.RenderStyle.solid)
                    Text("X-ray").tag(Renderer.RenderStyle.xray)
                    Text("Foam").tag(Renderer.RenderStyle.foam)
                }
                .pickerStyle(.segmented)
                if vm.renderStyle == .xray {
                    slider("X-ray gain", $vm.xrayGain, 0.05...2)
                }
                if vm.renderStyle == .foam {
                    slider("Foam opacity", $vm.foamOpacity, 0.2...0.8)
                    // V2.1b: the foam chambers only develop flat walls where
                    // neighbouring spheres OVERLAP — radiusScale 0.5 (the
                    // default) makes them degenerate to points.
                    Text("Foam recommends Radius scale 0.7 (below ~0.55 the walls degenerate to points).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                // V2.2 tint modes (feeds solid AND X-ray — the tint buffer is
                // shared). "Default" = chemical colours in RDE mode, neutral
                // otherwise (the pre-V2.2 behaviour).
                Picker("Tint", selection: $vm.tintMode) {
                    ForEach(SimViewModel.TintMode.allCases) { m in
                        Text(m.label).tag(m)
                    }
                }
                if vm.tintMode == .food {
                    slider("Heat scale (0 = auto)", $vm.heatScale, 0...8)
                }
                if vm.tintMode == .pulse {
                    slider("Pulse decay", $vm.pulseDecay, 0.8...0.999)
                }
            }

            // V2.6 camera choreography: the target tracks the LIVE cells'
            // centroid (exponentially smoothed, host-side); circle mode
            // advances the yaw at a fixed per-frame rate. Drag-orbit keeps
            // rotating yaw/pitch around the moving target by construction.
            GroupBox("Camera (V2.6)") {
                Toggle(isOn: $vm.followCentroid) { Text("Follow centroid") }
                slider("Follow smoothing", $vm.followSmoothing, 0...0.9)
                Toggle(isOn: $vm.orbitEnabled) { Text("Circle (orbit)") }
                slider("Orbit speed (rad/frame)", $vm.orbitRate, 0...0.05, step: 0.001)
                Text("Follow targets the live-cell centroid; drag orbits around the moving target.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            GroupBox("State") {
                HStack(spacing: 8) {
                    Button("Save State…") { vm.saveStatePanel() }
                    Button("Load State…") { vm.loadStatePanel() }
                }
                .disabled(vm.exportSetupActive)   // V2.7: a mid-clone state INSTALL would race the clone's reads
            }

            // V2.7 video export: NON-BLOCKING — the export runs on a CLONE
            // of the live state (own CellSim + own command queue) while the
            // GUI keeps stepping/rendering; the most recent completed frame
            // previews right here, progress shows frames + elapsed time,
            // and Stop finishes a PARTIAL video of what rendered so far.
            collapsible("Video export (V2.7)", isExpanded: $exportOpen) {
                VStack(alignment: .leading, spacing: 10) {
                    slider("Frames", $vm.exportFrames, 2...10000, step: 1, format: "%.0f")
                    slider("Steps / frame", $vm.exportStepsPerFrame, 0...2000, step: 10, format: "%.0f")
                    if vm.exportRunning {
                        Button("Stop export") { vm.stopExport() }
                            .buttonStyle(.borderedProminent)
                    } else {
                        Button("Export video…") { vm.exportVideoPanel() }
                            .buttonStyle(.borderedProminent)
                            .disabled(vm.exportSetupActive)   // V2.7: sub-second clone window
                    }
                    // V2.7 live preview: the most recent COMPLETED export
                    // frame (the offscreen render), scaled to fit — no
                    // file-manager hunting. Shows the export's frames, not
                    // the GUI render.
                    if let img = vm.exportPreviewImage {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(RoundedRectangle(cornerRadius: 6)
                                .strokeBorder(Color.primary.opacity(0.15)))
                    }
                    if let progress = vm.exportProgressText {
                        Text(progress)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    if vm.exportResultPath != nil || vm.exportFramesDir != nil {
                        Button("Show in Finder") { vm.revealExportResult() }
                    }
                    Text("Offscreen 1600×1200, H.264 .mp4 (30 fps) + frame PNGs; style/tint/follow/orbit as shown. Runs on a cloned sim — the sim keeps running; Stop yields the partial video.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // V2.8 growth drives: the mode picker is the preset-setter for
            // the three additive drives (U/L/R); each drive's slider sits
            // WITH its parameter group (uniform with the food knobs, light at
            // the top of the light group, RDE at the top of the RDE group) so
            // activation and tuning live together. A drive at 0 disables its
            // group's parameters (they cannot influence growth) — the light
            // DIRECTION stays enabled: it also drives the render shading.
            collapsible("Growth & drives (V2.8)", isExpanded: $growthOpen) {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("Mode", selection: $vm.growthMode) {
                        ForEach(SimViewModel.GrowthMode.allCases) { m in
                            Text(m.label).tag(m)
                        }
                    }
                    .pickerStyle(.segmented)
                    Text("Mode sets the three drives (U/L/R); the drive sliders combine freely.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    slider("Uniform drive U", $vm.growthUniform, 0...2)
                    Toggle(isOn: $vm.divisionEnabled) { Text("Cell division") }
                    slider("Food increment", $vm.foodIncrement, 0...0.05)
                    slider("Split threshold", $vm.splitThreshold, 0.2...3)
                }
            }
            // V2.7 death knobs (the V2.5 open issue: death parameters were
            // CLI/preset-side only). Placed directly under Growth — death is
            // division's inverse and its threshold rides the same food
            // ledger. Collapsible like the export group. Preset/state loads
            // sync these sliders (the panel shows what runs); explicit user
            // input wins after a load.
            collapsible("Death (starvation)", isExpanded: $deathOpen) {
                VStack(alignment: .leading, spacing: 10) {
                    Toggle(isOn: $vm.deathEnabled) { Text("Starvation death") }
                    slider("Death threshold", $vm.deathThreshold, 0...3)
                    slider("Death window (steps)", $vm.deathSteps, 0...2000, step: 10, format: "%.0f")
                    Text("A cell whose food ledger stays below the threshold for the window's steps dies (ring splice closes the hole). Window 0 = never. Threshold defaults to the split threshold.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // V2.3 type layout: phase-0 polar typing (axis + stem/cap
            // thresholds; between them the lamel/dal band) and the two-phase
            // growth switch. Dragging a knob re-types the LIVE cells; the
            // button re-applies after a manual change.
            GroupBox("Type layout (V2.3)") {
                Toggle(isOn: $vm.typingEnabled) { Text("Polar typing (on reset + live)") }
                slider("Axis X", $vm.typeAxisX, -1...1)
                slider("Axis Y", $vm.typeAxisY, -1...1)
                slider("Axis Z", $vm.typeAxisZ, -1...1)
                slider("Stem threshold t1", $vm.typeT1, -1...1)
                slider("Cap threshold t2", $vm.typeT2, -1...1)
                Button("Re-type cells") { vm.retypeCells() }
                slider("Phase switch X (divisions)", $vm.phaseX, 0...20000, step: 100)
                slider("Lamel rate", $vm.lamelRate, 0...3)
                slider("Dal rate", $vm.dalRate, 0...3)
            }

            // V2.4 differentiation phase: freeze division, let the RDE type
            // the undifferentiated cells (v >= cap threshold -> cap, v <= stem
            // threshold -> stem; typed cells then ignore the RDE — type-memory
            // hysteresis). The button takes effect from the next frame.
            GroupBox("Differentiation (V2.4)") {
                Button(vm.differentiationPhase ? "Resume growth" : "Differentiate now") {
                    vm.toggleDifferentiation()
                }
                .buttonStyle(.borderedProminent)
                if vm.differentiationPhase {
                    Text("Division frozen — RDE typing active (mode: \(vm.typingEnabled ? "band" : "region")).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                slider("Cap threshold v ≥", $vm.typeCapV, 0...1)
                slider("Stem threshold v ≤", $vm.typeStemV, 0...1)
                Text("Works in RDE growth mode; with polar typing on the band matrix stays in charge.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // V2.8: the light DRIVE strength at the top of its parameter
            // group. With the drive at 0 the light/nutrient coupling
            // parameters are disabled (dormant — they cannot influence
            // growth); the light direction stays live (render shading).
            collapsible("Light (growth + shading)", isExpanded: $lightOpen) {
                VStack(alignment: .leading, spacing: 10) {
                    slider("Light drive L", $vm.growthLight, 0...2)
                    Group {
                        slider("Diffusion", $vm.lightDiffusionRate, 0...1)
                        slider("Nutrient/step", $vm.lightNutrient, 0...0.2)
                        slider("Decay", $vm.lightDecay, 0...0.2)
                        slider("Coupling base", $vm.lightBase, 0...1)
                        slider("March interval (steps)", $vm.lightInterval, 1...64, step: 1)
                    }
                    .disabled(vm.growthLight <= 0)
                    if vm.growthLight <= 0 {
                        Text("Light drive is 0 — nutrient coupling dormant.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    slider("Light azimuth°", $vm.lightAzimuth, 0...360, step: 5)
                    slider("Light elevation°", $vm.lightElevation, 0...180, step: 5)
                    Text("Direction also shades the render (always active).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // V2.8: the RDE DRIVE strength at the top of its parameter group;
            // with the drive at 0 the chemistry parameters are disabled (the
            // field is not stepped, so they cannot influence anything).
            collapsible("RDE (pattern chemistry)", isExpanded: $rdeOpen) {
                VStack(alignment: .leading, spacing: 10) {
                    slider("RDE drive R", $vm.growthRde, 0...2)
                    Group {
                        slider("Feed F", $vm.rdeF, 0...0.08)
                        slider("Kill k", $vm.rdeK, 0...0.09)
                        slider("Du", $vm.rdeDu, 0...1)
                        slider("Dv", $vm.rdeDv, 0...1)
                        slider("Coupling base", $vm.rdeBase, 0...1)
                        slider("Coupling gain", $vm.rdeGain, 0...8)
                    }
                    .disabled(vm.growthRde <= 0)
                    if vm.growthRde <= 0 {
                        Text("RDE drive is 0 — chemistry dormant (the RDE food term is light-gated: it feeds only where light falls).")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            GroupBox("Repulsion") {
                slider("Radius of influence", $vm.roi, 0.5...4)
                slider("Strength", $vm.repulsionStrength, 0...0.5)
            }

            GroupBox("Forces") {
                slider("Spring", $vm.springFactor, 0...1)
                slider("Planar", $vm.planarFactor, 0...1)
                slider("Bulge", $vm.bulgeFactor, 0...1)
                slider("Radius scale", $vm.radiusScale, 0.25...0.8)
            }

            GroupBox("Run") {
                slider("Steps / frame", $vm.stepsPerFrame, 1...64, step: 1)
                slider("Jitter (on reset)", $vm.jitter, 0...0.5)
                slider("Subdiv (on reset)", $vm.subdivision, 2...5, step: 1)
            }

            Divider()

            if let msg = vm.statusMessage {
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Cells: \(vm.cellCount)")
                Text("Mean link length: \(String(format: "%.4f", vm.meanLinkLength))")
                Text("Steps/s: \(String(format: "%.0f", vm.stepsPerSecond))")
            }
            .font(.system(.body, design: .monospaced))

            Text(vm.device.name)
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()
        }
    }

    private func slider(_ label: String, _ value: Binding<Double>, _ range: ClosedRange<Double>, step: Double? = nil, format: String = "%.2f") -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.callout)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            if let step {
                Slider(value: value, in: range, step: step)
            } else {
                Slider(value: value, in: range)
            }
        }
    }

    /// V2.7: collapsible panel section (the user's request) — a
    /// DisclosureGroup with a headline label, replacing the GroupBox for the
    /// sections this milestone touches (export, death). V2.8 uses the same
    /// helper for the drive groups (growth drives, light, RDE) — one shared
    /// pattern for every collapsible section.
    private func collapsible<Content: View>(_ title: String,
                                            isExpanded: Binding<Bool>,
                                            @ViewBuilder content: @escaping () -> Content) -> some View {
        DisclosureGroup(isExpanded: isExpanded) {
            content()
        } label: {
            Text(title).font(.headline)
        }
    }
}

struct MetalViewContainer: NSViewRepresentable {
    let vm: SimViewModel

    func makeNSView(context: Context) -> MetalView { vm.metalView }
    func updateNSView(_ view: MetalView, context: Context) {}
}

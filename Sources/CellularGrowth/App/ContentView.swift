//  ContentView.swift
//  Main window layout: Metal view + parameter panel.

import SwiftUI

struct ContentView: View {
    @ObservedObject var vm: SimViewModel

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
            Text("Cellular Growth — V2.4")
                .font(.headline)

            HStack(spacing: 8) {
                Button(vm.playing ? "Pause" : "Play") { vm.playing.toggle() }
                    .buttonStyle(.borderedProminent)
                Button("Reset") { vm.reset() }
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

            GroupBox("State") {
                HStack(spacing: 8) {
                    Button("Save State…") { vm.saveStatePanel() }
                    Button("Load State…") { vm.loadStatePanel() }
                }
            }

            GroupBox("Growth") {
                Picker("Mode", selection: $vm.growthMode) {
                    ForEach(SimViewModel.GrowthMode.allCases) { m in
                        Text(m.label).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                Toggle(isOn: $vm.divisionEnabled) { Text("Cell division") }
                slider("Food increment", $vm.foodIncrement, 0...0.05)
                slider("Split threshold", $vm.splitThreshold, 0.2...3)
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

            GroupBox("Light (growth + shading)") {
                slider("Diffusion", $vm.lightDiffusionRate, 0...1)
                slider("Nutrient/step", $vm.lightNutrient, 0...0.2)
                slider("Decay", $vm.lightDecay, 0...0.2)
                slider("Coupling base", $vm.lightBase, 0...1)
                slider("March interval (steps)", $vm.lightInterval, 1...64, step: 1)
                slider("Light azimuth°", $vm.lightAzimuth, 0...360, step: 5)
                slider("Light elevation°", $vm.lightElevation, 0...180, step: 5)
            }

            GroupBox("RDE (mode: RDE)") {
                slider("Feed F", $vm.rdeF, 0...0.08)
                slider("Kill k", $vm.rdeK, 0...0.09)
                slider("Du", $vm.rdeDu, 0...1)
                slider("Dv", $vm.rdeDv, 0...1)
                slider("Coupling base", $vm.rdeBase, 0...1)
                slider("Coupling gain", $vm.rdeGain, 0...8)
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

    private func slider(_ label: String, _ value: Binding<Double>, _ range: ClosedRange<Double>, step: Double? = nil) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.callout)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
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
}

struct MetalViewContainer: NSViewRepresentable {
    let vm: SimViewModel

    func makeNSView(context: Context) -> MetalView { vm.metalView }
    func updateNSView(_ view: MetalView, context: Context) {}
}

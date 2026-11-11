//  ContentView.swift
//  Main window layout: Metal view + parameter panel.

import SwiftUI

struct ContentView: View {
    @ObservedObject var vm: SimViewModel

    var body: some View {
        HStack(spacing: 0) {
            MetalViewContainer(vm: vm)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            controlPanel
                .frame(width: 290)
                .padding(12)
                .background(Color(nsColor: .windowBackgroundColor))
        }
    }

    private var controlPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Cellular Growth — M6")
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
                }
                .pickerStyle(.segmented)
                if vm.renderStyle == .xray {
                    slider("X-ray gain", $vm.xrayGain, 0.05...2)
                }
                slider("Edges", $vm.edges, 0...1)
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

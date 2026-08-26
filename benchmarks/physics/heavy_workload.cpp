#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/EnergyDiagnostic.hpp"
#include "engine/modules/diagnostics/PlasmaWaveVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/Injectors.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>

int main()
{
    // Heavy Workload Parameters (~2.04M Particles, Order-3 Cubic Splines, 1000 Steps)
    constexpr std::size_t grid_cells = 2048;
    constexpr double      dx         = 0.05;
    constexpr double      dt         = 0.001;
    constexpr std::size_t ppc        = 1000;
    constexpr std::size_t nsteps     = 1000;
    constexpr std::size_t BS         = 64;
    constexpr float       target_n0  = 2.0f;

    assert(dt <= 0.95 * dx && "CFL condition violated.");

    Grid grid(grid_cells, dx, /*guards=*/4);

    // Order-3 Cubic Spline Shape (4-point stencil) for maximum math load
    using Shape     = kernels::shapes::SplineShape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::NoInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    // 1. Initialize Engine & Native Wave Perturbation
    EngineT engine_instance{grid, ppc, target_n0};

    const double    L  = static_cast<double>(grid_cells) * dx;
    const double    k  = 2.0 * std::numbers::pi / L;
    constexpr float v0 = 0.05f;

    auto& particles = engine_instance.particles();
    particles.init_positions_uniform(grid);
    particles.init_velocities_wave(v0, k, /*longitudinal_only=*/true);

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    pico::diagnostics::PlasmaWaveVerifier verifier(dt, dx, ppc, target_n0);
    using EnergyDiag = pico::diagnostics::EnergyDiagnostic<EngineT>;

    PICApp app(std::move(wrapper), dt);

    // 2. Main Execution & Wall-Clock Benchmark Loop
    const auto start_wall_time = std::chrono::high_resolution_clock::now();

    app.run(nsteps,
            [&](int /*step*/)
            {
                const auto m = EnergyDiag::evaluate(concrete_wrapper->engine());
                verifier.record_step(m.e_ex, m.e_total());
            });

    const auto   end_wall_time = std::chrono::high_resolution_clock::now();
    const double total_sec     = std::chrono::duration<double>(end_wall_time - start_wall_time).count();

    // 3. Performance Metrics & Verification
    const auto res = verifier.verify(/*energy_drift_tol_pct=*/2.0, /*freq_tol_pct=*/5.0);

    const std::size_t total_particles = concrete_wrapper->engine().particles().active_particles();
    const double      mup_s           = ((static_cast<double>(total_particles) * nsteps) / total_sec) / 1e6;

    // 4. Reporting
    std::ostringstream title_ss;
    title_ss << "Heavy Workload & Physics Report (n0 = " << std::fixed << std::setprecision(1) << target_n0 << ")";

    pico::ui::VerificationReport report("Heavy Plasma Wave Benchmark", res.passed, title_ss.str());

    // Setup & Performance Section
    report.add_row("Total Active Particles", std::to_string(total_particles));
    report.add_fixed_row("Execution Time", total_sec, 3, "s");
    report.add_fixed_row("Throughput", mup_s, 2, "MUP/s");

    // Physics Consistency Section
    report.add_pct_row("Max Energy Drift", res.max_energy_drift_pct, true);
    report.add_fixed_row("Measured Frequency (wp)", res.measured_freq, 4, "rad/s");
    report.add_fixed_row("Expected Frequency (wp)", res.expected_freq, 4, "rad/s");
    report.add_pct_row("Frequency Error", res.freq_error_pct, res.passed);

    report.print();
    return report.passed() ? 0 : 1;
}

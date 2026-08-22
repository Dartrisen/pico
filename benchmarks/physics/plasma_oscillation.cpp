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
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>

int main()
{
    constexpr std::size_t grid_cells = 256;
    constexpr double      dx         = 0.1;
    constexpr double      dt         = 0.002;
    constexpr std::size_t ppc        = 100;
    constexpr std::size_t nsteps     = 1000;
    constexpr std::size_t BS         = 64;
    constexpr float       target_n0  = 2.0f;

    assert(dx > 0.95 * dt && "CFL condition violated.");

    Grid grid(grid_cells, dx);

    using Shape     = kernels::shapes::Shape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::NoInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    // 1. Initialize Engine & Setup Native Particle Wave Perturbation
    EngineT engine_instance{grid, ppc, target_n0};

    const double    L  = static_cast<double>(grid_cells) * dx;
    const double    k  = 2.0 * std::numbers::pi / L;
    constexpr float v0 = 0.05f;

    auto& particles = engine_instance.particles();
    particles.init_positions_uniform(grid);
    particles.init_velocities_wave(v0, k);

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    // 2. Setup Verification & Diagnostics
    pico::diagnostics::PlasmaWaveVerifier verifier(dt, dx, ppc, target_n0);
    using EnergyDiag = pico::diagnostics::EnergyDiagnostic<EngineT>;

    PICApp app(std::move(wrapper), dt);

    // 3. Execution Loop
    app.run(nsteps,
            [&](int /*step*/)
            {
                const auto& eng = concrete_wrapper->engine();
                const auto  m   = EnergyDiag::evaluate(eng);
                verifier.record_step(m.e_ex, m.e_total());
            });

    // 4. Verification Analysis & Reporting
    const auto res = verifier.verify(/*drift_tol=*/2.0, /*freq_tol=*/5.0);

    std::ostringstream title_ss;
    title_ss << "Physics Verification Summary (n0 = " << std::fixed << std::setprecision(1) << target_n0 << ")";

    pico::ui::VerificationReport report("Plasma Wave Physics Verification", res.passed, title_ss.str());

    report.add_pct_row("Max Energy Drift", res.max_energy_drift_pct, true);
    report.add_fixed_row("Measured Frequency (wp)", res.measured_freq, 4, "rad/s");
    report.add_fixed_row("Expected Frequency (wp)", res.expected_freq, 4, "rad/s");
    report.add_pct_row("Frequency Error", res.freq_error_pct, res.passed);

    report.print();
    return report.passed() ? 0 : 1;
}

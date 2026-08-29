#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/Thermalizing.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/EnergyDiagnostic.hpp"
#include "engine/modules/diagnostics/EnergyVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/Injectors.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <tuple>

int main()
{
    // 1. Simulation Parameters
    constexpr std::size_t grid_cells = 256;
    constexpr double      dx         = 0.1;
    constexpr double      dt         = 0.002;
    constexpr std::size_t ppc        = 100;
    constexpr std::size_t nsteps     = 1000;
    constexpr std::size_t BS         = 64;

    assert(dt <= 0.95 * dx && "CFL condition violated: dx must be greater than 0.95 * dt for stability.");

    Grid grid(grid_cells, dx);

    using Shape     = kernels::shapes::Shape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::ThermalizingParticleBoundary<BS>;
    using Injector  = pico::modules::injector::NoInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    // 2. Engine Initialization & Particle Setup
    EngineT engine_instance{grid};

    const double    L  = static_cast<double>(grid_cells) * dx;
    const double    k  = 2.0 * std::numbers::pi / L;
    constexpr float v0 = 0.05f;

    auto uniform_density = [](double /*x*/) -> float { return 1.0f; };

    // Wave velocity profile with 64-bit phase evaluation
    auto wave_velocity = [v0, k](double x) -> std::tuple<float, float, float>
    {
        const double phase  = k * x;
        const float  v_wave = static_cast<float>(v0 * std::sin(phase));
        const float  v_y    = static_cast<float>(v0 * std::cos(phase));
        return {v_wave, v_y, 0.0f};
    };

    // Register species via new engine profile API
    engine_instance.add_species_profile(ppc, uniform_density, wave_velocity,
                                        /*base_charge=*/-1.0f,
                                        /*base_mass=*/1.0f);

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    // 3. Diagnostics Setup
    pico::diagnostics::EnergyVerifier energy_verifier(/*drift_tolerance_pct=*/2.0);
    using EnergyDiag = pico::diagnostics::EnergyDiagnostic<EngineT>;

    // 4. Execution & Step Diagnostics
    PICApp app(std::move(wrapper), dt);
    app.run(nsteps,
            [&](int /*step*/)
            {
                const auto m = EnergyDiag::evaluate(concrete_wrapper->engine());
                energy_verifier.record_step(m.e_field_total(), m.e_kin_total());
            });

    // 5. Verification & Reporting
    const auto res = energy_verifier.verify();

    pico::ui::VerificationReport report("Energy Conservation Verification", res.passed);

    report.add_sci_row("Initial Energy", res.initial_energy, 4, "mc^2");
    report.add_sci_row("Final Energy", res.final_energy, 4, "mc^2");
    report.add_pct_row("Max Energy Drift", res.max_energy_drift_pct, res.passed);
    report.add_pct_row("Avg Energy Drift", res.avg_energy_drift_pct, true);

    report.print();
    return report.passed() ? 0 : 1;
}
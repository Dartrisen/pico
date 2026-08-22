#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/CurrentDiagnostic.hpp"
#include "engine/modules/diagnostics/CurrentVerifier.hpp"
#include "engine/modules/diagnostics/EnergyDiagnostic.hpp"
#include "engine/modules/diagnostics/EnergyVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/Injectors.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

// Applies initial ballistic drift velocity along X
template <typename Engine>
void apply_drift_velocity(Engine& engine, std::size_t grid_cells, double dx, float v_drift)
{
    auto&        particles  = engine.particles();
    const double L          = static_cast<double>(grid_cells) * dx;
    const double dx_p       = L / static_cast<double>(particles.active_particles());
    std::size_t  global_idx = 0;

    for (auto& block : particles)
    {
        for (std::size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            const double x0     = (static_cast<double>(global_idx) + 0.5) * dx_p;
            block.position_x[i] = static_cast<float>(x0);

            const float m       = block.mass[i];
            block.momentum_x[i] = m * v_drift;
            block.momentum_y[i] = 0.0f;
            block.momentum_z[i] = 0.0f;
        }
    }
}

// Zeroes Electric field components to maintain unperturbed ballistic drift
template <typename Engine>
void freeze_electric_field(Engine& eng, std::size_t grid_cells)
{
    auto& E = eng.fields().E;
    for (std::size_t i = 0; i < grid_cells; ++i)
    {
        E.field_x(i) = 0.0f;
        E.field_y(i) = 0.0f;
        E.field_z(i) = 0.0f;
    }
}

int main()
{
    constexpr std::size_t grid_cells = 256;
    constexpr double      dx         = 0.1;
    constexpr double      dt         = 0.002;
    constexpr std::size_t ppc        = 100;
    constexpr std::size_t nsteps     = 1000;
    constexpr std::size_t BS         = 64;
    constexpr float       v_drift    = 0.05f;
    constexpr float       target_n0  = 1.0f;

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

    // 1. Initialize Engine & Setup Drift
    EngineT engine_instance{grid, ppc, target_n0};
    apply_drift_velocity(engine_instance, grid_cells, dx, v_drift);

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    // 2. Diagnostic Verifiers & Diagnostic Reducers Setup
    const double expected_current = (-1.0 * static_cast<double>(target_n0) * static_cast<double>(v_drift)) / dx;

    pico::diagnostics::EnergyVerifier  energy_verifier(/*drift_tolerance_pct=*/2.0);
    pico::diagnostics::CurrentVerifier current_verifier(expected_current, /*tolerance_pct=*/5.0);

    using EnergyDiag  = pico::diagnostics::EnergyDiagnostic<EngineT>;
    using CurrentDiag = pico::diagnostics::CurrentDiagnostic<EngineT>;

    // 3. Execution Loop
    PICApp app(std::move(wrapper), dt);
    app.run(nsteps,
            [&](int /*step*/)
            {
                auto& eng = concrete_wrapper->engine();

                freeze_electric_field(eng, grid_cells);

                // Standardized energy and current diagnostic reductions
                const auto energy_m  = EnergyDiag::evaluate(eng);
                const auto current_m = CurrentDiag::evaluate(eng, expected_current);

                energy_verifier.record_step(energy_m.e_field_total(), energy_m.e_kin);
                current_verifier.record_step(current_m.avg_jx, current_m.max_ampere_res);
            });

    // 4. Verification & Reporting
    const auto energy_res  = energy_verifier.verify();
    const auto current_res = current_verifier.verify();

    pico::ui::VerificationReport report("Current & Flux Conservation Verification", energy_res.passed && current_res.passed);

    report.add_sci_row("Initial Energy", energy_res.initial_energy);
    report.add_sci_row("Final Energy", energy_res.final_energy);
    report.add_pct_row("Max Energy Drift", energy_res.max_energy_drift_pct, energy_res.passed);
    report.add_pct_row("Avg Energy Drift", energy_res.avg_energy_drift_pct, true);
    report.add_fixed_row("Expected Drift Current (Jx)", current_res.expected_current);
    report.add_fixed_row("Measured Avg Deposited Jx", current_res.avg_measured_current);
    report.add_pct_row("Current Error", current_res.current_error_pct, current_res.passed);
    report.add_sci_row("Max Ampere Residual", current_res.max_ampere_residual);

    report.print();
    return report.passed() ? 0 : 1;
}

#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/boundary/SilverMuller.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/EnergyDiagnostic.hpp"
#include "engine/modules/diagnostics/FieldEnvelopeDiagnostic.hpp"
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

int main()
{
    // Simulation Domain & Time Parameters
    constexpr std::size_t grid_cells = 256; // Domain length L = 12.8
    constexpr double      dx         = 0.05;
    constexpr double      dt         = 0.001;
    constexpr std::size_t BS         = 64;

    constexpr double t_check = 7.0;  // Expected envelope peak at x = 4.0
    constexpr double t_final = 20.0; // Pulse exits domain

    const std::size_t mid_check_step = static_cast<std::size_t>(std::round(t_check / dt));
    const std::size_t nsteps         = static_cast<std::size_t>(std::round(t_final / dt));

    constexpr float       a0          = 1.0f;
    constexpr float       tau         = 1.0f;
    constexpr float       t_peak      = 3.0f;
    constexpr std::size_t inject_cell = 0;

    assert(dx > 0.95 * dt && "CFL condition violated.");

    Grid grid(grid_cells, dx);

    using Shape     = kernels::shapes::Shape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::SilverMullerFieldBoundary<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::PlaneWaveLaserInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    // 1. Initialize Laser Engine & Boundary Conditions
    Injector laser_injector(inject_cell, a0, tau, t_peak);
    EngineT  engine_instance{grid, /*ppc=*/0, BoundaryF{}, BoundaryP{}, std::move(laser_injector), /*n0=*/0.0f};

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    using EnergyDiag   = pico::diagnostics::EnergyDiagnostic<EngineT>;
    using EnvelopeDiag = pico::diagnostics::FieldEnvelopeDiagnostic<EngineT>;

    PICApp app(std::move(wrapper), dt);

    double max_field_energy   = 0.0;
    double final_field_energy = 0.0;
    double measured_peak_x    = 0.0;

    // 2. Main Simulation Benchmark Loop
    app.run(nsteps,
            [&](int step_idx)
            {
                const std::size_t step     = static_cast<std::size_t>(step_idx);
                const auto        energy_m = EnergyDiag::evaluate(concrete_wrapper->engine());
                const double      e_field  = energy_m.e_field_total();

                max_field_energy = std::max(max_field_energy, e_field);

                if (step == mid_check_step)
                {
                    const auto env  = EnvelopeDiag::evaluate(concrete_wrapper->engine(), /*smoothing_window=*/30);
                    measured_peak_x = env.peak_x;
                }

                if (step == nsteps - 1)
                {
                    final_field_energy = e_field;
                }
            });

    // 3. Verification Metrics & Envelope Tracking Analysis
    const double expected_peak_x = (static_cast<double>(mid_check_step) * dt) - static_cast<double>(t_peak);
    const double pos_error       = std::abs(measured_peak_x - expected_peak_x);

    const double reflection_pct = (final_field_energy / max_field_energy) * 100.0;
    const bool   passed         = (reflection_pct < 2.0) && (pos_error <= 2.0 * dx) && (max_field_energy > 0.0);

    // 4. Verification Output Report
    pico::ui::VerificationReport report("Laser Wave Injection & Boundary Absorption Verification", passed);

    report.add_sci_row("Peak Transverse Energy", max_field_energy);
    report.add_sci_row("Final Residual Energy", final_field_energy);
    report.add_pct_row("Boundary Reflection Ratio", reflection_pct, reflection_pct < 2.0);
    report.add_fixed_row("Measured Peak Energy Position", measured_peak_x, 4, "x");
    report.add_fixed_row("Expected Peak Energy Position", expected_peak_x, 4, "x");

    report.print();
    return report.passed() ? 0 : 1;
}

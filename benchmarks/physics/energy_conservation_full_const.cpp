#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/ConsoleVisualiser.hpp"
#include "engine/modules/diagnostics/EnergyDiagnostic.hpp"
#include "engine/modules/diagnostics/EnergyVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/Injectors.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>

int main()
{
    // 1. Reference Physical Units (omega_0 Normalization: c = 1, omega_0 = 1)
    constexpr double c0       = 2.99792458e8;
    constexpr double lambda_0 = 1.0e-6;
    constexpr double omega_0  = 2.0 * std::numbers::pi * c0 / lambda_0;
    constexpr double unit_len = c0 / omega_0; // ~0.159155 um per code length unit

    // 2. Physical problem dimensions
    constexpr double plasma_len_si = 100.0e-6; // 100 microns plasma slab
    constexpr double vacuum_gap_si = 15.0e-6;  // 15 microns front gap
    constexpr double trail_gap_si  = 15.0e-6;  // 15 microns rear gap
    constexpr float  n_const       = 0.25f;    // Constant core density in n_critical units

    // Thermal velocities (v_th_e = 0.035 ensures lambda_D >= 0.3 * dx for finite grid stability)
    constexpr float mass_ratio  = 1836.0f;
    constexpr float v_thermal_e = 0.035f;
    const float     v_thermal_p = v_thermal_e / std::sqrt(mass_ratio);

    // Code length dimensions
    constexpr double x_start  = vacuum_gap_si / unit_len;
    constexpr double L_plasma = plasma_len_si / unit_len;
    constexpr double x_end    = x_start + L_plasma;
    constexpr double L_domain = (vacuum_gap_si + plasma_len_si + trail_gap_si) / unit_len;

    // Grid Setup
    constexpr std::size_t cells_per_lambda = 128; // Cells per lambda_0
    constexpr double      dx               = (2.0 * std::numbers::pi) / static_cast<double>(cells_per_lambda);
    constexpr double      dt               = 0.95 * dx;
    const std::size_t     grid_cells       = static_cast<std::size_t>(std::ceil(L_domain / dx));

    constexpr std::size_t ppc    = 10;
    constexpr std::size_t nsteps = 1000;
    constexpr std::size_t BS     = 64;

    // 3. Debye Length & Stability Assertions
    const double omega_pe    = std::sqrt(static_cast<double>(n_const));
    const double lambda_D    = static_cast<double>(v_thermal_e) / omega_pe;
    const double debye_to_dx = lambda_D / dx;

    assert(dt <= 0.95 * dx && "CFL condition violated: dt must be <= 0.95 * dx for stability.");
    assert(debye_to_dx >= 0.3 && "Finite grid instability: lambda_D must be >= 0.3 * dx");

    Grid grid(grid_cells, dx, /*guards=*/4);

    using Shape     = kernels::shapes::Shape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::OptEsirkepovDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::NoInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    // 4. Engine Initialization & Bounded Constant Density Setup
    EngineT engine_instance{grid};

    constexpr double ramp_buffer = 5.0e-6 / unit_len;
    const double     x_ramp_in   = x_start + ramp_buffer;
    const double     x_ramp_out  = x_end - ramp_buffer;

    auto constant_density = [=](double x) -> float
    {
        if (x < x_start || x > x_end)
            return 0.0f;

        // Smooth edge transitions at boundary walls prevent artificial sheath shocks
        if (x < x_ramp_in)
            return static_cast<float>((x - x_start) / ramp_buffer) * n_const;
        if (x > x_ramp_out)
            return static_cast<float>((x_end - x) / ramp_buffer) * n_const;

        return n_const; // Flat, uniform plateau density
    };

    // Add Species 0: Thermal Electrons
    engine_instance.add_species_thermal(ppc, constant_density, v_thermal_e,
                                        /*base_charge=*/-1.0f,
                                        /*base_mass=*/1.0f,
                                        /*vx_drift=*/0.0f, /*vy_drift=*/0.0f, /*vz_drift=*/0.0f,
                                        /*seed=*/1337u);

    // Add Species 1: Thermal Protons (Shared seed 1337u guarantees initial zero charge density rho=0)
    engine_instance.add_species_thermal(ppc, constant_density, v_thermal_p,
                                        /*base_charge=*/1.0f,
                                        /*base_mass=*/mass_ratio,
                                        /*vx_drift=*/0.0f, /*vy_drift=*/0.0f, /*vz_drift=*/0.0f,
                                        /*seed=*/1337u);

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    // 5. Diagnostics Setup
    pico::diagnostics::EnergyVerifier energy_verifier(/*drift_tolerance_pct=*/2.0);
    using EnergyDiag = pico::diagnostics::EnergyDiagnostic<EngineT>;

    // 6. Execution & Step Diagnostics
    PICApp app(std::move(wrapper), dt);
    using Visualizer = pico::diagnostics::ConsoleVisualizer<EngineT>;

    app.run(nsteps,
            [&](int step)
            {
                const auto m = EnergyDiag::evaluate(concrete_wrapper->engine());

                assert(m.e_kin_species.size() == 2 && "Expected 2 species kinetic energy components.");
                [[maybe_unused]] const double e_kin_electrons = m.e_kin_species[0];
                [[maybe_unused]] const double e_kin_protons   = m.e_kin_species[1];

                energy_verifier.record_step(m.e_field_total(), m.e_kin_total());

                if (step % 50 == 0)
                {
                    Visualizer::draw_density_sparkline(concrete_wrapper->engine(), /*display_width=*/64);
                    std::cout << "[Step " << std::setw(5) << step << "] "
                              << "Field E: " << std::scientific << std::setprecision(3) << m.e_field_total() << " | Kin E: " << m.e_kin() << " | Total E: " << m.e_total() << "\n";
                }
            });

    // 7. Verification & Reporting
    const auto res = energy_verifier.verify();

    pico::ui::VerificationReport report("Constant Density Proton-Electron Energy Conservation Verification", res.passed);

    report.add_fixed_row("Grid Step (dx)", dx, 5, "");
    report.add_fixed_row("Debye Length (lambda_D)", lambda_D, 5, "");
    report.add_fixed_row("lambda_D / dx Ratio", debye_to_dx, 3, " (>= 0.3)");
    report.add_sci_row("Initial Energy", res.initial_energy, 4, "mc^2");
    report.add_sci_row("Final Energy", res.final_energy, 4, "mc^2");
    report.add_pct_row("Max Energy Drift", res.max_energy_drift_pct, res.passed);
    report.add_pct_row("Avg Energy Drift", res.avg_energy_drift_pct, true);

    report.print();
    return report.passed() ? 0 : 1;
}
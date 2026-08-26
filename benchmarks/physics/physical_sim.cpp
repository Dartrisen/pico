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
#include "engine/modules/diagnostics/FieldEnvelopeDiagnostic.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/Injectors.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>

int main()
{
    constexpr double c0       = 2.99792458e8;
    constexpr double lambda_0 = 1.0e-6;
    constexpr double omega_0  = 2.0 * std::numbers::pi * c0 / lambda_0;
    constexpr double unit_len = c0 / omega_0;

    constexpr double plasma_len_si = 100.0e-6;
    constexpr double vacuum_gap_si = 15.0e-6;
    constexpr double trail_gap_si  = 15.0e-6;
    constexpr float  n_peak_nc     = 0.25f;

    constexpr float v_th_e = 0.05f;
    constexpr float mp_me  = 1836.0f;
    const float     v_th_p = v_th_e / std::sqrt(mp_me);

    constexpr double x_start  = vacuum_gap_si / unit_len;
    constexpr double L_plasma = plasma_len_si / unit_len;
    constexpr double x_end    = x_start + L_plasma;
    constexpr double L_domain = (vacuum_gap_si + plasma_len_si + trail_gap_si) / unit_len;

    constexpr float       laser_a0     = 0.05f;
    constexpr float       tau_duration = 30.0f;
    constexpr float       t_peak       = 75.0f;
    constexpr std::size_t inject_cell  = 0;

    constexpr std::size_t cells_per_lambda = 128;
    constexpr double      dx               = (2.0 * std::numbers::pi) / static_cast<double>(cells_per_lambda);
    constexpr double      dt               = 0.90 * dx;
    const std::size_t     grid_cells       = static_cast<std::size_t>(std::ceil(L_domain / dx));

    constexpr std::size_t ppc    = 10;
    constexpr std::size_t nsteps = 4000;
    constexpr std::size_t BS     = 64;

    const double omega_pe    = std::sqrt(static_cast<double>(n_peak_nc));
    const double lambda_D    = static_cast<double>(v_th_e) / omega_pe;
    const double debye_to_dx = lambda_D / dx;

    assert(dt <= 0.95 * dx && "CFL condition violated");
    assert(debye_to_dx >= 0.3 && "Finite grid instability threshold violated");

    Grid grid(grid_cells, dx, /*guards=*/4);

    using Shape     = kernels::shapes::Shape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::RelativisticBorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::OptEsirkepovDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::PlaneWaveLaserInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    Injector laser_injector(inject_cell, laser_a0, tau_duration, t_peak);

    constexpr float n_floor     = 1e-4f * n_peak_nc;
    auto            plasma_ramp = [=](double x) -> float
    {
        if (x < x_start || x > x_end)
            return 0.0f;
        const float ramp_fraction = static_cast<float>((x - x_start) / L_plasma);
        return n_floor + (n_peak_nc - n_floor) * ramp_fraction;
    };

    EngineT engine_instance{grid, BoundaryF{}, BoundaryP{}, std::move(laser_injector)};
    engine_instance.set_sort_frequency(10);

    engine_instance.add_species_thermal(ppc, plasma_ramp, v_th_e, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1337u);
    engine_instance.add_species_thermal(ppc, plasma_ramp, v_th_p, 1.0f, mp_me, 0.0f, 0.0f, 0.0f, 1337u);

    const std::size_t total_particles = grid.physical_size() * ppc * engine_instance.num_species();

    pico::diagnostics::EnergyVerifier energy_verifier(/*drift_tolerance_pct=*/2.0);
    using EnergyDiag   = pico::diagnostics::EnergyDiagnostic<EngineT>;
    using FieldEnvDiag = pico::diagnostics::FieldEnvelopeDiagnostic<EngineT>;
    using Visualizer   = pico::diagnostics::ConsoleVisualizer<EngineT>;

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    PICApp app(std::move(wrapper), static_cast<float>(dt));

    const auto start_wall_time = std::chrono::high_resolution_clock::now();

    app.run(nsteps,
            [&](int step)
            {
                if (step % 50 == 0)
                {
                    const auto   m         = EnergyDiag::evaluate(concrete_wrapper->engine());
                    const auto   laser_env = FieldEnvDiag::evaluate(concrete_wrapper->engine(), /*smoothing_window=*/3);
                    const double e_inj     = concrete_wrapper->engine().injector().injected_energy();

                    energy_verifier.record_step(m.e_field_total(), m.e_kin_total(), e_inj);
                    if (step > 0)
                        std::cout << "\033[1A\033[K";

                    Visualizer::draw_density_sparkline(concrete_wrapper->engine(), /*display_width=*/64);

                    std::cout << "[Step " << std::setw(5) << step << "] "
                              << "Field E: " << std::scientific << std::setprecision(3) << m.e_field_total() << " | Kin E: " << m.e_kin() << " | Inj E: " << e_inj << "\n";
                }
            });

    const auto   end_wall_time = std::chrono::high_resolution_clock::now();
    const double total_sec     = std::chrono::duration<double>(end_wall_time - start_wall_time).count();
    const double mup_s         = ((static_cast<double>(total_particles) * nsteps) / total_sec) / 1e6;

    const auto res = energy_verifier.verify();

    pico::ui::VerificationReport report("Laser-Plasma Energy Conservation Verification", res.passed);

    report.add_fixed_row("Grid Step (dx)", dx, 5, "");
    report.add_fixed_row("Debye Length (lambda_D)", lambda_D, 5, "");
    report.add_fixed_row("lambda_D / dx Ratio", debye_to_dx, 3, " (>= 0.3)");
    report.add_sci_row("Initial Energy E(0)", res.initial_energy, 4, "mc^2");
    report.add_sci_row("Total Injected Energy", res.final_injected_energy, 4, "mc^2");
    report.add_sci_row("Final Total Energy", res.final_energy, 4, "mc^2");
    report.add_pct_row("Max Energy Drift", res.max_energy_drift_pct, res.passed);
    report.add_pct_row("Avg Energy Drift", res.avg_energy_drift_pct, true);
    report.add_fixed_row("Throughput", mup_s, 2, "MUP/s");

    report.print();
    return res.passed ? 0 : 1;
}

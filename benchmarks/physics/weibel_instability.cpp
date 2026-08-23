#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/EnergyDiagnostic.hpp"
#include "engine/modules/diagnostics/WeibelVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/Injectors.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>

int main()
{
    // Weibel Instability Configuration (1D3V Electromagnetic)
    constexpr std::size_t grid_cells = 256;
    constexpr double      dx         = 0.2;
    constexpr double      dt         = 0.05;
    constexpr std::size_t ppc        = 500;
    constexpr std::size_t nsteps     = 2000;
    constexpr std::size_t BS         = 256;
    constexpr float       target_n0  = 1.0f;

    // Temperature Anisotropy: Transverse thermal velocities much higher than longitudinal (Ty, Tz >> Tx)
    constexpr double v_th_x = 0.01;
    constexpr double v_th_y = 0.15;
    constexpr double v_th_z = 0.15;

    assert(dx > 0.95 * dt && "CFL condition violated.");

    Grid grid(grid_cells, dx, /*guards=*/4);

    using Shape     = kernels::shapes::SplineShape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::OptEsirkepovDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::NoInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    // 1. Initialize Engine & Anisotropic Thermal State
    EngineT engine_instance{grid, ppc, target_n0};

    const double L = static_cast<double>(grid_cells) * dx;
    const double k = 2.0 * std::numbers::pi / L;

    auto& particles = engine_instance.particles();
    particles.init_positions_uniform(grid);
    particles.init_velocities_thermal(v_th_x, v_th_y, v_th_z, 0.0f, 0.0f, 0.0f, /*seed=*/42);

    // Seed a tiny magnetic field perturbation B_z(x) = B0 * cos(k * x) to kickstart linear growth
    const float       B0     = 1e-4f;
    const std::size_t G      = grid.guard_cells();
    auto&             fields = engine_instance.fields();

    for (std::size_t i = 0; i < grid.total_size(); ++i)
    {
        double x            = (static_cast<double>(i) - static_cast<double>(G) + 0.5) * dx;
        fields.B.field_z(i) = B0 * std::cos(static_cast<float>(k * x));
    }

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    pico::diagnostics::WeibelVerifier verifier(dt, v_th_x, v_th_y, target_n0);
    using EnergyDiag = pico::diagnostics::EnergyDiagnostic<EngineT>;

    PICApp app(std::move(wrapper), dt);

    // 2. Execution & Benchmark Loop
    const auto start_wall_time = std::chrono::high_resolution_clock::now();

    app.run(nsteps,
            [&](int /*step*/)
            {
                const auto m = EnergyDiag::evaluate(concrete_wrapper->engine());
                verifier.record_step(m.e_field_magnetic(), m.e_total());
            });

    const auto   end_wall_time = std::chrono::high_resolution_clock::now();
    const double total_sec     = std::chrono::duration<double>(end_wall_time - start_wall_time).count();

    // 3. Physical Verification & Performance Metrics
    const auto res = verifier.verify(/*energy_drift_tol_pct=*/5.0, /*gamma_tol_pct=*/35.0);

    const std::size_t total_particles = concrete_wrapper->engine().particles().active_particles();
    const double      mup_s           = ((static_cast<double>(total_particles) * nsteps) / total_sec) / 1e6;

    // 4. Output Summary Report via your VerificationReport framework
    std::ostringstream title_ss;
    title_ss << "Weibel Instability (Anisotropy Ty/Tx = " << std::fixed << std::setprecision(1) << (v_th_y / v_th_x) << ")";

    pico::ui::VerificationReport report("Weibel Instability Benchmark", res.passed, title_ss.str());

    report.add_row("Total Active Particles", std::to_string(total_particles));
    report.add_fixed_row("Execution Time", total_sec, 3, "s");
    report.add_fixed_row("Throughput", mup_s, 2, "MUP/s");

    report.add_pct_row("Max Energy Drift", res.max_energy_drift_pct, true);
    report.add_fixed_row("Measured Growth Rate (gamma)", res.measured_gamma, 5, "omega0^-1");
    report.add_fixed_row("Expected Growth Rate (gamma)", res.expected_gamma, 5, "omega0^-1");
    report.add_pct_row("Growth Rate Error", res.gamma_error_pct, res.passed);

    report.print();
    return report.passed() ? 0 : 1;
}
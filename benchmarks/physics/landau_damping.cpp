#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/EnergyDiagnostic.hpp"
#include "engine/modules/diagnostics/LandauVerifier.hpp"
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
#include <tuple>

int main()
{
    // Tuning for k * lambda_D ~ 0.31 with reduced PPC
    constexpr std::size_t grid_cells = 256;
    constexpr double      dx         = 0.1;
    constexpr double      dt         = 0.002;
    constexpr std::size_t ppc        = 1000;
    constexpr std::size_t nsteps     = 4000;
    constexpr std::size_t BS         = 256;
    constexpr float       target_n0  = 1.0f;

    constexpr float  v1   = 0.08f; // Scaled wave perturbation to remain above noise floor at 1000 PPC
    constexpr double v_th = 4.0;   // Thermal velocity for k*lambda_D ~ 0.31

    assert(dx > 0.95 * dt && "CFL condition violated.");

    Grid grid(grid_cells, dx, /*guards=*/4);

    using Shape     = kernels::shapes::SplineShape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::NoInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    // 1. Initialize Engine & Native Thermal State
    EngineT engine_instance{grid, ppc, target_n0};

    const double L = static_cast<double>(grid_cells) * dx;
    const double k = 2.0 * std::numbers::pi / L;

    auto& particles = engine_instance.particles();
    particles.init_positions_uniform(grid);

    // Portable Box-Muller normal generator to eliminate OS-dependent std::normal_distribution variance
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> u_dist(1e-7f, 1.0f - 1e-7f);

    auto sample_normal = [&](float mean, float stddev) -> float
    {
        const float u1 = u_dist(rng);
        const float u2 = u_dist(rng);
        const float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * std::numbers::pi_v<float> * u2);
        return mean + z0 * stddev;
    };

    const float float_vth = static_cast<float>(v_th);

    particles.init_velocities_profile(
            [&](double x) -> std::tuple<float, float, float>
            {
                const float v_wave = v1 * std::sin(static_cast<float>(k * x));
                return {sample_normal(0.0f, float_vth) + v_wave, sample_normal(0.0f, float_vth), sample_normal(0.0f, float_vth)};
            });

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    pico::diagnostics::LandauDampingVerifier verifier(dt, dx, k, v_th, target_n0);
    using EnergyDiag = pico::diagnostics::EnergyDiagnostic<EngineT>;

    PICApp app(std::move(wrapper), dt);

    // 2. Execution & Wall-Clock Benchmark Loop
    const auto start_wall_time = std::chrono::high_resolution_clock::now();

    app.run(nsteps,
            [&](int /*step*/)
            {
                const auto m = EnergyDiag::evaluate(concrete_wrapper->engine());
                verifier.record_step(m.e_ex, m.e_total());
            });

    const auto   end_wall_time = std::chrono::high_resolution_clock::now();
    const double total_sec     = std::chrono::duration<double>(end_wall_time - start_wall_time).count();

    // 3. Physical Verification & Performance Metrics
    const auto res = verifier.verify(/*energy_drift_tol_pct=*/2.0, /*freq_tol_pct=*/5.0, /*gamma_tol_pct=*/15.0);

    const std::size_t total_particles = concrete_wrapper->engine().particles().active_particles();
    const double      mup_s           = ((static_cast<double>(total_particles) * nsteps) / total_sec) / 1e6;

    // 4. Output Summary Report
    std::ostringstream title_ss;
    title_ss << "Landau Damping Verification (k = " << std::fixed << std::setprecision(3) << k << ")";

    pico::ui::VerificationReport report("Landau Damping Benchmark", res.passed, title_ss.str());

    report.add_row("Total Active Particles", std::to_string(total_particles));
    report.add_fixed_row("Execution Time", total_sec, 3, "s");
    report.add_fixed_row("Throughput", mup_s, 2, "MUP/s");

    report.add_pct_row("Max Energy Drift", res.max_energy_drift_pct, true);
    report.add_fixed_row("Measured Frequency (wr)", res.measured_freq, 4, "rad/s");
    report.add_fixed_row("Expected Frequency (wr)", res.expected_freq, 4, "rad/s");
    report.add_pct_row("Frequency Error", res.freq_error_pct, res.freq_error_pct <= 5.0);

    report.add_fixed_row("Measured Damping Rate (gamma)", res.measured_gamma, 5, "s^-1");
    report.add_fixed_row("Expected Damping Rate (gamma)", res.expected_gamma, 5, "s^-1");
    report.add_pct_row("Damping Rate Error", res.gamma_error_pct, res.passed);

    report.print();
    return report.passed() ? 0 : 1;
}

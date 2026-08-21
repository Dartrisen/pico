#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/LandauVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/PlaneWaveLaserInjector.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <chrono>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>

int main()
{
    // Tuning for k * lambda_D ~ 0.31 (Measurable Kinetic Damping)
    constexpr std::size_t grid_cells = 256;
    constexpr double      dx         = 0.1;
    constexpr double      dt         = 0.002;
    constexpr std::size_t ppc        = 4000;
    constexpr std::size_t nsteps     = 4000;
    constexpr std::size_t BS         = 64;
    constexpr float       target_n0  = 1.0f;

    constexpr float  v1   = 0.05f; // Velocity wave amplitude
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

    EngineT engine_instance{grid, ppc, target_n0};

    auto& particles = engine_instance.particles();

    const double L          = static_cast<double>(grid_cells) * dx;
    const double k          = 2.0 * std::numbers::pi / L;
    const double dx_p       = L / static_cast<double>(particles.active_particles());
    std::size_t  global_idx = 0;

    std::mt19937                    rng(42);
    std::normal_distribution<float> v_dist(0.0f, static_cast<float>(v_th));

    // Velocity perturbation: rho(t=0) = 0 is preserved, eliminating EM light-wave noise
    for (auto& block : particles)
    {
        for (std::size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            const double x0 = (static_cast<double>(global_idx) + 0.5) * dx_p;

            block.position_x[i] = static_cast<float>(x0);

            const float local_m = block.mass[i];
            const float v_wave  = v1 * std::sin(static_cast<float>(k * x0));

            block.momentum_x[i] = local_m * (v_dist(rng) + v_wave);
            block.momentum_y[i] = local_m * v_dist(rng);
            block.momentum_z[i] = local_m * v_dist(rng);
        }
    }

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    pico::diagnostics::LandauDampingVerifier verifier(dt, dx, k, v_th, target_n0);

    std::unique_ptr<IEngine> engine = std::move(wrapper);
    PICApp                   app(std::move(engine), dt);

    const auto start_wall_time = std::chrono::high_resolution_clock::now();

    app.run(nsteps,
            [&](int /*step*/)
            {
                const auto& eng = concrete_wrapper->engine();

                // Ex Field Energy
                double e_ex = 0.0;
                for (std::size_t i = 0; i < grid_cells; ++i)
                {
                    const float ex = eng.fields().E.field_x(i);
                    e_ex += 0.5 * static_cast<double>(ex * ex) * dx;
                }

                // Total EM Energy
                double e_field = 0.0;
                for (std::size_t i = 0; i < grid_cells; ++i)
                {
                    const float ex = eng.fields().E.field_x(i);
                    const float ey = eng.fields().E.field_y(i);
                    const float ez = eng.fields().E.field_z(i);
                    const float bx = eng.fields().B.field_x(i);
                    const float by = eng.fields().B.field_y(i);
                    const float bz = eng.fields().B.field_z(i);
                    e_field += 0.5 * static_cast<double>(ex * ex + ey * ey + ez * ez + bx * bx + by * by + bz * bz) * dx;
                }

                // Kinetic Energy
                double       e_kin  = 0.0;
                const double weight = 1.0 / static_cast<double>(ppc);
                for (const auto& block : eng.particles())
                {
                    for (std::size_t i = 0; i < block.activeCount; ++i)
                    {
                        const double px = block.momentum_x[i];
                        const double py = block.momentum_y[i];
                        const double pz = block.momentum_z[i];
                        const double m  = block.mass[i];
                        e_kin += 0.5 * (px * px + py * py + pz * pz) / m;
                    }
                }
                e_kin *= weight;

                verifier.record_step(e_ex, e_field + e_kin);
            });

    const auto   end_wall_time = std::chrono::high_resolution_clock::now();
    const double total_sec     = std::chrono::duration<double>(end_wall_time - start_wall_time).count();

    const auto res = verifier.verify(/*energy_drift_tol_pct=*/2.0, /*freq_tol_pct=*/5.0, /*gamma_tol_pct=*/15.0);

    const std::size_t total_particles = concrete_wrapper->engine().particles().active_particles();
    const double      mup_s           = ((static_cast<double>(total_particles) * nsteps) / total_sec) / 1e6;

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
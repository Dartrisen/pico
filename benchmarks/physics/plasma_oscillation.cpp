#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/PlasmaWaveVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <numbers>

int main()
{
    constexpr std::size_t grid_cells = 256;
    constexpr double      dx         = 0.1;
    constexpr double      dt         = 0.002;
    constexpr std::size_t ppc        = 100;
    constexpr std::size_t nsteps     = 1000;
    constexpr std::size_t BS         = 64;
    constexpr float       target_n0  = 2.0f; // Target density multiplier

    assert(dx > 0.95 * dt && "CFL condition violated.");

    Grid grid(grid_cells, dx);

    using Shape     = kernels::shapes::Shape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, BS>;

    // 1. Initialize Engine with custom density n0 = 2.0
    EngineT engine_instance{grid, ppc, target_n0};

    auto& particles = engine_instance.particles();

    const double L          = static_cast<double>(grid_cells) * dx;
    const double k          = 2.0 * std::numbers::pi / L;
    const float  v0         = 0.05f;
    const double dx_p       = L / static_cast<double>(particles.active_particles());
    std::size_t  global_idx = 0;

    // 2. Apply velocity perturbation (Preserve charge and mass initialized by engine)
    for (auto& block : particles)
    {
        for (std::size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            const double x0 = (static_cast<double>(global_idx) + 0.5) * dx_p;

            block.position_x[i] = static_cast<float>(x0);

            // p = m * v (where m is already scaled by target_n0 in init_density_constant)
            const float local_m = block.mass[i];
            block.momentum_x[i] = local_m * v0 * std::sin(static_cast<float>(k * x0));
            block.momentum_y[i] = 0.0f;
            block.momentum_z[i] = 0.0f;
        }
    }

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    // 3. Construct verifier with matching n0
    pico::diagnostics::PlasmaWaveVerifier verifier(dt, dx, ppc, target_n0);

    std::unique_ptr<IEngine> engine = std::move(wrapper);
    PICApp                   app(std::move(engine), dt);

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

                // Total Field Energy
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

                // Kinetic Energy (Scaled by macroparticle weight w = 1 / ppc)
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

    const auto res = verifier.verify(/*drift_tol=*/2.0, /*freq_tol=*/5.0);

    std::cout << "\n=== Automated Physics Verification (n0 = " << target_n0 << ") ===\n";
    std::cout << " Max Energy Drift : " << res.max_energy_drift_pct << " %\n";
    std::cout << " Measured Frequency: " << res.measured_freq << " rad/s\n";
    std::cout << " Expected Frequency: " << res.expected_freq << " rad/s\n";
    std::cout << " Frequency Error   : " << res.freq_error_pct << " %\n";
    std::cout << " Result            : " << (res.passed ? "✔ PASSED" : "❌ FAILED") << "\n\n";

    return res.passed ? 0 : 1;
}

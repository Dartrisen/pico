#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/CurrentVerifier.hpp"
#include "engine/modules/diagnostics/EnergyVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/PlaneWaveLaserInjector.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char** argv)
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

    EngineT engine_instance{grid, ppc, target_n0};

    auto&        particles  = engine_instance.particles();
    const double L          = static_cast<double>(grid_cells) * dx;
    const double dx_p       = L / static_cast<double>(particles.active_particles());
    std::size_t  global_idx = 0;

    for (auto& block : particles)
    {
        for (std::size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            const double x0 = (static_cast<double>(global_idx) + 0.5) * dx_p;

            block.position_x[i] = static_cast<float>(x0);

            const float m       = block.mass[i];
            block.momentum_x[i] = m * v_drift;
            block.momentum_y[i] = 0.0f;
            block.momentum_z[i] = 0.0f;
        }
    }

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    // Correct expected grid current density including 1 / dx
    const double expected_current = (-1.0 * static_cast<double>(target_n0) * static_cast<double>(v_drift)) / dx;

    pico::diagnostics::EnergyVerifier  energy_verifier(/*drift_tolerance_pct=*/2.0);
    pico::diagnostics::CurrentVerifier current_verifier(expected_current, /*tolerance_pct=*/5.0);

    std::unique_ptr<IEngine> engine = std::move(wrapper);
    PICApp                   app(std::move(engine), dt);

    app.run(nsteps,
            [&](int step)
            {
                auto& eng = concrete_wrapper->engine();

                // 1. Freeze electric field to maintain unperturbed ballistic drift
                for (std::size_t i = 0; i < grid_cells; ++i)
                {
                    eng.fields().E.field_x(i) = 0.0f;
                    eng.fields().E.field_y(i) = 0.0f;
                    eng.fields().E.field_z(i) = 0.0f;
                }

                // 2. Field Energy (zero under frozen field)
                double e_field = 0.0;

                // 3. Kinetic Energy
                double       e_kin  = 0.0;
                const double weight = static_cast<double>(target_n0) / static_cast<double>(ppc);

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

                energy_verifier.record_step(e_field, e_kin);

                // 4. Deposited Current Density & Uniformity Residual
                double sum_jx         = 0.0;
                double max_ampere_res = 0.0;

                for (std::size_t i = 0; i < grid_cells; ++i)
                {
                    const float jx_curr = eng.current().field_x(i);
                    sum_jx += static_cast<double>(jx_curr);

                    // Evaluate deposited current deviation relative to target density
                    const double local_res = std::abs(static_cast<double>(jx_curr) - expected_current);
                    max_ampere_res         = std::max(max_ampere_res, local_res);
                }

                const double avg_jx = sum_jx / static_cast<double>(grid_cells);
                current_verifier.record_step(avg_jx, max_ampere_res);
            });

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
#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/Thermalizing.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/EnergyVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/PlaneWaveLaserInjector.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
    // 1. Simulation Parameters
    constexpr std::size_t grid_cells = 256;
    constexpr double      dx         = 0.1;
    constexpr double      dt         = 0.002;
    constexpr std::size_t ppc        = 100;
    constexpr std::size_t nsteps     = 1000;
    constexpr std::size_t BS         = 64;

    assert(dx > 0.95 * dt && "CFL condition violated: dx must be greater than 0.95 * dt for stability.");

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

    EngineT engine_instance{grid, ppc};

    auto& particles = engine_instance.particles();

    const double L          = static_cast<double>(grid_cells) * dx;
    const double k          = 2.0 * M_PI / L;
    const float  v0         = 0.05f;
    const double dx_p       = L / static_cast<double>(particles.active_particles());
    std::size_t  global_idx = 0;

    for (auto& block : particles)
    {
        for (std::size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            const double x0 = (static_cast<double>(global_idx) + 0.5) * dx_p;

            block.position_x[i] = static_cast<float>(x0);

            // Momentum scaling with mass initialized by the engine
            const float m       = block.mass[i];
            block.momentum_x[i] = m * v0 * std::sin(static_cast<float>(k * x0));
            block.momentum_y[i] = m * v0 * std::cos(static_cast<float>(k * x0));
            block.momentum_z[i] = 0.0f;
        }
    }

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    // Energy Conservation Verifier (2.0% tolerance)
    pico::diagnostics::EnergyVerifier verifier(/*drift_tolerance_pct=*/2.0);

    std::unique_ptr<IEngine> engine = std::move(wrapper);
    PICApp                   app(std::move(engine), dt);

    app.run(nsteps,
            [&](int /*step*/)
            {
                const auto& eng = concrete_wrapper->engine();

                // 1. Calculate Electromagnetic Field Energy
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

                // 2. Calculate Kinetic Energy (weighted by macroparticle weight w = 1 / ppc)
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

                verifier.record_step(e_field, e_kin);
            });

    const auto res = verifier.verify();

    pico::ui::VerificationReport report("Energy Conservation Verification", res.passed);

    report.add_sci_row("Initial Energy", res.initial_energy, 4, "mc^2");
    report.add_sci_row("Final Energy", res.final_energy, 4, "mc^2");
    report.add_pct_row("Max Energy Drift", res.max_energy_drift_pct, res.passed);
    report.add_pct_row("Avg Energy Drift", res.avg_energy_drift_pct, true);

    report.print();
    return report.passed() ? 0 : 1;
}

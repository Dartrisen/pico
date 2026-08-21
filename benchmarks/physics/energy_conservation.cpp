#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
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
    constexpr double      dt         = 0.02;
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

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, BS>;

    EngineT engine_instance{grid, ppc};

    auto& particles = engine_instance.particles();
    particles.set_active(grid_cells * ppc);

    const double L          = grid_cells * dx;
    const double k          = 2.0 * M_PI / L;
    const float  v0         = 0.05f;
    const double dx_p       = L / static_cast<double>(particles.active_particles());
    std::size_t  global_idx = 0;

    for (auto& block : particles)
    {
        for (std::size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            const double x0 = (global_idx + 0.5) * dx_p;

            block.position_x[i] = static_cast<float>(x0);
            block.momentum_x[i] = v0 * std::sin(static_cast<float>(k * x0));
            block.momentum_y[i] = v0 * std::cos(static_cast<float>(k * x0));
            block.momentum_z[i] = 0.0f;
            block.charge[i]     = -1.0f;
            block.mass[i]       = 1.0f;
        }
    }

    std::unique_ptr<IEngine> engine = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));

    PICApp app(std::move(engine), dt);
    app.run(nsteps);

    return 0;
}
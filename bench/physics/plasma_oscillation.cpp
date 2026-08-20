#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cmath>
#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
    // 1. Simulation Parameters
    const size_t grid_cells = 256;
    const double dx         = 0.1;
    const double dt         = 0.02;
    const size_t ppc        = 100; // High PPC for low noise
    const size_t nsteps     = 500;

    Grid                  grid(grid_cells, dx);
    constexpr std::size_t BS = 64;

    using Shape  = kernels::shapes::Shape<1>;
    using Field  = pico::modules::field::YeeMaxwell<BS>;
    using Push   = pico::modules::pusher::BorisPusher<BS>;
    using Gather = pico::modules::gather::Gather<Shape, BS>;
    using Dep    = pico::modules::deposit::SimpleDeposit<Shape, BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BS>;

    EngineT engine_instance{grid, ppc};

    // 2. Initial Condition: Seed sinusoidal spatial perturbation x = x0 + alpha * cos(k * x0)
    auto& particles = engine_instance.particles();
    particles.set_active(grid_cells * ppc);

    const double L     = grid_cells * dx;
    const double k     = 2.0 * M_PI / L; // Fundamental mode k=1
    const float  alpha = 0.05f;          // Perturbation amplitude
    const float  v0    = 0.01f;          // Initial velocity amplitude

    size_t       global_idx = 0;
    const double dx_p       = L / static_cast<double>(particles.active_particles());

    for (auto& block : particles)
    {
        for (size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            double x0 = (global_idx + 0.5) * dx_p;

            // Perturb initial positions and momenta
            block.position_x[i] = static_cast<float>(x0 + alpha * std::sin(k * x0));
            block.momentum_x[i] = v0 * std::sin(k * x0);
            block.momentum_y[i] = 0.0f;
            block.momentum_z[i] = 0.0f;
            block.charge[i]     = -1.0f;
            block.mass[i]       = 1.0f;
        }
    }

    // 3. Diagnostics & Step Loop
    std::cout << "Step,Time,FieldEnergy_Ex\n";

    for (size_t step = 0; step < nsteps; ++step)
    {
        engine_instance.advance_impl(dt);

        // Compute total Ex field energy UE = 0.5 * sum(Ex^2) * dx
        double      field_energy = 0.0;
        const auto& fields       = engine_instance.fields();
        for (size_t i = 0; i < grid_cells; ++i)
        {
            float Ex = fields.E.field_x(i);
            field_energy += 0.5 * Ex * Ex * dx;
        }

        if (step % 5 == 0)
        {
            std::cout << step << "," << step * dt << "," << field_energy << "\n";
        }
    }

    return 0;
}
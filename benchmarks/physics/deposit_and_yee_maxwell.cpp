#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cmath>
#include <iostream>

int main()
{
    constexpr std::size_t grid_cells = 32;
    constexpr double      dx         = 0.1;
    constexpr double      dt         = 0.01;
    constexpr std::size_t ppc        = 1;
    constexpr std::size_t BS         = 64;

    Grid grid(grid_cells, dx);

    using Shape     = kernels::shapes::Shape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, BS>;

    EngineT engine{grid, ppc};

    // Set up a single particle in cell 5 moving right along X
    auto& particles = engine.particles();
    particles.set_active(1);

    for (auto& block : particles)
    {
        block.activeCount   = 1;
        block.position_x[0] = static_cast<float>(5.5 * dx); // Middle of cell 5
        block.momentum_x[0] = 1.0f;                         // Velocity v_x = 1.0
        block.momentum_y[0] = 0.0f;
        block.momentum_z[0] = 0.0f;
        block.charge[0]     = -1.0f;
        block.mass[0]       = 1.0f;
    }

    std::cout << "=== Testing Deposit & YeeMaxwell Coupling ===\n\n";

    // Advance 1 timestep (runs zero_out, gather, push, deposit, fold_currents, solve)
    engine.advance_impl(dt);

    // 1. Check Current Deposition in engine.current()
    double      max_jx  = 0.0;
    std::size_t jx_cell = 0;
    for (std::size_t i = 0; i < grid_cells; ++i)
    {
        const float jx = std::abs(engine.current().field_x(i));
        if (jx > max_jx)
        {
            max_jx  = jx;
            jx_cell = i;
        }
    }

    std::cout << "[Test 1: Deposit Check]\n";
    std::cout << "  Max |Jx| deposited: " << max_jx << " at cell " << jx_cell << "\n";
    if (max_jx < 1e-6)
    {
        std::cout << "  ❌ FAIL: Jx is 0.0! deposit_block is not writing into engine.current().\n\n";
    }
    else
    {
        std::cout << "  ✔ PASS: Jx correctly deposited into engine.current().\n\n";
    }

    // 2. Check Ex Field Update from Jx in engine.fields()
    double      max_ex  = 0.0;
    std::size_t ex_cell = 0;
    for (std::size_t i = 0; i < grid_cells; ++i)
    {
        const float ex = std::abs(engine.fields().E.field_x(i));
        if (ex > max_ex)
        {
            max_ex  = ex;
            ex_cell = i;
        }
    }

    std::cout << "[Test 2: YeeMaxwell Ex Update Check]\n";
    std::cout << "  Max |Ex| generated: " << max_ex << " at cell " << ex_cell << "\n";
    if (max_ex < 1e-6)
    {
        std::cout << "  ❌ FAIL: Ex remains 0.0! FieldSolver is not applying dEx/dt = -Jx.\n\n";
    }
    else
    {
        std::cout << "  ✔ PASS: Ex updated from Jx.\n\n";
    }

    // 3. Check Current Grid Reset between timesteps
    const double step1_jx = max_jx;
    engine.advance_impl(dt);

    double step2_max_jx = 0.0;
    for (std::size_t i = 0; i < grid_cells; ++i)
    {
        step2_max_jx = std::max(step2_max_jx, static_cast<double>(std::abs(engine.current().field_x(i))));
    }

    std::cout << "[Test 3: Current Grid Reset Check]\n";
    std::cout << "  Step 1 Max Jx: " << step1_jx << " | Step 2 Max Jx: " << step2_max_jx << "\n";
    if (step2_max_jx > 1.5 * step1_jx)
    {
        std::cout << "  ❌ FAIL: Jx is accumulating across steps!\n\n";
    }
    else
    {
        std::cout << "  ✔ PASS: current_.zero_out() properly resets currents every step.\n\n";
    }

    return 0;
}
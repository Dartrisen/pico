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
#include <iomanip>
#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
    // 1. Simulation Parameters
    const size_t grid_cells = 256;
    const double dx         = 0.1;
    const double dt         = 0.02;
    const size_t ppc        = 100;
    const size_t nsteps     = 1000;
    assert(dx > 0.95 * dt && "CFL condition violated: dx must be greater than 0.95 * dt for stability.");

    Grid                  grid(grid_cells, dx);
    constexpr std::size_t BS = 64;

    using Shape    = kernels::shapes::Shape<3>;
    using Field    = pico::modules::field::YeeMaxwell<BS>;
    using Push     = pico::modules::pusher::BorisPusher<BS>;
    using Gather   = pico::modules::gather::Gather<Shape, BS>;
    using Dep      = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using Boundary = pico::modules::boundary::PeriodicBoundaryHandler<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, Boundary, BS>;

    EngineT engine_instance{grid, ppc};

    // 2. Initial Condition: Seed coupled transverse and longitudinal velocity waves
    auto& particles = engine_instance.particles();
    particles.set_active(grid_cells * ppc);

    const double L  = grid_cells * dx;
    const double k  = 2.0 * M_PI / L;
    const float  v0 = 0.05f;

    size_t       global_idx = 0;
    const double dx_p       = L / static_cast<double>(particles.active_particles());

    for (auto& block : particles)
    {
        for (size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            double x0 = (global_idx + 0.5) * dx_p;

            block.position_x[i] = static_cast<float>(x0);
            block.momentum_x[i] = v0 * std::sin(k * x0);
            block.momentum_y[i] = v0 * std::cos(k * x0);
            block.momentum_z[i] = 0.0f;
            block.charge[i]     = -1.0f;
            block.mass[i]       = 1.0f;
        }
    }

    // 3. Output Header
    std::cout << "Step,Time,E_Kinetic,E_Field_E,E_Field_B,E_Total,RelError\n";

    double E_total_0 = 0.0;
    // Calculate macro-particle weight for diagnostic normalization
    const double w_p = dx / static_cast<double>(ppc);

    for (size_t step = 0; step < nsteps; ++step)
    {
        // Compute Particle Kinetic Energy: Sum( 0.5 * m * v^2 )
        double E_kinetic = 0.0;
        for (const auto& block : particles)
        {
            for (size_t i = 0; i < block.activeCount; ++i)
            {
                float px = block.momentum_x[i];
                float py = block.momentum_y[i];
                float pz = block.momentum_z[i];
                float m  = block.mass[i];
                E_kinetic += 0.5 * (px * px + py * py + pz * pz) / m * w_p;
            }
        }

        // Compute EM Field Energies: 0.5 * Integral( E^2 + B^2 ) dx
        double      E_field_E = 0.0;
        double      E_field_B = 0.0;
        const auto& fields    = engine_instance.fields();

        for (size_t i = 0; i < grid_cells; ++i)
        {
            float Ex = fields.E.field_x(i);
            float Ey = fields.E.field_y(i);
            float Ez = fields.E.field_z(i);

            float Bx = fields.B.field_x(i);
            float By = fields.B.field_y(i);
            float Bz = fields.B.field_z(i);

            E_field_E += 0.5 * (Ex * Ex + Ey * Ey + Ez * Ez) * dx;
            E_field_B += 0.5 * (Bx * Bx + By * By + Bz * Bz) * dx;
        }

        double E_total = E_kinetic + E_field_E + E_field_B;

        if (step == 0)
        {
            E_total_0 = E_total;
        }

        double rel_error = std::abs(E_total - E_total_0) / E_total_0;

        if (step % 20 == 0)
        {
            std::cout << step << "," << step * dt << "," << E_kinetic << "," << E_field_E << "," << E_field_B << ","
                      << E_total << "," << rel_error << "\n";
        }

        engine_instance.advance_impl(dt);
    }

    return 0;
}
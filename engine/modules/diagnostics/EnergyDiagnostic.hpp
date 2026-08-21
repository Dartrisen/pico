#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>

namespace pico::diagnostics
{

template <class EngineT>
class EnergyDiagnostic
{
private:
    double initial_energy_{-1.0};

public:
    void log(const EngineT& engine, std::size_t step, double time)
    {
        const auto&       fields     = engine.fields();
        const auto&       particles  = engine.particles();
        const double      dx         = fields.E.grid().cell_size();
        const std::size_t grid_cells = fields.E.grid().physical_size();

        const double w_p = dx / static_cast<double>(engine.particles_per_cell());

        // 1. Kinetic Energy
        double E_kinetic = 0.0;
        for (const auto& block : particles)
        {
            for (std::size_t i = 0; i < block.activeCount; ++i)
            {
                const float px = block.momentum_x[i];
                const float py = block.momentum_y[i];
                const float pz = block.momentum_z[i];
                const float m  = block.mass[i];
                E_kinetic += 0.5 * (px * px + py * py + pz * pz) / m * w_p;
            }
        }

        // 2. Field Energy
        double E_field_E = 0.0;
        double E_field_B = 0.0;
        for (std::size_t i = 0; i < grid_cells; ++i)
        {
            const float Ex = fields.E.field_x(i);
            const float Ey = fields.E.field_y(i);
            const float Ez = fields.E.field_z(i);

            const float Bx = fields.B.field_x(i);
            const float By = fields.B.field_y(i);
            const float Bz = fields.B.field_z(i);

            E_field_E += 0.5 * (Ex * Ex + Ey * Ey + Ez * Ez) * dx;
            E_field_B += 0.5 * (Bx * Bx + By * By + Bz * Bz) * dx;
        }

        const double E_total = E_kinetic + E_field_E + E_field_B;
        if (initial_energy_ < 0.0)
        {
            initial_energy_ = E_total;
        }

        const double rel_error = std::abs(E_total - initial_energy_) / initial_energy_;

        // Console output formatted into aligned table columns
        std::cout << std::left << std::setw(8) << step << std::setw(12) << std::fixed << std::setprecision(4) << time
                  << std::scientific << std::setprecision(4) << std::setw(14) << E_kinetic << std::setw(14) << E_field_E
                  << std::setw(14) << E_field_B << std::setw(14) << E_total << std::setw(12) << rel_error
                  << std::defaultfloat << "\n";
    }
};

} // namespace pico::diagnostics
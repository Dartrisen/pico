#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace pico::diagnostics
{

struct EnergyMetrics
{
    double e_kin{0.0};
    double e_ex{0.0};
    double e_ey{0.0};
    double e_ez{0.0};
    double e_bx{0.0};
    double e_by{0.0};
    double e_bz{0.0};

    [[nodiscard]] constexpr double e_field_electric() const noexcept
    {
        return e_ex + e_ey + e_ez;
    }
    [[nodiscard]] constexpr double e_field_magnetic() const noexcept
    {
        return e_bx + e_by + e_bz;
    }
    [[nodiscard]] constexpr double e_field_total() const noexcept
    {
        return e_field_electric() + e_field_magnetic();
    }
    [[nodiscard]] constexpr double e_total() const noexcept
    {
        return e_kin + e_field_total();
    }
};

template <class EngineT>
class EnergyDiagnostic
{
private:
    double initial_energy_{-1.0};

public:
    /// Single-pass diagnostic computation across fields and active particles
    [[nodiscard]] static EnergyMetrics evaluate(const EngineT& engine)
    {
        EnergyMetrics metrics{};
        const auto&   fields    = engine.fields();
        const auto&   particles = engine.particles();

        const double      dx         = fields.E.grid().cell_size();
        const std::size_t grid_cells = fields.E.grid().physical_size();
        const double      ppc_weight = 1.0 / static_cast<double>(engine.particles_per_cell());

        // 1. Single-pass field energy calculation across all 6 field components
        for (std::size_t i = 0; i < grid_cells; ++i)
        {
            const std::size_t buf_i = fields.E.grid().physical_to_buffer(i);

            const double ex = fields.E.field_x(buf_i);
            const double ey = fields.E.field_y(buf_i);
            const double ez = fields.E.field_z(buf_i);
            const double bx = fields.B.field_x(buf_i);
            const double by = fields.B.field_y(buf_i);
            const double bz = fields.B.field_z(buf_i);

            metrics.e_ex += 0.5 * ex * ex * dx;
            metrics.e_ey += 0.5 * ey * ey * dx;
            metrics.e_ez += 0.5 * ez * ez * dx;
            metrics.e_bx += 0.5 * bx * bx * dx;
            metrics.e_by += 0.5 * by * by * dx;
            metrics.e_bz += 0.5 * bz * bz * dx;
        }

        // 2. Particle Kinetic Energy calculation
        for (const auto& block : particles)
        {
            for (std::size_t i = 0; i < block.activeCount; ++i)
            {
                const double px = block.momentum_x[i];
                const double py = block.momentum_y[i];
                const double pz = block.momentum_z[i];
                const double m  = block.mass[i];
                metrics.e_kin += 0.5 * (px * px + py * py + pz * pz) / m;
            }
        }
        metrics.e_kin *= ppc_weight;

        return metrics;
    }

    /// Logs current energy snapshot in tabular alignment
    void log(const EngineT& engine, std::size_t step, double time, std::ostream& os = std::cout)
    {
        const EnergyMetrics m     = evaluate(engine);
        const double        total = m.e_total();

        if (initial_energy_ < 0.0)
        {
            initial_energy_ = total;
        }

        const double rel_error = std::abs(total - initial_energy_) / initial_energy_;

        os << std::left << std::setw(8) << step << std::fixed << std::setw(12) << std::setprecision(4) << time << std::scientific << std::setprecision(4) << std::setw(14)
           << m.e_kin << std::setw(14) << m.e_field_electric() << std::setw(14) << m.e_field_magnetic() << std::setw(14) << total << std::setw(12) << rel_error << std::defaultfloat
           << "\n";
    }

    static void print_header(std::ostream& os = std::cout)
    {
        os << std::left << std::setw(8) << "Step" << std::setw(12) << "Time" << std::setw(14) << "E_Kinetic" << std::setw(14) << "E_Electric" << std::setw(14) << "E_Magnetic"
           << std::setw(14) << "E_Total" << std::setw(12) << "Rel_Err" << "\n";
    }
};

} // namespace pico::diagnostics

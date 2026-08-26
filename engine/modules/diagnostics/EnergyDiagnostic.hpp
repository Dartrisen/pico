#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace pico::diagnostics
{

struct EnergyMetrics
{
    std::vector<double> e_kin_species{};
    double              e_ex{0.0};
    double              e_ey{0.0};
    double              e_ez{0.0};
    double              e_bx{0.0};
    double              e_by{0.0};
    double              e_bz{0.0};

    [[nodiscard]] double e_kin_total() const noexcept
    {
        return std::accumulate(e_kin_species.begin(), e_kin_species.end(), 0.0);
    }
    [[nodiscard]] double e_kin() const noexcept
    {
        return e_kin_total();
    }
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
    [[nodiscard]] double e_total() const noexcept
    {
        return e_kin_total() + e_field_total();
    }
};

template <class EngineT>
class EnergyDiagnostic
{
private:
    double initial_energy_{-1.0};

public:
    /// Single-pass diagnostic computation across electromagnetic fields and per-species particles
    [[nodiscard]] static EnergyMetrics evaluate(const EngineT& engine)
    {
        EnergyMetrics     metrics{};
        const auto&       fields     = engine.fields();
        const auto&       grid       = fields.E.grid();
        const double      dx         = grid.cell_size();
        const std::size_t grid_cells = grid.physical_size();

        // 1. Single-pass field energy calculation across all 6 field components
        for (std::size_t i = 0; i < grid_cells; ++i)
        {
            const std::size_t buf_i = grid.physical_to_buffer(i);

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

        // 2. Per-Species Particle Kinetic Energy calculation
        const auto& species_list = engine.species();
        metrics.e_kin_species.reserve(species_list.size());

        for (const auto& species : species_list)
        {
            if (species.active_particles() == 0)
            {
                metrics.e_kin_species.push_back(0.0);
                continue;
            }

            const double inv_m0      = 1.0 / static_cast<double>(species.base_mass());
            const double species_ppc = static_cast<double>(species.active_particles()) / static_cast<double>(grid_cells);
            const double ppc_weight  = (species_ppc > 0.0) ? (1.0 / species_ppc) : 1.0;

            double species_e_kin = 0.0;
            for (const auto& block : species)
            {
                for (std::size_t i = 0; i < block.activeCount; ++i)
                {
                    const double px = block.momentum_x[i];
                    const double py = block.momentum_y[i];
                    const double pz = block.momentum_z[i];
                    const double w  = block.weight[i];

                    species_e_kin += 0.5 * w * (px * px + py * py + pz * pz) * inv_m0;
                }
            }
            metrics.e_kin_species.push_back(species_e_kin * ppc_weight);
        }

        return metrics;
    }

    /// Logs current energy snapshot with dynamic per-species kinetic columns
    void log(const EngineT& engine, std::size_t step, double time, std::ostream& os = std::cout)
    {
        const EnergyMetrics m     = evaluate(engine);
        const double        total = m.e_total();

        if (initial_energy_ < 0.0)
        {
            initial_energy_ = total;
        }

        const double rel_error = (initial_energy_ > 0.0) ? std::abs(total - initial_energy_) / initial_energy_ : 0.0;

        os << std::left << std::setw(8) << step << std::fixed << std::setw(12) << std::setprecision(4) << time << std::scientific << std::setprecision(4);

        for (const double species_kin : m.e_kin_species)
        {
            os << std::setw(14) << species_kin;
        }

        os << std::setw(14) << m.e_kin_total() << std::setw(14) << m.e_field_electric() << std::setw(14) << m.e_field_magnetic() << std::setw(14) << total << std::setw(12)
           << rel_error << std::defaultfloat << "\n";
    }

    static void print_header(const EngineT& engine, std::ostream& os = std::cout)
    {
        print_header(engine.species().size(), os);
    }

    static void print_header(std::size_t num_species, std::ostream& os = std::cout)
    {
        os << std::left << std::setw(8) << "Step" << std::setw(12) << "Time";

        for (std::size_t s = 0; s < num_species; ++s)
        {
            os << std::setw(14) << ("E_Kin_S" + std::to_string(s));
        }

        os << std::setw(14) << "E_Kin_Tot" << std::setw(14) << "E_Electric" << std::setw(14) << "E_Magnetic" << std::setw(14) << "E_Total" << std::setw(12) << "Rel_Err" << "\n";
    }
};

} // namespace pico::diagnostics
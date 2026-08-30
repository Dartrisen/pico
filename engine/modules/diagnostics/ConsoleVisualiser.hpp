#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace pico::diagnostics
{

template <class EngineT>
class ConsoleVisualizer
{
public:
    /// Renders a Unicode sparkline of charge density n(x) for a single species index
    static void draw_species_density_sparkline(const EngineT& engine, std::size_t species_idx, std::size_t display_width = 64)
    {
        const auto& species_list = engine.species();
        if (species_idx >= species_list.size())
        {
            return;
        }

        const auto&       species = species_list[species_idx];
        const auto&       grid    = engine.fields().E.grid();
        const std::size_t n_cells = grid.physical_size();

        if (species.active_particles() == 0 || n_cells == 0)
        {
            return;
        }

        const double inv_dx      = 1.0 / grid.cell_size();
        const double abs_q0      = std::abs(static_cast<double>(species.base_charge()));
        const double species_ppc = static_cast<double>(species.active_particles()) / static_cast<double>(n_cells);
        const double ppc_weight  = (species_ppc > 0.0) ? (1.0 / species_ppc) : 1.0;

        // 1. Accumulate particle charge density onto physical grid cells for this species
        std::vector<double> density(n_cells, 0.0);
        for (const auto& block : species)
        {
            for (std::size_t i = 0; i < block.activeCount; ++i)
            {
                const double x = block.position_x[i];
                if (x < 0.0)
                {
                    continue;
                }

                const std::size_t cell = static_cast<std::size_t>(x * inv_dx);
                if (cell < n_cells)
                {
                    density[cell] += abs_q0 * block.weight[i];
                }
            }
        }

        // Scale total cell charge sum by species PPC to convert to normalized density
        for (std::size_t c = 0; c < n_cells; ++c)
        {
            density[c] *= ppc_weight;
        }

        // 2. Downsample grid cells to fit terminal display width
        std::vector<double> bins(display_width, 0.0);
        const double        cells_per_bin = static_cast<double>(n_cells) / display_width;

        double max_n = 0.001; // Guard against divide-by-zero
        for (std::size_t b = 0; b < display_width; ++b)
        {
            const std::size_t start_c = static_cast<std::size_t>(b * cells_per_bin);
            std::size_t       end_c   = static_cast<std::size_t>((b + 1) * cells_per_bin);
            end_c                     = std::min(end_c, n_cells);

            for (std::size_t c = start_c; c < end_c; ++c)
            {
                bins[b] += density[c];
            }
            bins[b] /= (end_c > start_c ? (end_c - start_c) : 1);
            max_n = std::max(max_n, bins[b]);
        }

        // 3. Render Unicode sparkline with ANSI color gradient
        static constexpr std::string_view blocks[] = {" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"};

        std::cout << "\033[36m[Species " << species_idx << " n(x)] |" << "\033[0m";
        for (double val : bins)
        {
            const double      norm = std::clamp(val / max_n, 0.0, 1.0);
            const std::size_t idx  = static_cast<std::size_t>(norm * 8.0);

            if (norm > 0.7)
            {
                std::cout << "\033[31m"; // High density: Red
            }
            else if (norm > 0.3)
            {
                std::cout << "\033[33m"; // Mid density: Yellow
            }
            else
            {
                std::cout << "\033[36m"; // Low density: Cyan
            }

            std::cout << blocks[idx];
        }
        std::cout << "\033[0m| Max: " << std::fixed << std::setprecision(2) << max_n << " n_c\n";
    }

    /// Renders stacked density sparklines sequentially for all species
    static void draw_density_sparkline(const EngineT& engine, std::size_t display_width = 64)
    {
        for (std::size_t s = 0; s < engine.species().size(); ++s)
        {
            draw_species_density_sparkline(engine, s, display_width);
        }
    }
};

} // namespace pico::diagnostics
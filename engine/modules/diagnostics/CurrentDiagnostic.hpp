#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pico::diagnostics
{

struct CurrentMetrics
{
    double avg_jx{0.0};
    double avg_jy{0.0};
    double avg_jz{0.0};
    double max_ampere_res{0.0};
};

template <class EngineT>
class CurrentDiagnostic
{
public:
    /// Single-pass diagnostic reduction across all current density components
    [[nodiscard]] static CurrentMetrics evaluate(const EngineT& engine, double expected_jx = 0.0)
    {
        CurrentMetrics    metrics{};
        const auto&       J     = engine.current();
        const auto&       grid  = J.grid();
        const std::size_t cells = grid.physical_size();

        double sum_jx  = 0.0;
        double sum_jy  = 0.0;
        double sum_jz  = 0.0;
        double max_res = 0.0;

        for (std::size_t i = 0; i < cells; ++i)
        {
            const std::size_t buf_i = grid.physical_to_buffer(i);

            const double jx = static_cast<double>(J.field_x(buf_i));
            const double jy = static_cast<double>(J.field_y(buf_i));
            const double jz = static_cast<double>(J.field_z(buf_i));

            sum_jx += jx;
            sum_jy += jy;
            sum_jz += jz;

            const double res = std::abs(jx - expected_jx);
            max_res          = std::max(max_res, res);
        }

        const double inv_cells = 1.0 / static_cast<double>(cells);
        metrics.avg_jx         = sum_jx * inv_cells;
        metrics.avg_jy         = sum_jy * inv_cells;
        metrics.avg_jz         = sum_jz * inv_cells;
        metrics.max_ampere_res = max_res;

        return metrics;
    }
};

} // namespace pico::diagnostics
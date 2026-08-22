#pragma once

#include <algorithm>
#include <vector>

namespace pico::diagnostics
{

struct SpatialEnvelope
{
    double              total_transverse_energy{0.0};
    double              peak_x{0.0};
    std::vector<double> energy_density;
};

template <class EngineT>
class FieldEnvelopeDiagnostic
{
public:
    [[nodiscard]] static SpatialEnvelope evaluate(const EngineT& engine, int smoothing_window = 0)
    {
        SpatialEnvelope   result{};
        const auto&       fields = engine.fields();
        const auto&       grid   = fields.E.grid();
        const std::size_t cells  = grid.physical_size();
        const double      dx     = grid.cell_size();

        result.energy_density.resize(cells, 0.0);

        for (std::size_t i = 0; i < cells; ++i)
        {
            const std::size_t buf_i = grid.physical_to_buffer(i);
            const double      ey    = fields.E.field_y(buf_i);
            const double      bz    = fields.B.field_z(buf_i);

            const double u           = 0.5 * (ey * ey + bz * bz);
            result.energy_density[i] = u;
            result.total_transverse_energy += u * dx;
        }

        if (smoothing_window > 0 && cells > static_cast<std::size_t>(2 * smoothing_window))
        {
            double      max_density = 0.0;
            std::size_t peak_idx    = 0;

            for (std::size_t i = smoothing_window; i < cells - smoothing_window; ++i)
            {
                double sum = 0.0;
                for (int w = -smoothing_window; w <= smoothing_window; ++w)
                {
                    sum += result.energy_density[static_cast<std::size_t>(static_cast<int>(i) + w)];
                }
                if (sum > max_density)
                {
                    max_density = sum;
                    peak_idx    = i;
                }
            }
            result.peak_x = static_cast<double>(peak_idx) * dx;
        }

        return result;
    }
};

} // namespace pico::diagnostics

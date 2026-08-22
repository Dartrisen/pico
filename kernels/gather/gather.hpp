#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_block.hpp"

#include <algorithm>
#include <cstddef>

namespace kernels::gather
{

template <class Shape, std::size_t BLOCK_SIZE>
struct FieldGather
{
    static void gather(const particle::ParticleBlock<BLOCK_SIZE>& pb, const EMFields<BLOCK_SIZE>& fields, const Grid& grid, FieldScratch<BLOCK_SIZE>& scratch)
    {
        constexpr int S = Shape::S;

        const double dx     = grid.cell_size();
        const int    guards = static_cast<int>(grid.guard_cells());
        const double max_x  = (static_cast<double>(grid.physical_size()) - 1e-7) * dx;

        for (std::size_t p = 0; p < pb.activeCount; ++p)
        {
            // Clamp x safely inside [0, physical_domain) to guard against float precision overflow
            const double x = std::clamp(static_cast<double>(pb.position_x[p]), 0.0, max_x);

            int    i0_node = 0, i0_half = 0;
            double w_node_d[S]{}, w_half_d[S]{};

            Shape::weights(x, dx, i0_node, w_node_d);
            Shape::weights(x - 0.5 * dx, dx, i0_half, w_half_d);

            // Safe signed index arithmetic before casting to std::size_t
            const int idx_half = i0_half + guards;
            const int idx_node = i0_node + guards;

            const std::size_t half_start = static_cast<std::size_t>(std::max(0, idx_half));
            const std::size_t node_start = static_cast<std::size_t>(std::max(0, idx_node));

            float ex = 0.0f, by = 0.0f, bz = 0.0f;
            for (int s = 0; s < S; ++s)
            {
                const float w = static_cast<float>(w_half_d[s]);
                ex += w * fields.E.field_x(half_start + s);
                by += w * fields.B.field_y(half_start + s);
                bz += w * fields.B.field_z(half_start + s);
            }

            float ey = 0.0f, ez = 0.0f, bx = 0.0f;
            for (int s = 0; s < S; ++s)
            {
                const float w = static_cast<float>(w_node_d[s]);
                ey += w * fields.E.field_y(node_start + s);
                ez += w * fields.E.field_z(node_start + s);
                bx += w * fields.B.field_x(node_start + s);
            }

            scratch.Ex[p] = ex;
            scratch.Ey[p] = ey;
            scratch.Ez[p] = ez;
            scratch.Bx[p] = bx;
            scratch.By[p] = by;
            scratch.Bz[p] = bz;
        }
    }
};

} // namespace kernels::gather
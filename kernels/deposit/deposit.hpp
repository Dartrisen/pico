#pragma once

#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"

namespace kernels::deposit
{

template <class Shape, std::size_t BLOCK_SIZE>
struct CurrentDeposit
{
    static void deposit(const particle::ParticleBlock<BLOCK_SIZE>& pb, FieldSystem<BLOCK_SIZE>& J, const Grid& grid, float dt, float ppc)
    {
        constexpr int S = Shape::S;

        const double dx           = grid.cell_size();
        const float  inv_dx       = static_cast<float>(1.0 / dx);
        const float  scale_factor = (1.0f / ppc) * inv_dx;
        const int    guards       = static_cast<int>(grid.guard_cells());

        for (std::size_t p = 0; p < pb.activeCount; ++p)
        {
            const double x        = pb.position_x[p];
            const float  inv_m    = 1.0f / pb.mass[p];
            const float  q_factor = pb.charge[p] * scale_factor;

            const float qvx = q_factor * (pb.momentum_x[p] * inv_m);
            const float qvy = q_factor * (pb.momentum_y[p] * inv_m);
            const float qvz = q_factor * (pb.momentum_z[p] * inv_m);

            int    i0_node = 0, i0_half = 0;
            double w_node_d[S]{}, w_half_d[S]{};

            // Call generic shape weights evaluation
            Shape::weights(x, dx, i0_node, w_node_d);
            Shape::weights(x - 0.5 * dx, dx, i0_half, w_half_d);

            const std::size_t half_start = static_cast<std::size_t>(i0_half + guards);
            const std::size_t node_start = static_cast<std::size_t>(i0_node + guards);

            for (int s = 0; s < S; ++s)
            {
                J.field_x(half_start + s) += static_cast<float>(w_half_d[s]) * qvx;
            }

            for (int s = 0; s < S; ++s)
            {
                const float w = static_cast<float>(w_node_d[s]);
                J.field_y(node_start + s) += w * qvy;
                J.field_z(node_start + s) += w * qvz;
            }
        }
    }
};

} // namespace kernels::deposit
#pragma once

#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"

#include <algorithm>
#include <cstddef>

namespace kernels::deposit
{

template <class Shape, std::size_t BLOCK_SIZE>
struct CurrentDeposit
{
    static void deposit(const particle::ParticleBlock<BLOCK_SIZE>& pb, FieldSystem<BLOCK_SIZE>& J, const Grid& grid, [[maybe_unused]] float dt, float ppc, float base_charge)
    {
        constexpr int S = Shape::S;

        const double dx     = grid.cell_size();
        const float  inv_dx = static_cast<float>(1.0 / dx);

        const float base_scale = (base_charge / ppc) * inv_dx;

        const int    guards = static_cast<int>(grid.guard_cells());
        const double max_x  = (static_cast<double>(grid.physical_size()) - 1e-7) * dx;

        for (std::size_t p = 0; p < pb.activeCount; ++p)
        {
            const double x         = std::clamp(static_cast<double>(pb.position_x[p]), 0.0, max_x);
            const float  inv_gamma = pb.inv_gamma[p];
            const float  q_factor  = pb.weight[p] * base_scale;

            // Physical current flux: q * v = q * (u * inv_gamma)
            const float qvx = q_factor * (pb.momentum_x[p] * inv_gamma);
            const float qvy = q_factor * (pb.momentum_y[p] * inv_gamma);
            const float qvz = q_factor * (pb.momentum_z[p] * inv_gamma);

            int    i0_node = 0, i0_half = 0;
            double w_node_d[S]{}, w_half_d[S]{};

            Shape::weights(x, dx, i0_node, w_node_d);
            Shape::weights(x - 0.5 * dx, dx, i0_half, w_half_d);

            const int idx_half = i0_half + guards;
            const int idx_node = i0_node + guards;

            const std::size_t half_start = static_cast<std::size_t>(std::max(0, idx_half));
            const std::size_t node_start = static_cast<std::size_t>(std::max(0, idx_node));

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
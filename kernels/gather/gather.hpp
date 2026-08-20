#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_block.hpp"

namespace kernels::gather
{
    template <class Shape, std::size_t BLOCK_SIZE>
    struct FieldGather
    {
    private:
        template <::field::FieldComp Component>
        static float interpolate_component(const FieldSystem<BLOCK_SIZE>& field_system, int start_idx,
                                           const float (&w)[Shape::S])
        {
            float result = 0.0f;
            for (int s = 0; s < Shape::S; ++s)
            {
                result += w[s] * field_system.template field<Component>(start_idx + s);
            }
            return result;
        }

    public:
        static void gather(const particle::ParticleBlock<BLOCK_SIZE>& particle_block,
                           const EMFields<BLOCK_SIZE>& fields, const Grid& grid, FieldScratch<BLOCK_SIZE>& scratch)
        {
            constexpr int S = Shape::S;

            const double dx     = grid.cell_size();
            const int    guards = static_cast<int>(grid.guard_cells());

            for (std::size_t p = 0; p < particle_block.activeCount; ++p)
            {
                const double x = particle_block.position_x[p];

                // 1. Double-precision shape evaluations
                int    i0_node = 0, i0_half = 0;
                double w_node_d[S]{}, w_half_d[S]{};

                Shape::weights(x, dx, i0_node, w_node_d);
                Shape::weights(x - 0.5 * dx, dx, i0_half, w_half_d);

                // 2. Convert weights to float ONCE for fast SIMD gathering
                float w_node[S], w_half[S];
                for (int s = 0; s < S; ++s)
                {
                    w_node[s] = static_cast<float>(w_node_d[s]);
                    w_half[s] = static_cast<float>(w_half_d[s]);
                }

                const int node_start = i0_node + guards;
                const int half_start = i0_half + guards;

                // 3. Clean stencil accumulation
                scratch.Ex[p] = interpolate_component<::field::FieldComp::X>(fields.E, half_start, w_half);
                scratch.By[p] = interpolate_component<::field::FieldComp::Y>(fields.B, half_start, w_half);
                scratch.Bz[p] = interpolate_component<::field::FieldComp::Z>(fields.B, half_start, w_half);

                scratch.Ey[p] = interpolate_component<::field::FieldComp::Y>(fields.E, node_start, w_node);
                scratch.Ez[p] = interpolate_component<::field::FieldComp::Z>(fields.E, node_start, w_node);
                scratch.Bx[p] = interpolate_component<::field::FieldComp::X>(fields.B, node_start, w_node);
            }
        }
    };
} // namespace kernels::gather

#pragma once
#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"

#include <iostream>

namespace kernels::deposit
{

    template <class Shape, size_t BLOCK_SIZE>
    struct CurrentDeposit
    {
        static inline void deposit(const particle::ParticleBlock<BLOCK_SIZE>& pb, FieldSystem<BLOCK_SIZE>& J,
                                   const Grid& grid, double dt, uint32_t n_ppc)
        {
            for (size_t p = 0; p < pb.activeCount; ++p)
            {
                int    i0;
                double w[Shape::S];

                // 1. Calculate shape weights for particle position
                Shape::weights(pb.position_x[p], grid.cell_size(), i0, w);

                float qvx = pb.charge[p] * pb.momentum_x[p];
                float qvy = pb.charge[p] * pb.momentum_y[p];
                float qvz = pb.charge[p] * pb.momentum_z[p];

                // 2. Deposit current across the shape support stencil
                const float inv_ppc = 1.0f / n_ppc; // Normalization factor for current deposition
                for (int s = 0; s < Shape::S; ++s)
                {
                    int idx = i0 + s;
                    if (idx < 0 || idx >= static_cast<int>(grid.size()))
                        continue;

                    float ws = static_cast<float>(w[s]) * inv_ppc;
                    J.field_x(idx) += ws * qvx;
                    J.field_y(idx) += ws * qvy;
                    J.field_z(idx) += ws * qvz;
                }
            }
        }
    };

} // namespace kernels::deposit
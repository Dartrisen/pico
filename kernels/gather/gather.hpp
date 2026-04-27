#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"
#include "kernels/shapes/spline.hpp"

namespace kernels::gather
{

    template <class Shape, size_t BLOCK_SIZE>
    struct FieldGather
    {
        static inline void gather(
            const particle::ParticleBlock<BLOCK_SIZE> &pb,
            const EMFields<BLOCK_SIZE> &fields,
            const Grid &grid,
            FieldScratch<BLOCK_SIZE> &scratch)
        {
            scratch.clear();

            constexpr int S = Shape::S;
            static_assert(S > 0 && S <= 8, "shape support must be reasonable");

            for (size_t p = 0; p < pb.activeCount; ++p)
            {
                int i0;
                double w[S];

                Shape::weights(pb.position_x[p], grid.cell_size(), i0, w);

                scratch.Ex[p] = 0.0f;
                scratch.Bx[p] = 0.0f;

                for (int s = 0; s < S; ++s)
                {
                    int idx = i0 + s;
                    if (idx < 0 || idx >= static_cast<int>(grid.size()))
                        continue;

                    float weight = static_cast<float>(w[s]);
                    scratch.Ex[p] += weight * fields.E.field_x(idx);
                    scratch.Bx[p] += weight * fields.B.field_x(idx);
                }
            }
        }
    };

} // namespace kernels::gather

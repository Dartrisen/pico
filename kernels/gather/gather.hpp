#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"
#include "kernels/shapes/spline.hpp"

namespace kernels::gather
{

    template <class Shape, size_t BLOCK_SIZE>
    struct FieldGather
    {
        static inline void gather(const particle::ParticleBlock<BLOCK_SIZE>& pb, const EMFields<BLOCK_SIZE>& fields,
                                  const Grid& grid, FieldScratch<BLOCK_SIZE>& scratch)
        {
            constexpr int S = Shape::S;
            static_assert(S > 0 && S <= 8);

            const float dx        = grid.cell_size();
            const int   grid_size = static_cast<int>(grid.size());

            for (size_t p = 0; p < pb.activeCount; ++p)
            {
                int    i0;
                double w[S];

                Shape::weights(pb.position_x[p], dx, i0, w);

                // register accumulators
                float ex = 0.f;
                float bx = 0.f;

                const auto& E = fields.E;
                const auto& B = fields.B;

                for (int s = 0; s < S; ++s)
                {
                    const int idx = i0 + s;

                    if ((unsigned) idx >= (unsigned) grid_size)
                        continue;

                    const float weight = static_cast<float>(w[s]);

                    ex += weight * E.field_x(idx);
                    bx += weight * B.field_x(idx);
                }

                scratch.Ex[p] = ex;
                scratch.Bx[p] = bx;
            }
        }
    };

} // namespace kernels::gather
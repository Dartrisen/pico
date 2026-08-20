#pragma once

#include "data/field/include/field_em.hpp"
#include "data/grid/include/grid.hpp"
#include "spline.hpp"

namespace kernels::shapes
{

    struct Vec3
    {
        double x, y, z;
    };

    // 1D-oriented field grid support; grow later into FieldGrid3D
    using FieldGrid = Grid;

    template <int Order>
    struct Shape
    {
        static constexpr int S = SplineTraits<Order>::support;

        template <size_t BLOCK_SIZE>
        static inline void gather1d(const particle::ParticleBlock<BLOCK_SIZE>& pb, const EMFields<BLOCK_SIZE>& fields,
                                    const Grid& grid, FieldScratch<BLOCK_SIZE>& scratch)
        {
            for (size_t p = 0; p < pb.activeCount; ++p)
            {
                int    i0;
                double w[S];
                SplineShape<Order>::weights(pb.position_x[p], grid.cell_size(), i0, w);

                scratch.Ex[p] = 0.0f;
                scratch.Ey[p] = 0.0f;
                scratch.Ez[p] = 0.0f;
                scratch.Bx[p] = 0.0f;
                scratch.By[p] = 0.0f;
                scratch.Bz[p] = 0.0f;

                for (int s = 0; s < S; ++s)
                {
                    int idx = i0 + s;
                    if (idx < 0 || idx >= static_cast<int>(grid.size()))
                        continue;

                    float fE = fields.E.field_x(idx);
                    float fB = fields.B.field_x(idx);

                    scratch.Ex[p] += static_cast<float>(w[s] * fE);
                    scratch.Ey[p] += static_cast<float>(w[s] * fE);
                    scratch.Ez[p] += static_cast<float>(w[s] * fE);
                    scratch.Bx[p] += static_cast<float>(w[s] * fB);
                    scratch.By[p] += static_cast<float>(w[s] * fB);
                    scratch.Bz[p] += static_cast<float>(w[s] * fB);
                }
            }
        }

        static inline void interpolate(const Vec3& x, const FieldGrid& grid, Vec3& E, Vec3& B)
        {
            // 1D stub (x-direction only). For higher dimensions, split by axis.
            int    i0;
            double w[S];
            SplineShape<Order>::weights(x.x, grid.cell_size(), i0, w);

            double ex = 0.0, bx = 0.0;
            for (int s = 0; s < S; ++s)
            {
                int idx = i0 + s;
                if (idx < 0 || idx >= static_cast<int>(grid.size()))
                    continue;
                // in pure kernel, user must provide a field container; this is an API-level placeholder.
            }

            E = {ex, 0.0, 0.0};
            B = {bx, 0.0, 0.0};
        }
        static inline void weights(double x, double h, int& i0, double (&w)[S])
        {
            SplineShape<Order>::weights(x, h, i0, w);
        }
    };

} // namespace kernels::shapes

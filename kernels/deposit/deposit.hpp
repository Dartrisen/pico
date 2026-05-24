#pragma once

namespace kernels::deposit
{
    struct CIC
    {
        static inline void weights(float xp, float dx, int& i0, float& w0, float& w1) noexcept
        {
            float s = xp * (1.0f / dx);
            i0      = static_cast<int>(s);
            float f = s - i0;
            w0      = 1.0f - f;
            w1      = f;
        }
    };

    template <class Shape, size_t BLOCK_SIZE>
    struct CurrentDeposit
    {
        static inline void deposit(const particle::ParticleBlock<BLOCK_SIZE>& pb, FieldSystem<BLOCK_SIZE>& J,
                                   const Grid& grid, float dt) const
        {
            for (size_t p = 0; p < pb.size_x; ++p)
            {
                int   i0;
                float w0, w1;

                Shape::weights(pb.momentum_x[p], grid.cell_size(), i0, w0, w1);

                float qvx = pb.charge[p] * pb.momentum_x[p];
                float qvy = pb.charge[p] * pb.momentum_y[p];
                float qvz = pb.charge[p] * pb.momentum_z[p];

                size_t idx0 = grid.idx(i0);
                size_t idx1 = grid.idx(i0 + 1);

                float J0 = J.field_x(idx0);
                float J1 = J.field_x(idx1);

                J0 += w0 * qvx;
                J0 += w0 * qvy;
                J0 += w0 * qvz;

                J1 += w1 * qvx;
                J1 += w1 * qvy;
                J1 += w1 * qvz;
            }
        }
    };
} // namespace kernels::deposit

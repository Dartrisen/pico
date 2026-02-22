#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"

namespace kernels::pusher
{

    template <size_t BLOCK_SIZE>
    struct BorisPusher
    {
        static void push_block(
            particle::ParticleBlock<BLOCK_SIZE> &pb,
            const FieldScratch<BLOCK_SIZE> &fs,
            float dt)
        {
            const float half_dt = 0.5f * dt;

            for (size_t p = 0; p < pb.activeCount; ++p)
            {

                // --- half electric kick ---
                float ux = pb.momentum_x[p] + half_dt * fs.Ex[p];
                float uy = pb.momentum_y[p] + half_dt * fs.Ey[p];
                float uz = pb.momentum_z[p] + half_dt * fs.Ez[p];

                // --- magnetic rotation ---
                float tx = half_dt * fs.Bx[p];
                float ty = half_dt * fs.By[p];
                float tz = half_dt * fs.Bz[p];

                float t2 = tx * tx + ty * ty + tz * tz;
                float sx = 2.f * tx / (1.f + t2);
                float sy = 2.f * ty / (1.f + t2);
                float sz = 2.f * tz / (1.f + t2);

                // u' = u- + u- x t
                float uxp = ux + (uy * tz - uz * ty);
                float uyp = uy + (uz * tx - ux * tz);
                float uzp = uz + (ux * ty - uy * tx);

                // u+ = u- + u' x s
                ux += (uyp * sz - uzp * sy);
                uy += (uzp * sx - uxp * sz);
                uz += (uxp * sy - uyp * sx);

                // --- half electric kick ---
                pb.momentum_x[p] = ux + half_dt * fs.Ex[p];
                pb.momentum_y[p] = uy + half_dt * fs.Ey[p];
                pb.momentum_z[p] = uz + half_dt * fs.Ez[p];

                // --- position update ---
                pb.position_x[p] += dt * pb.momentum_x[p];
            }
        }
    };

} // namespace kernels::pusher
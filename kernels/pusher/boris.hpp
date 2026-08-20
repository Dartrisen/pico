#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"

namespace kernels::pusher
{

    template <size_t BLOCK_SIZE>
    struct BorisPusher
    {
        static void push_block(particle::ParticleBlock<BLOCK_SIZE>& pb, const FieldScratch<BLOCK_SIZE>& fs, float dt)
        {
            const float half_dt = 0.5f * dt;

            for (size_t p = 0; p < pb.activeCount; ++p)
            {
                const float q_over_m  = pb.charge[p] / pb.mass[p];
                const float half_q_dt = half_dt * q_over_m;

                // --- half electric kick ---
                float ux = pb.momentum_x[p] + half_q_dt * fs.Ex[p];
                float uy = pb.momentum_y[p] + half_q_dt * fs.Ey[p];
                float uz = pb.momentum_z[p] + half_q_dt * fs.Ez[p];

                // --- magnetic rotation ---
                float tx = half_q_dt * fs.Bx[p];
                float ty = half_q_dt * fs.By[p];
                float tz = half_q_dt * fs.Bz[p];

                float t2 = tx * tx + ty * ty + tz * tz;
                float sx = 2.0f * tx / (1.0f + t2);
                float sy = 2.0f * ty / (1.0f + t2);
                float sz = 2.0f * tz / (1.0f + t2);

                // u' = u- + u- x t
                float uxp = ux + (uy * tz - uz * ty);
                float uyp = uy + (uz * tx - ux * tz);
                float uzp = uz + (ux * ty - uy * tx);

                // u+ = u- + u' x s
                ux += (uyp * sz - uzp * sy);
                uy += (uzp * sx - uxp * sz);
                uz += (uxp * sy - uyp * sx);

                // --- half electric kick ---
                pb.momentum_x[p] = ux + half_q_dt * fs.Ex[p];
                pb.momentum_y[p] = uy + half_q_dt * fs.Ey[p];
                pb.momentum_z[p] = uz + half_q_dt * fs.Ez[p];

                // --- position update (v = p / m) ---
                float vx = pb.momentum_x[p] / pb.mass[p];
                pb.position_x[p] += dt * vx;
            }
        }
    };

} // namespace kernels::pusher
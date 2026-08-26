#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"

namespace kernels::pusher
{

template <size_t BLOCK_SIZE>
struct BorisPusher
{
    static void push_block(particle::ParticleBlock<BLOCK_SIZE>& pb, const FieldScratch<BLOCK_SIZE>& fs, float dt, float q_over_m)
    {
        // Pre-calculate rotation & impulse factor once for the entire block pass
        const float half_qm_dt = 0.5f * q_over_m * dt;

        // clang-format off
        #pragma omp simd
        // clang-format on
        for (size_t p = 0; p < pb.activeCount; ++p)
        {
            // --- half electric kick ---
            float px = pb.momentum_x[p] + half_qm_dt * fs.Ex[p];
            float py = pb.momentum_y[p] + half_qm_dt * fs.Ey[p];
            float pz = pb.momentum_z[p] + half_qm_dt * fs.Ez[p];

            // --- magnetic rotation ---
            float tx = half_qm_dt * fs.Bx[p];
            float ty = half_qm_dt * fs.By[p];
            float tz = half_qm_dt * fs.Bz[p];

            float t2 = tx * tx + ty * ty + tz * tz;
            float sx = 2.0f * tx / (1.0f + t2);
            float sy = 2.0f * ty / (1.0f + t2);
            float sz = 2.0f * tz / (1.0f + t2);

            // p' = p- + p- x t
            float pxp = px + (py * tz - pz * ty);
            float pyp = py + (pz * tx - px * tz);
            float pzp = pz + (px * ty - py * tx);

            // p+ = p- + p' x s
            px += (pyp * sz - pzp * sy);
            py += (pzp * sx - pxp * sz);
            pz += (pxp * sy - pyp * sx);

            // --- half electric kick ---
            px += half_qm_dt * fs.Ex[p];
            py += half_qm_dt * fs.Ey[p];
            pz += half_qm_dt * fs.Ez[p];

            // Write back updated momentum/velocity
            pb.momentum_x[p] = px;
            pb.momentum_y[p] = py;
            pb.momentum_z[p] = pz;

            // --- position update (v = px) ---
            pb.position_x[p] += static_cast<double>(dt * px);
        }
    }
};

} // namespace kernels::pusher

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
        // clang-format off
        #pragma omp simd
        #pragma clang loop vectorize(enable) interleave(enable)
        // clang-format on
        for (size_t p = 0; p < pb.activeCount; ++p)
        {
            const float q          = pb.charge[p];
            const float inv_m      = 1.0f / pb.mass[p];
            const float half_q_dt  = half_dt * q;       // Impulse factor (q * dt / 2)
            const float half_qm_dt = half_q_dt * inv_m; // Rotation factor (q * dt / 2m)

            // --- half electric kick ---
            float px = pb.momentum_x[p] + half_q_dt * fs.Ex[p];
            float py = pb.momentum_y[p] + half_q_dt * fs.Ey[p];
            float pz = pb.momentum_z[p] + half_q_dt * fs.Ez[p];

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
            px += half_q_dt * fs.Ex[p];
            py += half_q_dt * fs.Ey[p];
            pz += half_q_dt * fs.Ez[p];

            // Write back updated momentum
            pb.momentum_x[p] = px;
            pb.momentum_y[p] = py;
            pb.momentum_z[p] = pz;

            // --- position update (v = p / m) ---
            pb.position_x[p] += dt * px * inv_m;
        }
    }
};

} // namespace kernels::pusher

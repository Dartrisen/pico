#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"

#include <cmath>

namespace kernels::pusher
{

template <size_t BLOCK_SIZE>
struct RelativisticBorisPusher
{
    static void push_block(particle::ParticleBlock<BLOCK_SIZE>& pb, const FieldScratch<BLOCK_SIZE>& fs, float dt)
    {
        const float half_dt = 0.5f * dt;

        // clang-format off
        #pragma omp simd
        // clang-format on
        for (size_t p = 0; p < pb.activeCount; ++p)
        {
            const float q         = pb.charge[p];
            const float m         = pb.mass[p];
            const float half_q_dt = half_dt * q;

            // --- Half electric kick ---
            float px = pb.momentum_x[p] + half_q_dt * fs.Ex[p];
            float py = pb.momentum_y[p] + half_q_dt * fs.Ey[p];
            float pz = pb.momentum_z[p] + half_q_dt * fs.Ez[p];

            // --- Compute Lorentz factor gamma from intermediate momentum ---
            // gamma = sqrt(1 + p^2 / (m^2))
            const float p_sq  = px * px + py * py + pz * pz;
            const float gamma = std::sqrt(1.0f + p_sq / (m * m));

            // --- Magnetic rotation factor (scaled by 1/gamma) ---
            // t = (q * dt / (2 * gamma * m)) * B
            const float inv_gamma_m      = 1.0f / (gamma * m);
            const float half_qm_dt_gamma = half_q_dt * inv_gamma_m;

            float tx = half_qm_dt_gamma * fs.Bx[p];
            float ty = half_qm_dt_gamma * fs.By[p];
            float tz = half_qm_dt_gamma * fs.Bz[p];

            float t2 = tx * tx + ty * ty + tz * tz;
            float sx = 2.0f * tx / (1.0f + t2);
            float sy = 2.0f * ty / (1.0f + t2);
            float sz = 2.0f * tz / (1.0f + t2);

            // --- Magnetic rotation (p' = p- + p- x t) ---
            float pxp = px + (py * tz - pz * ty);
            float pyp = py + (pz * tx - px * tz);
            float pzp = pz + (px * ty - py * tx);

            // p+ = p- + p' x s
            px += (pyp * sz - pzp * sy);
            py += (pzp * sx - pxp * sz);
            pz += (pxp * sy - pyp * sx);

            // --- Second half electric kick ---
            px += half_q_dt * fs.Ex[p];
            py += half_q_dt * fs.Ey[p];
            pz += half_q_dt * fs.Ez[p];

            // Write back updated momentum
            pb.momentum_x[p] = px;
            pb.momentum_y[p] = py;
            pb.momentum_z[p] = pz;

            // --- Position update using relativistic velocity (v = p / (gamma * m)) ---
            const float p_final_sq  = px * px + py * py + pz * pz;
            const float gamma_final = std::sqrt(1.0f + p_final_sq / (m * m));

            pb.position_x[p] += dt * px / (gamma_final * m);
        }
    }
};

} // namespace kernels::pusher
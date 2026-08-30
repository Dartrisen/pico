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
        const float half_qm_dt = 0.5f * dt * q_over_m;

        // clang-format off
        #pragma omp simd
        // clang-format on
        for (size_t p = 0; p < pb.activeCount; ++p)
        {
            // Load specific momentum (u = v in non-relativistic limit)
            float ux = pb.momentum_x[p];
            float uy = pb.momentum_y[p];
            float uz = pb.momentum_z[p];

            // --- First half electric kick ---
            ux += half_qm_dt * fs.Ex[p];
            uy += half_qm_dt * fs.Ey[p];
            uz += half_qm_dt * fs.Ez[p];

            // --- Magnetic rotation (gamma = 1.0) ---
            const float tx = half_qm_dt * fs.Bx[p];
            const float ty = half_qm_dt * fs.By[p];
            const float tz = half_qm_dt * fs.Bz[p];

            const float t2 = tx * tx + ty * ty + tz * tz;
            const float sx = 2.0f * tx / (1.0f + t2);
            const float sy = 2.0f * ty / (1.0f + t2);
            const float sz = 2.0f * tz / (1.0f + t2);

            // --- Magnetic rotation (u' = u- + u- x t) ---
            const float uxp = ux + (uy * tz - uz * ty);
            const float uyp = uy + (uz * tx - ux * tz);
            const float uzp = uz + (ux * ty - uy * tx);

            // u+ = u- + u' x s
            ux += (uyp * sz - uzp * sy);
            uy += (uzp * sx - uxp * sz);
            uz += (uxp * sy - uyp * sx);

            // --- Second half electric kick ---
            ux += half_qm_dt * fs.Ex[p];
            uy += half_qm_dt * fs.Ey[p];
            uz += half_qm_dt * fs.Ez[p];

            // Write back updated specific momentum
            pb.momentum_x[p] = ux;
            pb.momentum_y[p] = uy;
            pb.momentum_z[p] = uz;

            // --- Position update (v = u) & inv_gamma update ---
            pb.position_x[p] += dt * ux;
            pb.inv_gamma[p] = 1.0f;
        }
    }
};

} // namespace kernels::pusher

#pragma once

#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"

#include <algorithm>
#include <cstddef>

namespace kernels::deposit
{

template <class Shape, std::size_t BLOCK_SIZE>
struct EsirkepovDeposit
{
    static void deposit(const particle::ParticleBlock<BLOCK_SIZE>& pb, FieldSystem<BLOCK_SIZE>& J, const Grid& grid, float dt, float ppc)
    {
        constexpr int S = Shape::S;

        const double dx           = grid.cell_size();
        const float  inv_dx       = static_cast<float>(1.0 / dx);
        const float  inv_dt       = 1.0f / dt;
        const float  scale_factor = (1.0f / ppc) * inv_dx;
        const int    guards       = static_cast<int>(grid.guard_cells());
        const double max_x        = (static_cast<double>(grid.physical_size()) - 1e-7) * dx;

        for (std::size_t p = 0; p < pb.activeCount; ++p)
        {
            const float  inv_m = 1.0f / pb.mass[p];
            const double vx    = static_cast<double>(pb.momentum_x[p] * inv_m);
            const double x2    = std::clamp(static_cast<double>(pb.position_x[p]), 0.0, max_x);
            const double x1    = std::clamp(x2 - vx * static_cast<double>(dt), 0.0, max_x);

            int    i0_1 = 0, i0_2 = 0;
            double w1[S]{}, w2[S]{};
            Shape::weights(x1, dx, i0_1, w1);
            Shape::weights(x2, dx, i0_2, w2);

            const int min_node = std::min(i0_1, i0_2);
            const int max_node = std::max(i0_1, i0_2) + S - 1;

            const float j_scale          = pb.charge[p] * (1.0f / ppc) * inv_dt;
            double      accumulated_flux = 0.0;

            for (int node = min_node; node < max_node; ++node)
            {
                const double w1_val = (node >= i0_1 && node < i0_1 + S) ? w1[node - i0_1] : 0.0;
                const double w2_val = (node >= i0_2 && node < i0_2 + S) ? w2[node - i0_2] : 0.0;
                const double dw     = w2_val - w1_val;

                accumulated_flux -= dw;

                const int idx_half = node + guards;
                if (idx_half >= 0)
                {
                    J.field_x(static_cast<std::size_t>(idx_half)) += static_cast<float>(accumulated_flux) * j_scale;
                }
            }

            const float q_factor = pb.charge[p] * scale_factor;
            const float qvy      = q_factor * (pb.momentum_y[p] * inv_m);
            const float qvz      = q_factor * (pb.momentum_z[p] * inv_m);

            int    i0_mid = 0;
            double w_mid[S]{};
            Shape::weights(0.5 * (x1 + x2), dx, i0_mid, w_mid);
            const int node_start = std::max(0, i0_mid + guards);

            for (int s = 0; s < S; ++s)
            {
                const float w = static_cast<float>(w_mid[s]);
                J.field_y(static_cast<std::size_t>(node_start + s)) += w * qvy;
                J.field_z(static_cast<std::size_t>(node_start + s)) += w * qvz;
            }
        }
    }
};

template <class Shape, std::size_t BLOCK_SIZE>
struct FastEsirkepovDeposit
{
    static void deposit(const particle::ParticleBlock<BLOCK_SIZE>& pb, FieldSystem<BLOCK_SIZE>& J, const Grid& grid, float dt, float ppc)
    {
        constexpr int S = Shape::S;

        const double dx           = grid.cell_size();
        const double inv_dx       = 1.0 / dx;
        const double scale_factor = (1.0 / static_cast<double>(ppc)) * inv_dx;
        const double j_scale      = (1.0 / static_cast<double>(ppc)) / static_cast<double>(dt);
        const int    guards       = static_cast<int>(grid.guard_cells());
        const double max_x        = (static_cast<double>(grid.physical_size()) - 1e-7) * dx;

        for (std::size_t p = 0; p < pb.activeCount; ++p)
        {
            const double inv_m = 1.0 / static_cast<double>(pb.mass[p]);
            const double vx    = static_cast<double>(pb.momentum_x[p]) * inv_m;

            const double x2 = std::clamp(static_cast<double>(pb.position_x[p]), 0.0, max_x);
            const double x1 = std::clamp(x2 - vx * static_cast<double>(dt), 0.0, max_x);

            int    i0_1 = 0, i0_2 = 0;
            double w1[S]{}, w2[S]{};

            Shape::weights(x1, dx, i0_1, w1);
            Shape::weights(x2, dx, i0_2, w2);

            const int min_node  = std::min(i0_1, i0_2);
            const int max_node  = std::max(i0_1, i0_2) + S;
            const int off1      = i0_1 - min_node;
            const int off2      = i0_2 - min_node;
            const int num_faces = (max_node - min_node) - 1;

            double dw[S + 2]{};
            for (int s = 0; s < S; ++s)
            {
                dw[off1 + s] -= w1[s];
                dw[off2 + s] += w2[s];
            }

            const float q_jscale = pb.charge[p] * static_cast<float>(j_scale);
            double      flux     = 0.0;

            for (int k = 0; k < num_faces; ++k)
            {
                flux -= dw[k];
                const std::size_t idx_half = static_cast<std::size_t>(min_node + guards + k);
                J.field_x(idx_half) += static_cast<float>(flux) * q_jscale;
            }

            const float q_factor = pb.charge[p] * static_cast<float>(scale_factor);
            const float qvy      = q_factor * static_cast<float>(pb.momentum_y[p] * inv_m);
            const float qvz      = q_factor * static_cast<float>(pb.momentum_z[p] * inv_m);

            const std::size_t idx_node1 = static_cast<std::size_t>(i0_1 + guards);
            const std::size_t idx_node2 = static_cast<std::size_t>(i0_2 + guards);

            for (int s = 0; s < S; ++s)
            {
                const float w1_f = static_cast<float>(w1[s]);
                const float w2_f = static_cast<float>(w2[s]);

                J.field_y(idx_node1 + s) += 0.5f * w1_f * qvy;
                J.field_y(idx_node2 + s) += 0.5f * w2_f * qvy;

                J.field_z(idx_node1 + s) += 0.5f * w1_f * qvz;
                J.field_z(idx_node2 + s) += 0.5f * w2_f * qvz;
            }
        }
    }
};

template <class Shape, std::size_t BLOCK_SIZE>
struct OptEsirkepovDeposit
{
    static void deposit(const particle::ParticleBlock<BLOCK_SIZE>& pb, FieldSystem<BLOCK_SIZE>& J, const Grid& grid, float dt, float ppc)
    {
        constexpr int S = Shape::S;

        const float dx           = static_cast<float>(grid.cell_size());
        const float inv_dx       = 1.0f / dx;
        const float scale_factor = (1.0f / ppc) * inv_dx;
        const float j_scale      = (1.0f / ppc) / dt;
        const int   guards       = static_cast<int>(grid.guard_cells());
        const float max_x        = static_cast<float>((grid.physical_size() - 1e-7) * grid.cell_size());

        for (std::size_t p = 0; p < pb.activeCount; ++p)
        {
            const float inv_m = 1.0f / pb.mass[p];
            const float vx    = pb.momentum_x[p] * inv_m;

            const double x2 = std::clamp(static_cast<double>(pb.position_x[p]), 0.0, static_cast<double>(max_x));
            const double x1 = std::clamp(x2 - static_cast<double>(vx * dt), 0.0, static_cast<double>(max_x));

            int    i0_1 = 0, i0_2 = 0;
            double w1_d[S]{}, w2_d[S]{};

            Shape::weights(x1, static_cast<double>(dx), i0_1, w1_d);
            Shape::weights(x2, static_cast<double>(dx), i0_2, w2_d);

            const float q_jscale = pb.charge[p] * j_scale;
            const int   delta_i  = i0_2 - i0_1;

            // --- 1. Fast-Path: Particle stayed in same grid cell (delta_i == 0) ---
            if (delta_i == 0)
            {
                float     accum_dw = 0.0f;
                const int base_idx = i0_1 + guards;
                for (int s = 0; s < S - 1; ++s)
                {
                    const float dw = static_cast<float>(w2_d[s] - w1_d[s]);
                    accum_dw -= dw;
                    J.field_x(static_cast<std::size_t>(base_idx + s)) += accum_dw * q_jscale;
                }
            }
            // --- 2. Slow-Path: Particle crossed a cell boundary ---
            else
            {
                const int min_i     = std::min(i0_1, i0_2);
                const int max_i     = std::max(i0_1, i0_2) + S;
                const int num_faces = max_i - min_i - 1;

                float     dw_arr[S + 2] = {0.0f};
                const int off1          = i0_1 - min_i;
                const int off2          = i0_2 - min_i;

                for (int s = 0; s < S; ++s)
                {
                    dw_arr[off1 + s] -= static_cast<float>(w1_d[s]);
                    dw_arr[off2 + s] += static_cast<float>(w2_d[s]);
                }

                float     flux     = 0.0f;
                const int base_idx = min_i + guards;
                for (int k = 0; k < num_faces; ++k)
                {
                    flux -= dw_arr[k];
                    J.field_x(static_cast<std::size_t>(base_idx + k)) += flux * q_jscale;
                }
            }

            const float q_factor = pb.charge[p] * scale_factor;
            const float qvy      = q_factor * (pb.momentum_y[p] * inv_m);
            const float qvz      = q_factor * (pb.momentum_z[p] * inv_m);

            const std::size_t idx_node1 = static_cast<std::size_t>(i0_1 + guards);
            const std::size_t idx_node2 = static_cast<std::size_t>(i0_2 + guards);

            for (int s = 0; s < S; ++s)
            {
                const float w1_f = static_cast<float>(w1_d[s]);
                const float w2_f = static_cast<float>(w2_d[s]);

                J.field_y(idx_node1 + s) += 0.5f * w1_f * qvy;
                J.field_y(idx_node2 + s) += 0.5f * w2_f * qvy;

                J.field_z(idx_node1 + s) += 0.5f * w1_f * qvz;
                J.field_z(idx_node2 + s) += 0.5f * w2_f * qvz;
            }
        }
    }
};

} // namespace kernels::deposit
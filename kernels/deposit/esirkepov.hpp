#pragma once

#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace kernels::deposit
{

template <class Shape, std::size_t BLOCK_SIZE>
struct OptEsirkepovDeposit
{
    static void deposit(const particle::ParticleBlock<BLOCK_SIZE>& pb, FieldSystem<BLOCK_SIZE>& J, const Grid& grid, float dt, float ppc, float base_charge, float base_mass)
    {
        constexpr int S = Shape::S;

        const double dx           = grid.cell_size();
        const double inv_dx       = 1.0 / dx;
        const double j_scale      = (1.0 / static_cast<double>(ppc)) / static_cast<double>(dt);
        const double scale_factor = (1.0 / static_cast<double>(ppc)) * inv_dx;
        const int    guards       = static_cast<int>(grid.guard_cells());
        const double inv_m        = 1.0 / static_cast<double>(base_mass);

        const std::size_t max_idx = J.num_blocks() * BLOCK_SIZE;

        for (std::size_t p = 0; p < pb.activeCount; ++p)
        {
            const float q_p = pb.weight[p] * base_charge;

            // 1. Calculate proper relativistic velocity v = u / gamma
            const double ux   = static_cast<double>(pb.momentum_x[p]);
            const double uy   = static_cast<double>(pb.momentum_y[p]);
            const double uz   = static_cast<double>(pb.momentum_z[p]);
            const double u_sq = ux * ux + uy * uy + uz * uz;

            // Relativistic factor gamma (c = 1)
            const double gamma       = std::sqrt(1.0 + u_sq);
            const double inv_gamma_m = inv_m / gamma;

            const double vx = ux * inv_gamma_m;
            const double vy = uy * inv_gamma_m;
            const double vz = uz * inv_gamma_m;

            // 2. Compute positions without std::clamp so guard cells collect boundary flux
            const double x2 = static_cast<double>(pb.position_x[p]);
            const double x1 = x2 - vx * static_cast<double>(dt);

            int    i0_1 = 0, i0_2 = 0;
            double w1[S]{}, w2[S]{};

            Shape::weights(x1, dx, i0_1, w1);
            Shape::weights(x2, dx, i0_2, w2);

            const int min_node = std::min(i0_1, i0_2);
            const int max_node = std::max(i0_1, i0_2) + S;

            const float q_jscale         = q_p * static_cast<float>(j_scale);
            double      accumulated_flux = 0.0;

            // 3. Longitudinal current deposition (J_x)
            for (int node = min_node; node < max_node - 1; ++node)
            {
                const double w1_val = (node >= i0_1 && node < i0_1 + S) ? w1[node - i0_1] : 0.0;
                const double w2_val = (node >= i0_2 && node < i0_2 + S) ? w2[node - i0_2] : 0.0;

                accumulated_flux -= (w2_val - w1_val);

                const int idx_half = node + guards;
                if (idx_half >= 0 && static_cast<std::size_t>(idx_half) < max_idx)
                {
                    J.field_x(static_cast<std::size_t>(idx_half)) += static_cast<float>(accumulated_flux) * q_jscale;
                }
            }

            // 4. Transverse current deposition (J_y, J_z) using physical velocities
            const float q_factor = q_p * static_cast<float>(scale_factor);
            const float qvy      = q_factor * static_cast<float>(vy);
            const float qvz      = q_factor * static_cast<float>(vz);

            for (int s = 0; s < S; ++s)
            {
                const float w1_f = static_cast<float>(w1[s]);
                const float w2_f = static_cast<float>(w2[s]);

                const int n1 = i0_1 + guards + s;
                if (n1 >= 0 && static_cast<std::size_t>(n1) < max_idx)
                {
                    J.field_y(static_cast<std::size_t>(n1)) += 0.5f * w1_f * qvy;
                    J.field_z(static_cast<std::size_t>(n1)) += 0.5f * w1_f * qvz;
                }

                const int n2 = i0_2 + guards + s;
                if (n2 >= 0 && static_cast<std::size_t>(n2) < max_idx)
                {
                    J.field_y(static_cast<std::size_t>(n2)) += 0.5f * w2_f * qvy;
                    J.field_z(static_cast<std::size_t>(n2)) += 0.5f * w2_f * qvz;
                }
            }
        }
    }
};

} // namespace kernels::deposit
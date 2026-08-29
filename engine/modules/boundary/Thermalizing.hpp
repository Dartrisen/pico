#pragma once

#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"

#include <cmath>
#include <random>

namespace pico::modules::boundary
{

template <std::size_t BLOCK_SIZE>
class ThermalizingParticleBoundary
{
private:
    float        v_thermal_{0.01f}; // sqrt(k_B * T_wall / m)
    std::mt19937 rng_{1337};

public:
    ThermalizingParticleBoundary() = default;

    explicit ThermalizingParticleBoundary(float v_thermal) : v_thermal_(v_thermal) {}

    void apply(particle::ParticleBlock<BLOCK_SIZE>& block, const Grid& grid)
    {
        const double                         L = static_cast<double>(grid.physical_size()) * grid.cell_size();
        std::normal_distribution<float>      gauss(0.0f, v_thermal_);
        std::exponential_distribution<float> rayleigh(1.0f / (2.0f * v_thermal_ * v_thermal_));

        for (std::size_t p = 0; p < block.activeCount; ++p)
        {
            const double x = block.position_x[p];

            // Left Wall Collision (x < 0)
            if (x < 0.0)
            {
                const float px = std::sqrt(rayleigh(rng_));
                const float py = gauss(rng_);
                const float pz = gauss(rng_);

                block.position_x[p] = 0.0;
                block.momentum_x[p] = px;
                block.momentum_y[p] = py;
                block.momentum_z[p] = pz;

                const float p_sq   = px * px + py * py + pz * pz;
                block.inv_gamma[p] = 1.0f / std::sqrt(1.0f + p_sq);
            }
            // Right Wall Collision (x >= L)
            else if (x >= L)
            {
                const float px = -std::sqrt(rayleigh(rng_));
                const float py = gauss(rng_);
                const float pz = gauss(rng_);

                block.position_x[p] = L - 1e-6;
                block.momentum_x[p] = px;
                block.momentum_y[p] = py;
                block.momentum_z[p] = pz;

                const float p_sq   = px * px + py * py + pz * pz;
                block.inv_gamma[p] = 1.0f / std::sqrt(1.0f + p_sq);
            }
        }
    }
};

} // namespace pico::modules::boundary
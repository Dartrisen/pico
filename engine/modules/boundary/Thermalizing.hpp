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
        float        v_thermal_; // sqrt(k_B * T_wall / m)
        std::mt19937 rng_{1337};

    public:
        ThermalizingParticleBoundary() = default;

        explicit ThermalizingParticleBoundary(float v_thermal) : v_thermal_(v_thermal) {}

        void apply(particle::ParticleBlock<BLOCK_SIZE>& block, const Grid& grid)
        {
            const double                         L = static_cast<double>(grid.physical_size()) * grid.cell_size();
            std::normal_distribution<float>      gauss(0.0f, v_thermal_);
            std::exponential_distribution<float> Rayleigh(1.0f / (2.0f * v_thermal_ * v_thermal_));

            for (std::size_t p = 0; p < block.activeCount; ++p)
            {
                double x = block.position_x[p];

                // Left Wall Collision (x < 0)
                if (x < 0.0)
                {
                    block.position_x[p] = 0.0;
                    // Inject positive momentum x (half-Maxwellian flux)
                    block.momentum_x[p] = block.mass[p] * std::sqrt(Rayleigh(rng_));
                    block.momentum_y[p] = block.mass[p] * gauss(rng_);
                    block.momentum_z[p] = block.mass[p] * gauss(rng_);
                }
                // Right Wall Collision (x >= L)
                else if (x >= L)
                {
                    block.position_x[p] = L - 1e-6;
                    // Inject negative momentum x
                    block.momentum_x[p] = -block.mass[p] * std::sqrt(Rayleigh(rng_));
                    block.momentum_y[p] = block.mass[p] * gauss(rng_);
                    block.momentum_z[p] = block.mass[p] * gauss(rng_);
                }
            }
        }
    };
} // namespace pico::modules::boundary
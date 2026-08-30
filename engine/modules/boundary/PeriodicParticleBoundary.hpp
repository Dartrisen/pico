#pragma once

#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_system.hpp"

#include <cstddef>

namespace pico::modules::boundary
{
template <std::size_t BLOCK_SIZE = 64>
class PeriodicBoundaryParticleHandler
{
public:
    void apply(particle::ParticleBlock<BLOCK_SIZE>& block, const Grid& grid) const noexcept
    {
        const float domain_length = static_cast<float>(grid.physical_size()) * static_cast<float>(grid.cell_size());

        for (std::size_t i = 0; i < block.activeCount; ++i)
        {
            // Wrap left boundary
            while (block.position_x[i] < 0.0f)
            {
                block.position_x[i] += domain_length;
            }

            // Wrap right boundary
            while (block.position_x[i] >= domain_length)
            {
                block.position_x[i] -= domain_length;
            }
        }
    }
};
} // namespace pico::modules::boundary
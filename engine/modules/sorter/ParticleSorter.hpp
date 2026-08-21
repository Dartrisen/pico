#pragma once

#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"
#include "kernels/sorter/particle_sorter.hpp"

namespace pico::modules::sorter
{

template <std::size_t BLOCK_SIZE>
class ParticleSorter
{
public:
    void sort_block(particle::ParticleBlock<BLOCK_SIZE>& block, const Grid& grid)
    {
        kernels::sorter::ParticleSorter<BLOCK_SIZE>::sort_block(block, grid, scratch_);
    }

private:
    kernels::sorter::ParticleSorterScratch<BLOCK_SIZE> scratch_;
};

} // namespace pico::modules::sorter
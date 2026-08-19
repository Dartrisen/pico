#pragma once

#include "data/field/include/field_em.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"
#include "kernels/gather/gather.hpp"

namespace pico::modules::gather
{

    template <class Shape, size_t BLOCK_SIZE>
    struct Gather
    {
        void gather_block(const particle::ParticleBlock<BLOCK_SIZE>& block, const EMFields<BLOCK_SIZE>& fields,
                          const Grid& grid, FieldScratch<BLOCK_SIZE>& scratch)
        {
            kernels::gather::FieldGather<Shape, BLOCK_SIZE>::gather(block, fields, grid, scratch);
        }
    };

} // namespace pico::modules::gather

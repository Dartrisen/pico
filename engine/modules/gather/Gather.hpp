#pragma once

#include "kernels/gather/gather.hpp"
#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"

namespace pico::modules::gather
{

    template <class Shape, size_t BLOCK_SIZE>
    struct Gather
    {
        static void gather(
            const particle::ParticleSystem<BLOCK_SIZE> &particles,
            const EMFields<BLOCK_SIZE> &fields,
            const Grid &grid,
            FieldScratch<BLOCK_SIZE> &scratch)
        {
            for (auto &block : particles)
            {
                kernels::gather::FieldGather<Shape, BLOCK_SIZE>::gather(
                    block, fields, grid, scratch);
            }
        }
    };

} // namespace pico::modules::gather

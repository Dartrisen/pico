#pragma once
#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"
#include "kernels/pusher/boris.hpp"

namespace pico::modules::pusher
{

    template <
            // bool Relativistic,
            // class Shape,
            size_t BLOCK_SIZE>
    struct BorisPusher
    {
        void push(particle::ParticleSystem<BLOCK_SIZE>& particles, const FieldScratch<BLOCK_SIZE>& fields,
                  double dt) const
        {
            for (auto& block : particles)
            {
                kernels::pusher::BorisPusher<BLOCK_SIZE>::push_block(block, fields, dt);
            }
        }
    };

} // namespace pico::modules::pusher
#pragma once
#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"
#include "kernels/pusher/boris.hpp"
#include "kernels/pusher/relativistic_boris.hpp"

namespace pico::modules::pusher
{

template <size_t BLOCK_SIZE>
struct BorisPusher
{
    void push_block(particle::ParticleBlock<BLOCK_SIZE>& block, const FieldScratch<BLOCK_SIZE>& fields, double dt, float q_over_m) const
    {
        kernels::pusher::BorisPusher<BLOCK_SIZE>::push_block(block, fields, dt, q_over_m);
    }
};

template <size_t BLOCK_SIZE>
struct RelativisticBorisPusher
{
    void push_block(particle::ParticleBlock<BLOCK_SIZE>& block, const FieldScratch<BLOCK_SIZE>& fields, double dt, float q_over_m) const
    {
        kernels::pusher::RelativisticBorisPusher<BLOCK_SIZE>::push_block(block, fields, dt, q_over_m);
    }
};

} // namespace pico::modules::pusher
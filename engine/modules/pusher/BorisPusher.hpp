#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"
#include "kernels/pusher/boris.hpp"
#include "kernels/pusher/relativistic_boris.hpp"

#if defined(__APPLE__)
#include "kernels/pusher/metal_boris.hpp"
#endif

namespace pico::modules::pusher
{

enum class ExecutionMode
{
    CPU_Standard,
    CPU_Relativistic,
    GPU_Metal
};

// Standard CPU OpenMP Boris Pusher
template <size_t BLOCK_SIZE>
struct BorisPusher
{
    void push_block(particle::ParticleBlock<BLOCK_SIZE>& block, const FieldScratch<BLOCK_SIZE>& fields, float dt) const
    {
        kernels::pusher::BorisPusher<BLOCK_SIZE>::push_block(block, fields, dt);
    }
};

// Relativistic CPU OpenMP Boris Pusher
template <size_t BLOCK_SIZE>
struct RelativisticBorisPusher
{
    void push_block(particle::ParticleBlock<BLOCK_SIZE>& block, const FieldScratch<BLOCK_SIZE>& fields, float dt) const
    {
        kernels::pusher::RelativisticBorisPusher<BLOCK_SIZE>::push_block(block, fields, dt);
    }
};

// Dedicated GPU Metal Boris Pusher
template <size_t BLOCK_SIZE>
struct MetalBorisPusher
{
    // clang-format off
    #if defined(__APPLE__)
    kernels::pusher::MetalRelativisticBorisPusher<BLOCK_SIZE> metal_pusher;
    #endif
    // clang-format on

    void push_block(particle::ParticleBlock<BLOCK_SIZE>& block, const FieldScratch<BLOCK_SIZE>& fields, float dt) const
    {
        // clang-format off
        #if defined(__APPLE__)
        metal_pusher.push_block(block, fields, dt);
        #endif
        // clang-format on
    }

    void sync() const
    {
        // clang-format off
        #if defined(__APPLE__)
        metal_pusher.sync();
        #endif
        // clang-format on
    }
};

// Hot-Switchable Unified Engine Pusher
template <size_t BLOCK_SIZE>
struct HotSwitchBorisPusher
{
    ExecutionMode mode = ExecutionMode::CPU_Relativistic;
    // clang-format off
    #if defined(__APPLE__)
    kernels::pusher::MetalRelativisticBorisPusher<BLOCK_SIZE> metal_pusher;
    #endif
    // clang-format on

    void set_mode(ExecutionMode new_mode)
    {
        mode = new_mode;
    }

    void push_block(particle::ParticleBlock<BLOCK_SIZE>& block, const FieldScratch<BLOCK_SIZE>& fields, float dt) const
    {
        switch (mode)
        {
            case ExecutionMode::GPU_Metal:
                // clang-format off
                #if defined(__APPLE__)
                metal_pusher.push_block(block, fields, dt);
                break;
                #endif
                // clang-format on
            case ExecutionMode::CPU_Standard:
                kernels::pusher::BorisPusher<BLOCK_SIZE>::push_block(block, fields, dt);
                break;

            case ExecutionMode::CPU_Relativistic:
            default:
                kernels::pusher::RelativisticBorisPusher<BLOCK_SIZE>::push_block(block, fields, dt);
                break;
        }
    }
};

} // namespace pico::modules::pusher
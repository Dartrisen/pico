#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"

#include <vector>

namespace MTL
{

class Device;
class CommandQueue;
class ComputePipelineState;
class Buffer;

} // namespace MTL

namespace kernels::pusher
{

template <size_t BLOCK_SIZE>
class MetalRelativisticBorisPusher
{
public:
    MetalRelativisticBorisPusher();
    ~MetalRelativisticBorisPusher();

    void push_block(particle::ParticleBlock<BLOCK_SIZE>& pb, const FieldScratch<BLOCK_SIZE>& fs, float dt) const;
    void sync() const;

private:
    MTL::Device*               device         = nullptr;
    MTL::CommandQueue*         command_queue  = nullptr;
    MTL::ComputePipelineState* pipeline_state = nullptr;

    static constexpr size_t   POOL_SIZE = 512;
    std::vector<MTL::Buffer*> buffer_pool;
    mutable size_t            pool_index = 0;
};

} // namespace kernels::pusher
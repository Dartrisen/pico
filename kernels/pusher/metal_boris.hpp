#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace kernels::pusher
{

struct BufferSlot
{
    NS::SharedPtr<MTL::Buffer>        buffer;
    NS::SharedPtr<MTL::CommandBuffer> last_command;
    std::mutex                        mutex;
};

template <size_t BLOCK_SIZE>
class MetalRelativisticBorisPusher
{
public:
    MetalRelativisticBorisPusher();
    ~MetalRelativisticBorisPusher() = default;

    MetalRelativisticBorisPusher(const MetalRelativisticBorisPusher&)            = delete;
    MetalRelativisticBorisPusher& operator=(const MetalRelativisticBorisPusher&) = delete;

    MetalRelativisticBorisPusher(MetalRelativisticBorisPusher&& other) noexcept;
    MetalRelativisticBorisPusher& operator=(MetalRelativisticBorisPusher&& other) noexcept;

    void push_block(particle::ParticleBlock<BLOCK_SIZE>& pb, const FieldScratch<BLOCK_SIZE>& fs, float dt) const;
    void sync() const;

private:
    NS::SharedPtr<MTL::Device>               device_;
    NS::SharedPtr<MTL::CommandQueue>         command_queue_;
    NS::SharedPtr<MTL::ComputePipelineState> pipeline_state_;

    static constexpr size_t                          POOL_SIZE = 64;
    mutable std::vector<std::unique_ptr<BufferSlot>> buffer_pool_;
    mutable std::atomic<size_t>                      pool_index_{0};
};

} // namespace kernels::pusher
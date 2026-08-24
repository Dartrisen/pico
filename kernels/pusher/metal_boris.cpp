#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "metal_boris.hpp"

#include "rules_cc/cc/runfiles/runfiles.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
#include <stdexcept>
#include <string>
#include <vector>

using rules_cc::cc::runfiles::Runfiles;

namespace kernels::pusher
{

static std::string resolve_metallib_path()
{
    if (const char* env = std::getenv("PICO_METAL_LIBRARY"))
        return env;

    char     exe_path[1024];
    uint32_t size = sizeof(exe_path);
    _NSGetExecutablePath(exe_path, &size);

    std::string               error{};
    std::unique_ptr<Runfiles> runfiles(Runfiles::Create(exe_path, &error));
    if (!runfiles)
        runfiles.reset(Runfiles::CreateForTest(&error));

    if (!runfiles)
        throw std::runtime_error("Bazel runfiles init failed: " + error);

    return runfiles->Rlocation("pico/kernels/pusher/boris.metallib");
}

template <size_t BLOCK_SIZE>
MetalRelativisticBorisPusher<BLOCK_SIZE>::MetalRelativisticBorisPusher()
{
    device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device_)
        throw std::runtime_error("Metal GPU device is unavailable.");

    command_queue_ = NS::TransferPtr(device_->newCommandQueue());
    if (!command_queue_)
        throw std::runtime_error("Could not create Metal command queue.");

    std::string metallib_path = resolve_metallib_path();
    NS::Error*  error         = nullptr;
    auto        path          = NS::RetainPtr(NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding));

    auto library = NS::TransferPtr(device_->newLibrary(path.get(), &error));
    if (!library)
    {
        std::string message = "Could not load metallib from resolved path: " + metallib_path;
        if (error && error->localizedDescription())
            message += " (" + std::string(error->localizedDescription()->utf8String()) + ")";
        throw std::runtime_error(message);
    }

    auto fn_name = NS::RetainPtr(NS::String::string("relativistic_boris_push", NS::UTF8StringEncoding));
    auto kernel  = NS::TransferPtr(library->newFunction(fn_name.get()));

    if (!kernel)
        throw std::runtime_error("Could not find shader function 'relativistic_boris_push' in metallib.");

    pipeline_state_ = NS::TransferPtr(device_->newComputePipelineState(kernel.get(), &error));
    if (!pipeline_state_)
        throw std::runtime_error("Could not create Metal compute pipeline state.");

    const size_t bytes_per_block_buffer = 12 * BLOCK_SIZE * sizeof(float);
    buffer_pool_.reserve(POOL_SIZE);
    for (size_t i = 0; i < POOL_SIZE; ++i)
    {
        auto slot    = std::make_unique<BufferSlot>();
        slot->buffer = NS::TransferPtr(device_->newBuffer(bytes_per_block_buffer, MTL::ResourceStorageModeShared));
        if (!slot->buffer)
            throw std::runtime_error("Failed to allocate pre-allocated Metal ring buffer.");
        buffer_pool_.push_back(std::move(slot));
    }
}

template <size_t BLOCK_SIZE>
MetalRelativisticBorisPusher<BLOCK_SIZE>::MetalRelativisticBorisPusher(MetalRelativisticBorisPusher&& other) noexcept
        : device_(std::move(other.device_)), command_queue_(std::move(other.command_queue_)), pipeline_state_(std::move(other.pipeline_state_)),
          buffer_pool_(std::move(other.buffer_pool_)), pool_index_(other.pool_index_.load(std::memory_order_relaxed))
{
}

template <size_t BLOCK_SIZE>
MetalRelativisticBorisPusher<BLOCK_SIZE>& MetalRelativisticBorisPusher<BLOCK_SIZE>::operator=(MetalRelativisticBorisPusher&& other) noexcept
{
    if (this != &other)
    {
        device_         = std::move(other.device_);
        command_queue_  = std::move(other.command_queue_);
        pipeline_state_ = std::move(other.pipeline_state_);
        buffer_pool_    = std::move(other.buffer_pool_);
        pool_index_.store(other.pool_index_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
}

template <size_t BLOCK_SIZE>
void MetalRelativisticBorisPusher<BLOCK_SIZE>::push_block(particle::ParticleBlock<BLOCK_SIZE>& pb, const FieldScratch<BLOCK_SIZE>& fs, float dt) const
{
    if (pb.activeCount == 0)
        return;

    const uint32_t active_count = static_cast<uint32_t>(pb.activeCount);
    const size_t   active_bytes = active_count * sizeof(float);

    const size_t idx  = pool_index_.fetch_add(1, std::memory_order_relaxed);
    auto&        slot = *buffer_pool_[idx % POOL_SIZE];

    std::lock_guard<std::mutex> lock(slot.mutex);

    // Guaranteed wait on slot reuse: ensures completion callback has finished copying back
    if (slot.last_command)
    {
        slot.last_command->waitUntilCompleted();
        slot.last_command.reset();
    }

    MTL::Buffer* buf   = slot.buffer.get();
    float*       base  = static_cast<float*>(buf->contents());
    float*       pos_x = base + 0 * BLOCK_SIZE;
    float*       mom_x = base + 1 * BLOCK_SIZE;
    float*       mom_y = base + 2 * BLOCK_SIZE;
    float*       mom_z = base + 3 * BLOCK_SIZE;
    float*       q     = base + 4 * BLOCK_SIZE;
    float*       m     = base + 5 * BLOCK_SIZE;
    float*       Ex    = base + 6 * BLOCK_SIZE;
    float*       Ey    = base + 7 * BLOCK_SIZE;
    float*       Ez    = base + 8 * BLOCK_SIZE;
    float*       Bx    = base + 9 * BLOCK_SIZE;
    float*       By    = base + 10 * BLOCK_SIZE;
    float*       Bz    = base + 11 * BLOCK_SIZE;

    for (size_t i = 0; i < active_count; ++i)
        pos_x[i] = static_cast<float>(pb.position_x[i]);

    std::memcpy(mom_x, pb.momentum_x.data(), active_bytes);
    std::memcpy(mom_y, pb.momentum_y.data(), active_bytes);
    std::memcpy(mom_z, pb.momentum_z.data(), active_bytes);
    std::memcpy(q, pb.charge.data(), active_bytes);
    std::memcpy(m, pb.mass.data(), active_bytes);

    std::memcpy(Ex, fs.Ex.data(), active_bytes);
    std::memcpy(Ey, fs.Ey.data(), active_bytes);
    std::memcpy(Ez, fs.Ez.data(), active_bytes);
    std::memcpy(Bx, fs.Bx.data(), active_bytes);
    std::memcpy(By, fs.By.data(), active_bytes);
    std::memcpy(Bz, fs.Bz.data(), active_bytes);

    MTL::CommandBuffer*         command = command_queue_->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = command->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline_state_.get());

    const size_t block_bytes = BLOCK_SIZE * sizeof(float);
    for (NS::UInteger i = 0; i < 12; ++i)
        encoder->setBuffer(buf, i * block_bytes, i);

    encoder->setBytes(&dt, sizeof(dt), 12);
    encoder->setBytes(&active_count, sizeof(active_count), 13);

    const NS::UInteger threads_per_group = 256;
    const NS::UInteger threadgroups      = (active_count + threads_per_group - 1) / threads_per_group;

    encoder->dispatchThreadgroups(MTL::Size::Make(threadgroups, 1, 1), MTL::Size::Make(threads_per_group, 1, 1));
    encoder->endEncoding();

    command->addCompletedHandler(
            [&pb, pos_x, mom_x, mom_y, mom_z, active_bytes, active_count](MTL::CommandBuffer* cb)
            {
                if (cb->status() == MTL::CommandBufferStatusError)
                    return;

                for (size_t i = 0; i < active_count; ++i)
                    pb.position_x[i] = static_cast<double>(pos_x[i]);

                std::memcpy(pb.momentum_x.data(), mom_x, active_bytes);
                std::memcpy(pb.momentum_y.data(), mom_y, active_bytes);
                std::memcpy(pb.momentum_z.data(), mom_z, active_bytes);
            });

    slot.last_command = NS::RetainPtr(command);
    command->commit();
}

template <size_t BLOCK_SIZE>
void MetalRelativisticBorisPusher<BLOCK_SIZE>::sync() const
{
    for (auto& slot_ptr : buffer_pool_)
    {
        std::lock_guard<std::mutex> lock(slot_ptr->mutex);
        if (slot_ptr->last_command)
        {
            slot_ptr->last_command->waitUntilCompleted();
            slot_ptr->last_command.reset();
        }
    }
}

template class MetalRelativisticBorisPusher<64>;
template class MetalRelativisticBorisPusher<128>;
template class MetalRelativisticBorisPusher<256>;
template class MetalRelativisticBorisPusher<512>;
template class MetalRelativisticBorisPusher<1024>;
template class MetalRelativisticBorisPusher<2048>;
template class MetalRelativisticBorisPusher<4096>;
template class MetalRelativisticBorisPusher<8192>;

} // namespace kernels::pusher
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "metal_boris.hpp"

#include "rules_cc/cc/runfiles/runfiles.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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
    device = MTL::CreateSystemDefaultDevice();
    if (!device)
        throw std::runtime_error("Metal GPU device is unavailable.");

    command_queue = device->newCommandQueue();
    if (!command_queue)
        throw std::runtime_error("Could not create Metal command queue.");

    std::string metallib_path = resolve_metallib_path();
    NS::Error*  error         = nullptr;
    auto*       path          = NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);

    MTL::Library* library = device->newLibrary(path, &error);
    if (!library)
    {
        std::string message = "Could not load metallib from resolved path: " + metallib_path;
        if (error && error->localizedDescription())
            message += " (" + std::string(error->localizedDescription()->utf8String()) + ")";
        throw std::runtime_error(message);
    }

    auto*          fn_name = NS::String::string("relativistic_boris_push", NS::UTF8StringEncoding);
    MTL::Function* kernel  = library->newFunction(fn_name);

    if (!kernel)
    {
        library->release();
        throw std::runtime_error("Could not find shader function 'relativistic_boris_push' in metallib.");
    }

    pipeline_state = device->newComputePipelineState(kernel, &error);
    kernel->release();
    library->release();

    if (!pipeline_state)
        throw std::runtime_error("Could not create Metal compute pipeline state.");

    // Pre-allocate persistent pool of unified memory GPU buffers (12 arrays per block)
    const size_t bytes_per_block_buffer = 12 * BLOCK_SIZE * sizeof(float);
    buffer_pool.resize(POOL_SIZE);
    for (size_t i = 0; i < POOL_SIZE; ++i)
    {
        buffer_pool[i] = device->newBuffer(bytes_per_block_buffer, MTL::ResourceStorageModeShared);
        if (!buffer_pool[i])
            throw std::runtime_error("Failed to allocate pre-allocated Metal ring buffer.");
    }
}

template <size_t BLOCK_SIZE>
MetalRelativisticBorisPusher<BLOCK_SIZE>::~MetalRelativisticBorisPusher()
{
    for (auto* buf : buffer_pool)
    {
        if (buf)
            buf->release();
    }
    buffer_pool.clear();

    if (pipeline_state)
        pipeline_state->release();
    if (command_queue)
        command_queue->release();
    if (device)
        device->release();
}

template <size_t BLOCK_SIZE>
void MetalRelativisticBorisPusher<BLOCK_SIZE>::push_block(particle::ParticleBlock<BLOCK_SIZE>& pb, const FieldScratch<BLOCK_SIZE>& fs, float dt) const
{
    if (pb.activeCount == 0)
        return;

    MTL::Buffer* buf = buffer_pool[pool_index % POOL_SIZE];
    pool_index++;

    float* base  = static_cast<float*>(buf->contents());
    float* pos_x = base + 0 * BLOCK_SIZE;
    float* mom_x = base + 1 * BLOCK_SIZE;
    float* mom_y = base + 2 * BLOCK_SIZE;
    float* mom_z = base + 3 * BLOCK_SIZE;
    float* q     = base + 4 * BLOCK_SIZE;
    float* m     = base + 5 * BLOCK_SIZE;
    float* Ex    = base + 6 * BLOCK_SIZE;
    float* Ey    = base + 7 * BLOCK_SIZE;
    float* Ez    = base + 8 * BLOCK_SIZE;
    float* Bx    = base + 9 * BLOCK_SIZE;
    float* By    = base + 10 * BLOCK_SIZE;
    float* Bz    = base + 11 * BLOCK_SIZE;

    for (size_t i = 0; i < pb.activeCount; ++i)
        pos_x[i] = static_cast<float>(pb.position_x[i]);

    const size_t active_bytes = pb.activeCount * sizeof(float);
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

    MTL::CommandBuffer*         command = command_queue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = command->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline_state);

    const size_t block_bytes = BLOCK_SIZE * sizeof(float);
    for (NS::UInteger i = 0; i < 12; ++i)
        encoder->setBuffer(buf, i * block_bytes, i);

    encoder->setBytes(&dt, sizeof(dt), 12);

    const NS::UInteger threads_per_group = 256;
    const NS::UInteger threadgroups      = (pb.activeCount + threads_per_group - 1) / threads_per_group;

    encoder->dispatchThreadgroups(MTL::Size::Make(threadgroups, 1, 1), MTL::Size::Make(threads_per_group, 1, 1));

    encoder->endEncoding();

    // ASYNCHRONOUS: Add a completion handler to copy back results only when GPU finishes
    command->addCompletedHandler(
            [&pb, pos_x, mom_x, mom_y, mom_z, active_bytes](MTL::CommandBuffer*)
            {
                for (size_t i = 0; i < pb.activeCount; ++i)
                    pb.position_x[i] = static_cast<double>(pos_x[i]);

                std::memcpy(pb.momentum_x.data(), mom_x, active_bytes);
                std::memcpy(pb.momentum_y.data(), mom_y, active_bytes);
                std::memcpy(pb.momentum_z.data(), mom_z, active_bytes);
            });

    command->commit();
}

template <size_t BLOCK_SIZE>
void MetalRelativisticBorisPusher<BLOCK_SIZE>::sync() const
{
    // Call once per main simulation timestep after looping over all blocks
    MTL::CommandBuffer* command = command_queue->commandBuffer();
    command->commit();
    command->waitUntilCompleted();
}

template class MetalRelativisticBorisPusher<128>;
template class MetalRelativisticBorisPusher<256>;
template class MetalRelativisticBorisPusher<512>;
template class MetalRelativisticBorisPusher<1024>;
template class MetalRelativisticBorisPusher<2048>;
template class MetalRelativisticBorisPusher<4096>;
template class MetalRelativisticBorisPusher<8192>;

} // namespace kernels::pusher
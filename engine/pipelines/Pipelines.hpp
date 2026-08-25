#pragma once

#include "data/field/include/field_em.hpp"
#include "data/field/include/field_system.hpp"
#include "data/particle/include/particle_system.hpp"
#include "engine/perf/LocalityProfiler.hpp"
#include "engine/perf/PipelineProfiler.hpp"

#include <omp.h>
#include <vector>

namespace pico::pipeline
{

template <std::size_t BLOCK_SIZE>
struct PipelineContext
{
    particle::ParticleSystem<BLOCK_SIZE>&            particles;
    const EMFields<BLOCK_SIZE>&                      fields;
    std::vector<FieldSystem<BLOCK_SIZE>>&            thread_currents;
    std::vector<FieldScratch<BLOCK_SIZE>>&           thread_scratch;
    std::vector<pico::diagnostics::LocalityMetrics>& thread_metrics;
    const Grid&                                      grid;
    double                                           dt;
    std::size_t                                      particles_per_cell;
    pico::perf::PipelineProfiler&                    profiler;
    bool                                             enable_profiling;
    bool                                             enable_locality;
};

struct CpuFusedPipeline
{
    template <std::size_t BLOCK_SIZE, typename GatherT, typename PusherT, typename ParticleBoundaryT, typename DepositT>
    static void execute(PipelineContext<BLOCK_SIZE>& ctx, GatherT& gather, PusherT& pusher, ParticleBoundaryT& particle_boundary, DepositT& deposit)
    {
#pragma omp parallel
        {
            const int tid            = omp_get_thread_num();
            auto&     thread_current = ctx.thread_currents[tid];
            auto&     thread_scratch = ctx.thread_scratch[tid];
            auto&     thread_metrics = ctx.thread_metrics[tid];

            thread_current.zero_out();
            thread_metrics.reset();

#pragma omp for schedule(static)
            for (auto& block : ctx.particles)
            {
                const uint64_t t0 = ctx.enable_profiling ? pico::perf::read_cpu_ticks() : 0;
                gather.gather_block(block, ctx.fields, ctx.grid, thread_scratch);

                const uint64_t t1 = ctx.enable_profiling ? pico::perf::read_cpu_ticks() : 0;
                pusher.push_block(block, thread_scratch, ctx.dt);
                particle_boundary.apply(block, ctx.grid);

                const uint64_t t2 = ctx.enable_profiling ? pico::perf::read_cpu_ticks() : 0;
                deposit.deposit_block(block, thread_current, ctx.grid, ctx.dt, ctx.particles_per_cell);

                if (ctx.enable_profiling)
                {
                    const uint64_t t3 = pico::perf::read_cpu_ticks();
                    ctx.profiler.add_ticks(pico::perf::Stage::Gather, t1 - t0);
                    ctx.profiler.add_ticks(pico::perf::Stage::Pusher, t2 - t1);
                    ctx.profiler.add_ticks(pico::perf::Stage::Deposit, t3 - t2);
                }
                if (ctx.enable_locality)
                {
                    pico::diagnostics::LocalityProfiler<particle::ParticleBlock<BLOCK_SIZE>>::process_block(block, ctx.grid, thread_metrics);
                }
            }
        }
    }
};

struct GpuStagedPipeline
{
    template <std::size_t BLOCK_SIZE, typename GatherT, typename PusherT, typename ParticleBoundaryT, typename DepositT>
    static void execute(PipelineContext<BLOCK_SIZE>& ctx, GatherT& gather, PusherT& pusher, ParticleBoundaryT& particle_boundary, DepositT& deposit)
    {
        for (auto& tc : ctx.thread_currents)
            tc.zero_out();
        for (auto& tm : ctx.thread_metrics)
            tm.reset();

// Stage 1: Parallel Field Gather & Async GPU Command Encoding
#pragma omp parallel
        {
            const int tid            = omp_get_thread_num();
            auto&     thread_scratch = ctx.thread_scratch[tid];

#pragma omp for schedule(static)
            for (auto& block : ctx.particles)
            {
                const uint64_t t0 = ctx.enable_profiling ? pico::perf::read_cpu_ticks() : 0;
                gather.gather_block(block, ctx.fields, ctx.grid, thread_scratch);

                const uint64_t t1 = ctx.enable_profiling ? pico::perf::read_cpu_ticks() : 0;
                pusher.push_block(block, thread_scratch, ctx.dt);

                if (ctx.enable_profiling)
                {
                    const uint64_t t2 = pico::perf::read_cpu_ticks();
                    ctx.profiler.add_ticks(pico::perf::Stage::Gather, t1 - t0);
                    ctx.profiler.add_ticks(pico::perf::Stage::Pusher, t2 - t1); // Command encoding duration
                }
            }
        }

        // Stage 2: Hardware Sync & Background Callback Execution
        const uint64_t sync_t0 = ctx.enable_profiling ? pico::perf::read_cpu_ticks() : 0;
        pusher.sync();
        if (ctx.enable_profiling)
        {
            const uint64_t sync_t1 = pico::perf::read_cpu_ticks();
            ctx.profiler.add_ticks(pico::perf::Stage::Pusher, sync_t1 - sync_t0); // GPU hardware wait duration
        }

// Stage 3: Post-Sync Particle Boundaries & Current Deposition
#pragma omp parallel
        {
            const int tid            = omp_get_thread_num();
            auto&     thread_current = ctx.thread_currents[tid];
            auto&     thread_metrics = ctx.thread_metrics[tid];

#pragma omp for schedule(static)
            for (auto& block : ctx.particles)
            {
                particle_boundary.apply(block, ctx.grid);

                const uint64_t t0 = ctx.enable_profiling ? pico::perf::read_cpu_ticks() : 0;
                deposit.deposit_block(block, thread_current, ctx.grid, ctx.dt, ctx.particles_per_cell);

                if (ctx.enable_profiling)
                {
                    const uint64_t t1 = pico::perf::read_cpu_ticks();
                    ctx.profiler.add_ticks(pico::perf::Stage::Deposit, t1 - t0);
                }
                if (ctx.enable_locality)
                {
                    pico::diagnostics::LocalityProfiler<particle::ParticleBlock<BLOCK_SIZE>>::process_block(block, ctx.grid, thread_metrics);
                }
            }
        }
    }
};

} // namespace pico::pipeline
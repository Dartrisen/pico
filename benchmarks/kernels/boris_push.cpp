#include "app/VerificationReport.hpp"
#include "data/field/include/field_block.hpp"
#include "data/field/include/field_em.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_system.hpp"
#include "engine/perf/PipelineProfiler.hpp"
#include "kernels/pusher/boris.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace benchmarks
{

void bench_boris_pusher_scaling()
{
    constexpr size_t    BS              = 16;
    std::vector<size_t> particle_counts = {1000, 4000, 8000, 16000, 40000, 100000};
    constexpr size_t    num_iterations  = 100;
    constexpr float     dt              = 0.01f;
    constexpr double    dx              = 0.1;

    Grid grid(64, dx);

    pico::ui::VerificationReport report("Boris Pusher Scaling Benchmark", true, "Throughput Performance Summary");

    for (size_t num_p : particle_counts)
    {
        particle::ParticleSystem<BS> particles(num_p);
        particles.set_active(num_p);

        for (auto& pb : particles)
        {
            for (size_t i = 0; i < pb.activeCount; ++i)
            {
                pb.position_x[i] = 10.0f + static_cast<float>(i) * 0.1f;
                pb.momentum_x[i] = 0.1f;
                pb.momentum_y[i] = 0.05f;
                pb.momentum_z[i] = 0.02f;
                pb.inv_gamma[i]  = 1.0f;
            }
        }

        EMFields<BS> fields(grid);
        fields.E.template set_fields<field::FieldComp::X>(0.1f);
        fields.E.template set_fields<field::FieldComp::Y>(0.05f);
        fields.E.template set_fields<field::FieldComp::Z>(0.02f);
        fields.B.template set_fields<field::FieldComp::X>(0.001f);
        fields.B.template set_fields<field::FieldComp::Y>(0.001f);
        fields.B.template set_fields<field::FieldComp::Z>(0.001f);

        FieldScratch<BS> scratch;

        pico::perf::PipelineProfiler profiler;

        // Warmup
        for (size_t iter = 0; iter < 5; ++iter)
        {
            for (auto& pb : particles)
            {
                for (size_t i = 0; i < pb.activeCount; ++i)
                {
                    scratch.Ex[i] = 0.1f;
                    scratch.Ey[i] = 0.05f;
                    scratch.Ez[i] = 0.02f;
                    scratch.Bx[i] = 0.001f;
                    scratch.By[i] = 0.001f;
                    scratch.Bz[i] = 0.001f;
                }
                kernels::pusher::BorisPusher<BS>::push_block(pb, scratch, dt, 1.0f);
            }
        }

        profiler.reset();

        // Benchmark measurement using PipelineProfiler RAII scope
        for (size_t iter = 0; iter < num_iterations; ++iter)
        {
            for (auto& pb : particles)
            {
                for (size_t i = 0; i < pb.activeCount; ++i)
                {
                    scratch.Ex[i] = 0.1f;
                    scratch.Ey[i] = 0.05f;
                    scratch.Ez[i] = 0.02f;
                    scratch.Bx[i] = 0.001f;
                    scratch.By[i] = 0.001f;
                    scratch.Bz[i] = 0.001f;
                }
                // Automatically measures and accumulates ticks into Stage::Pusher
                auto scope = profiler.time_stage(pico::perf::Stage::Pusher);
                kernels::pusher::BorisPusher<BS>::push_block(pb, scratch, dt, 1.0f);
            }
        }

        double elapsed_ms = profiler.milliseconds(pico::perf::Stage::Pusher);

        uint64_t total_particle_updates = num_p * num_iterations;
        double   particles_per_sec      = (total_particle_updates * 1000.0) / elapsed_ms;

        std::string label = std::to_string(num_p) + " Particles";
        report.add_sci_row(label, particles_per_sec, 2, "p/s");
    }

    report.print();
}

template <size_t BS>
void run_cache_row(pico::ui::VerificationReport& report, size_t total_particles, size_t num_iterations, float dt, const Grid& grid)
{
    particle::ParticleSystem<BS> particles(total_particles);
    particles.set_active(total_particles);

    for (auto& pb : particles)
    {
        for (size_t i = 0; i < pb.activeCount; ++i)
        {
            pb.position_x[i] = 10.0f + static_cast<float>(i) * 0.1f;
            pb.momentum_x[i] = 0.1f;
            pb.momentum_y[i] = 0.05f;
            pb.momentum_z[i] = 0.02f;
            pb.inv_gamma[i]  = 1.0f;
        }
    }

    EMFields<BS> fields(grid);
    fields.E.template set_fields<field::FieldComp::X>(0.1f);
    fields.E.template set_fields<field::FieldComp::Y>(0.05f);
    fields.E.template set_fields<field::FieldComp::Z>(0.02f);
    fields.B.template set_fields<field::FieldComp::X>(0.001f);
    fields.B.template set_fields<field::FieldComp::Y>(0.001f);
    fields.B.template set_fields<field::FieldComp::Z>(0.001f);
    FieldScratch<BS> scratch;

    pico::perf::PipelineProfiler profiler;

    for (size_t iter = 0; iter < num_iterations; ++iter)
    {
        for (auto& pb : particles)
        {
            for (size_t i = 0; i < pb.activeCount; ++i)
            {
                scratch.Ex[i] = 0.1f;
                scratch.Ey[i] = 0.05f;
                scratch.Ez[i] = 0.02f;
                scratch.Bx[i] = 0.001f;
                scratch.By[i] = 0.001f;
                scratch.Bz[i] = 0.001f;
            }
            auto scope = profiler.time_stage(pico::perf::Stage::Pusher);
            kernels::pusher::BorisPusher<BS>::push_block(pb, scratch, dt, 1.0f);
        }
    }

    double   elapsed_ms      = profiler.milliseconds(pico::perf::Stage::Pusher);
    uint64_t total_ops       = total_particles * num_iterations;
    double   ns_per_particle = (elapsed_ms * 1e6) / total_ops;

    bool is_fast_enough = (ns_per_particle < 15.0);

    std::string label = "Block Size (BS=" + std::to_string(BS) + ")";
    report.add_fixed_row(label, ns_per_particle, 2, "ns/p", is_fast_enough ? "\033[32m" : "\033[31m");
}

void bench_boris_pusher_cache_sensitivity()
{
    constexpr size_t total_particles = 10000;
    constexpr size_t num_iterations  = 50;
    constexpr float  dt              = 0.01f;
    constexpr double dx              = 0.1;

    Grid grid(64, dx);

    pico::ui::VerificationReport report("Boris Pusher Cache Sensitivity", true, "Block Size Latency Summary");

    run_cache_row<16>(report, total_particles, num_iterations, dt, grid);
    run_cache_row<32>(report, total_particles, num_iterations, dt, grid);
    run_cache_row<64>(report, total_particles, num_iterations, dt, grid);
    run_cache_row<128>(report, total_particles, num_iterations, dt, grid);

    report.print();
}

} // namespace benchmarks

int main()
{
    benchmarks::bench_boris_pusher_scaling();
    benchmarks::bench_boris_pusher_cache_sensitivity();
    return 0;
}
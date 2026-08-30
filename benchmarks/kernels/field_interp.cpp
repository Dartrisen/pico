#include "app/VerificationReport.hpp"
#include "data/field/include/field_em.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_system.hpp"
#include "engine/perf/PipelineProfiler.hpp"
#include "kernels/gather/gather.hpp"
#include "kernels/shapes/spline.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace benchmarks
{

/**
 * @brief Field interpolation: measure gather operation scaling across particle counts
 */
void bench_field_gather_scaling()
{
    constexpr size_t    BS              = 16;
    std::vector<size_t> particle_counts = {1000, 4000, 8000, 16000, 40000};
    constexpr size_t    num_iterations  = 100;
    constexpr double    dx              = 0.1;

    Grid grid(128, dx);

    pico::ui::VerificationReport report("Field Gather Scaling Benchmark", true, "Interpolation Throughput Summary");

    particle::ParticleSystem<BS> particles(particle_counts.back());
    float                        cell_size = grid.cell_size();

    for (size_t num_p : particle_counts)
    {
        particles.set_active(num_p);
        particles.init_positions_uniform(grid);

        // Create fields with realistic pattern
        EMFields<BS> fields(grid);
        fields.E.set_fields<field::FieldComp::X>(0.1f);
        fields.E.set_fields<field::FieldComp::Y>(0.05f);
        fields.E.set_fields<field::FieldComp::Z>(0.02f);
        fields.B.set_fields<field::FieldComp::X>(0.01f);
        fields.B.set_fields<field::FieldComp::Y>(0.01f);
        fields.B.set_fields<field::FieldComp::Z>(0.01f);
        FieldScratch<BS> scratch;

        pico::perf::PipelineProfiler profiler;

        // Warmup
        for (size_t iter = 0; iter < 3; ++iter)
        {
            for (auto& pb : particles)
            {
                kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid, scratch);
            }
        }

        profiler.reset();

        // Benchmark using PipelineProfiler Stage::Gather
        for (size_t iter = 0; iter < num_iterations; ++iter)
        {
            for (auto& pb : particles)
            {
                auto scope = profiler.time_stage(pico::perf::Stage::Gather);
                kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid, scratch);
            }
        }

        double   elapsed_ms        = profiler.milliseconds(pico::perf::Stage::Gather);
        uint64_t total_ops         = num_p * num_iterations;
        double   particles_per_sec = (total_ops * 1000.0) / elapsed_ms;

        std::string label = std::to_string(num_p) + " Particles";
        report.add_sci_row(label, particles_per_sec, 2, "p/s");
    }

    report.print();
}

/**
 * @brief Memory bandwidth and timing estimate for gather operations
 */
void bench_field_gather_bandwidth()
{
    constexpr size_t BS             = 16;
    constexpr size_t num_particles  = 20000;
    constexpr size_t num_iterations = 100;
    constexpr double dx             = 0.1;

    Grid grid(128, dx);

    particle::ParticleSystem<BS> particles(num_particles);
    particles.set_active(num_particles);
    particles.init_positions_uniform(grid);

    EMFields<BS> fields(grid);
    fields.E.set_fields<field::FieldComp::X>(1.0f);
    fields.E.set_fields<field::FieldComp::Y>(0.5f);
    fields.E.set_fields<field::FieldComp::Z>(0.2f);
    fields.B.set_fields<field::FieldComp::X>(0.01f);
    fields.B.set_fields<field::FieldComp::Y>(0.01f);
    fields.B.set_fields<field::FieldComp::Z>(0.01f);
    FieldScratch<BS> scratch;

    pico::perf::PipelineProfiler profiler;

    for (size_t iter = 0; iter < num_iterations; ++iter)
    {
        for (auto& pb : particles)
        {
            auto scope = profiler.time_stage(pico::perf::Stage::Gather);
            kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid, scratch);
        }
    }

    double   elapsed_sec    = profiler.seconds(pico::perf::Stage::Gather);
    uint64_t bytes_accessed = num_particles * num_iterations * 20; // Conservative estimate per particle
    double   bandwidth_gbps = (static_cast<double>(bytes_accessed) / elapsed_sec) / 1e9;

    pico::ui::VerificationReport report("Field Gather Bandwidth", true, "Memory Bandwidth Estimate");
    report.add_fixed_row("Effective Bandwidth", bandwidth_gbps, 2, "GB/s");
    report.add_fixed_row("Total Execution Time", elapsed_sec, 3, "sec");
    report.add_fixed_row("Particle Updates", (num_particles * num_iterations) / 1e6, 2, "M updates");
    report.print();
}

} // namespace benchmarks

int main()
{
    benchmarks::bench_field_gather_scaling();
    benchmarks::bench_field_gather_bandwidth();

    return 0;
}
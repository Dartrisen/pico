#include "app/VerificationReport.hpp"
#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_system.hpp"
#include "engine/perf/PipelineProfiler.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace benchmarks
{

/**
 * @brief Sequential read benchmark: measure peak sequential bandwidth baseline
 */
void bench_sequential_bandwidth()
{
    pico::ui::VerificationReport report("Sequential Memory Bandwidth", true, "Baseline Sequential Read Performance");
    std::vector<size_t>          sizes_mb = {1, 4, 8, 32, 64, 256};
    pico::perf::PipelineProfiler profiler;

    for (size_t mb : sizes_mb)
    {
        size_t             num_elements = (mb * 1024 * 1024) / sizeof(float);
        std::vector<float> data(num_elements);

        for (size_t i = 0; i < num_elements; ++i)
            data[i] = static_cast<float>(i);

        // Warmup
        float sum = 0.0f;
        for (size_t i = 0; i < num_elements; ++i)
            sum += data[i];
        asm volatile("" : "+m"(sum));

        profiler.reset();
        {
            auto scope = profiler.time_stage(pico::perf::Stage::FieldSolver);
            sum        = 0.0f;
            for (size_t iter = 0; iter < 10; ++iter)
            {
                for (size_t i = 0; i < num_elements; ++i)
                    sum += data[i];
            }
            asm volatile("" : "+m"(sum));
        }

        double elapsed_sec       = profiler.seconds(pico::perf::Stage::FieldSolver);
        double bytes_transferred = static_cast<double>(num_elements) * 10 * sizeof(float);
        double bandwidth_gbps    = (bytes_transferred / elapsed_sec) / 1e9;

        std::string label = "Size: " + std::to_string(mb) + " MB";
        report.add_fixed_row(label, bandwidth_gbps, 2, "GB/s");
    }

    report.print();
}

/**
 * @brief Random access benchmark: measure memory latency via random reads
 */
void bench_random_bandwidth()
{
    pico::ui::VerificationReport report("Random Memory Access", true, "Latency-Bound Cache Miss Penalty");
    std::vector<size_t>          sizes_kb = {16, 64, 256, 1024, 4096};
    pico::perf::PipelineProfiler profiler;

    for (size_t kb : sizes_kb)
    {
        size_t              num_elements = (kb * 1024) / sizeof(float);
        std::vector<float>  data(num_elements);
        std::vector<size_t> indices(num_elements);

        for (size_t i = 0; i < num_elements; ++i)
        {
            data[i]    = static_cast<float>(i);
            indices[i] = i;
        }
        std::random_shuffle(indices.begin(), indices.end());

        // Warmup
        float sum = 0.0f;
        for (size_t i = 0; i < num_elements; ++i)
            sum += data[indices[i]];
        asm volatile("" : "+m"(sum));

        profiler.reset();
        constexpr size_t iterations = 100;
        {
            auto scope = profiler.time_stage(pico::perf::Stage::Gather);
            sum        = 0.0f;
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < num_elements; ++i)
                    sum += data[indices[i]];
            }
            asm volatile("" : "+m"(sum));
        }

        double elapsed_sec = profiler.seconds(pico::perf::Stage::Gather);
        double ops_per_sec = (static_cast<double>(num_elements) * iterations) / elapsed_sec;
        double latency_ns  = 1.0 / (ops_per_sec / 1e9);

        std::string label = "Size: " + std::to_string(kb) + " KB";
        report.add_fixed_row(label, latency_ns, 3, "ns latency");
    }

    report.print();
}

/**
 * @brief Particle position array: measure SOA bandwidth
 */
void bench_particle_bandwidth()
{
    pico::ui::VerificationReport report("Particle Position Array Bandwidth", true, "Structure-of-Arrays (SoA) Layout");
    constexpr size_t             BS            = 16;
    std::vector<size_t>          num_particles = {1000, 10000, 50000};
    pico::perf::PipelineProfiler profiler;

    particle::ParticleSystem<BS> particles(num_particles.back());
    for (size_t np : num_particles)
    {
        particles.set_active(np);

        size_t idx = 0;
        for (auto& pb : particles)
        {
            for (size_t i = 0; i < pb.activeCount; ++i)
            {
                pb.position_x[i] = 10.0f + static_cast<float>(idx++);
            }
        }

        profiler.reset();
        float            sum        = 0.0f;
        constexpr size_t iterations = 1000;
        {
            auto scope = profiler.time_stage(pico::perf::Stage::Deposit);
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (auto& pb : particles)
                {
                    for (size_t i = 0; i < pb.activeCount; ++i)
                    {
                        sum += pb.position_x[i];
                    }
                }
            }
            asm volatile("" : "+m"(sum));
        }

        double elapsed_sec    = profiler.seconds(pico::perf::Stage::Deposit);
        double bytes          = static_cast<double>(np) * 4 * iterations;
        double bandwidth_gbps = (bytes / elapsed_sec) / 1e9;

        std::string label = "Particles: " + std::to_string(np);
        report.add_fixed_row(label, bandwidth_gbps, 2, "GB/s");
    }

    report.print();
}

/**
 * @brief Field grid: measure strided access bandwidth
 */
void bench_field_bandwidth()
{
    pico::ui::VerificationReport report("Field Grid Access", true, "Strided and Scattered Grid Access Pattern");
    constexpr size_t             BS         = 16;
    constexpr double             dx         = 0.1;
    std::vector<size_t>          grid_sizes = {32, 64, 128, 256};
    pico::perf::PipelineProfiler profiler;

    for (size_t gs : grid_sizes)
    {
        Grid            grid(gs, dx);
        FieldSystem<BS> field(grid);

        profiler.reset();
        float            sum        = 0.0f;
        constexpr size_t iterations = 100;
        {
            auto scope = profiler.time_stage(pico::perf::Stage::Pusher);
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t b = 0; b < field.num_blocks(); ++b)
                {
                    const auto& block = field.block(b);
                    const auto& data  = block.component<field::FieldComp::X>();
                    for (size_t i = 0; i < BS; ++i)
                    {
                        sum += data[i];
                    }
                }
            }
            asm volatile("" : "+m"(sum));
        }

        double elapsed_sec    = profiler.seconds(pico::perf::Stage::Pusher);
        double total_floats   = static_cast<double>(field.num_blocks()) * BS * iterations;
        double bytes          = total_floats * 4;
        double bandwidth_gbps = (bytes / elapsed_sec) / 1e9;

        std::string label = "Grid: " + std::to_string(gs);
        report.add_fixed_row(label, bandwidth_gbps, 2, "GB/s");
    }

    report.print();
}

} // namespace benchmarks

int main()
{
    benchmarks::bench_sequential_bandwidth();
    benchmarks::bench_random_bandwidth();
    benchmarks::bench_particle_bandwidth();
    benchmarks::bench_field_bandwidth();

    return 0;
}
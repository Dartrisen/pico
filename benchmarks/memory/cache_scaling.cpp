#include "app/VerificationReport.hpp"
#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/perf/PipelineProfiler.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace benchmarks
{

/**
 * @brief Cache hierarchy scaling: measure performance vs working set size
 */
void bench_cache_scaling()
{
    pico::ui::VerificationReport report("Cache Hierarchy Scaling", true, "Working Set Size & Cache Effects");
    std::vector<size_t>          sizes_kb = {1, 4, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    pico::perf::PipelineProfiler profiler;

    for (size_t kb : sizes_kb)
    {
        size_t             num_floats = (kb * 1024) / sizeof(float);
        std::vector<float> data(num_floats);

        for (size_t i = 0; i < num_floats; ++i)
            data[i] = 1.0f + static_cast<float>(i % 16);

        // Warmup
        float sum = 0.0f;
        for (size_t i = 0; i < num_floats; ++i)
            sum += data[i];
        asm volatile("" : "+m"(sum));

        profiler.reset();
        constexpr size_t iterations = 1000;

        {
            auto scope = profiler.time_stage(pico::perf::Stage::FieldSolver);
            sum        = 0.0f;
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < num_floats; ++i)
                    sum += data[i];
            }
            asm volatile("" : "+m"(sum));
        }

        double elapsed_sec      = profiler.seconds(pico::perf::Stage::FieldSolver);
        double operations       = static_cast<double>(num_floats) * iterations;
        double giga_ops_per_sec = (operations / elapsed_sec) / 1e9;

        const char* working_set = kb <= 32 ? "small" : kb <= 256 ? "medium" : kb <= 8192 ? "large" : "DRAM-scale";

        std::string label = "Size: " + std::to_string(kb) + " KB [" + working_set + "]";
        report.add_fixed_row(label, giga_ops_per_sec, 2, "Gops/s");
    }

    report.print();
}

/**
 * @brief Field block size optimization using VerificationReport
 */
void bench_field_block_size_scaling()
{
    pico::ui::VerificationReport report("Field Block Size Sensitivity", true, "Grid Impact on Cache");
    constexpr size_t             BS = 8;
    constexpr size_t             GS = 64;
    constexpr double             dx = 0.1;

    Grid                         grid(GS, dx);
    FieldSystem<BS>              field(grid);
    pico::perf::PipelineProfiler profiler;

    std::vector<size_t> hypothetical_block_sizes = {16, 32, 64, 128};

    for (size_t hypothetical_bs : hypothetical_block_sizes)
    {
        double scale_factor = static_cast<double>(hypothetical_bs) / BS;

        profiler.reset();
        float            sum        = 0.0f;
        constexpr size_t iterations = 100;

        {
            auto scope = profiler.time_stage(pico::perf::Stage::Deposit);
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

        double elapsed_sec    = profiler.seconds(pico::perf::Stage::Deposit);
        double adjusted_sec   = elapsed_sec * scale_factor;
        double blocks_per_sec = (static_cast<double>(field.num_blocks()) * iterations) / adjusted_sec;

        std::string label = "Block Size: " + std::to_string(hypothetical_bs);
        report.add_fixed_row(label, blocks_per_sec, 2, "blocks/s");
    }

    report.print();
}

/**
 * @brief Stride impact using VerificationReport
 */
void bench_stride_impact()
{
    pico::ui::VerificationReport report("Stride Impact on Cache Efficiency", true, "Non-Unit Stride Degradation");
    size_t                       total_elements = 10000;
    std::vector<float>           data(total_elements);

    for (size_t i = 0; i < total_elements; ++i)
        data[i] = static_cast<float>(i % 256);

    std::vector<size_t>          strides = {1, 2, 4, 8, 16, 32, 64};
    pico::perf::PipelineProfiler profiler;

    for (size_t stride : strides)
    {
        profiler.reset();
        float            sum        = 0.0f;
        constexpr size_t iterations = 1000;

        {
            auto scope = profiler.time_stage(pico::perf::Stage::Gather);
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < total_elements; i += stride)
                {
                    sum += data[i];
                }
            }
            asm volatile("" : "+m"(sum));
        }

        double elapsed_sec      = profiler.seconds(pico::perf::Stage::Gather);
        size_t ops              = (total_elements / stride) * iterations;
        double giga_ops_per_sec = (static_cast<double>(ops) / elapsed_sec) / 1e9;

        std::string label = "Stride: " + std::to_string(stride);
        report.add_fixed_row(label, giga_ops_per_sec, 2, "Gops/s");
    }

    report.print();
}

} // namespace benchmarks

int main()
{
    benchmarks::bench_cache_scaling();
    benchmarks::bench_field_block_size_scaling();
    benchmarks::bench_stride_impact();

    return 0;
}
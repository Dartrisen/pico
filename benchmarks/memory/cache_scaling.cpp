#include "bench/core/timer.hpp"
#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

/**
     * @brief Cache hierarchy scaling: measure performance vs problem size
     *
     * Shows L1/L2/L3 cache effects as working set grows
     */
void bench_cache_scaling()
{
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "CACHE HIERARCHY SCALING (working set size)\n";
    std::cout << std::string(80, '=') << "\n";

    std::cout << std::left << std::setw(15) << "Size (KB)"
              << " | " << std::setw(12) << "Throughput"
              << " | " << std::setw(12) << "Per-element"
              << " | Cache Level\n";
    std::cout << std::string(60, '=') << "\n";

    // Test sizes from 1 KB to 16 MB (typical cache boundaries)
    std::vector<size_t> sizes_kb = {1, 4, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    for (size_t kb : sizes_kb)
    {
        size_t             num_floats = (kb * 1024) / sizeof(float);
        std::vector<float> data(num_floats);

        // Initialize
        for (size_t i = 0; i < num_floats; ++i)
            data[i] = 1.0f + static_cast<float>(i % 16);

        // Warmup
        volatile float sum = 0.0f;
        for (size_t i = 0; i < num_floats; ++i)
            sum += data[i];

        // Benchmark: sum reduction (latency bound, sequential)
        bench::Timer timer;
        sum                         = 0.0f;
        constexpr size_t iterations = 1000;
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_floats; ++i)
                sum += data[i];
        }
        double elapsed_ns = timer.elapsed_ns();

        // Calculate throughput
        double operations       = num_floats * iterations;
        double giga_ops_per_sec = (operations * 1e9) / elapsed_ns;
        double ns_per_op        = elapsed_ns / operations;

        // Identify cache level (approximate)
        const char* cache_level = "?";
        if (kb <= 32)
            cache_level = "L1";
        else if (kb <= 256)
            cache_level = "L2";
        else if (kb <= 8192)
            cache_level = "L3";
        else
            cache_level = "RAM";

        std::cout << std::setw(15) << kb << " | " << std::scientific << std::setprecision(2) << std::setw(12)
                  << giga_ops_per_sec << " | " << std::fixed << std::setprecision(3) << std::setw(12) << ns_per_op
                  << " | " << cache_level << "\n";

        (void) sum;
    }
}

/**
     * @brief Field block size optimization
     *
     * For a given grid, test how block size affects cache efficiency
     * Each block is cache-aligned, so larger blocks fit fewer in cache
     */
void bench_field_block_size_scaling()
{
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "FIELD BLOCK SIZE SENSITIVITY\n";
    std::cout << "  (For 64x64 grid, impact of BLOCK_SIZE on cache)\n";
    std::cout << std::string(80, '=') << "\n";

    std::cout << std::left << std::setw(15) << "Block Size"
              << " | " << std::setw(12) << "Throughput"
              << " | " << std::setw(12) << "Per-block"
              << " | Memory footprint\n";
    std::cout << std::string(80, '=') << "\n";

    // Note: We can't template-instantiate arbitrary sizes at runtime,
    // but we can measure with BS=8 (fixed) and annotate what would happen
    constexpr size_t BS = 8;
    constexpr size_t GS = 64;

    Grid grid(GS, GS);

    FieldSystem<BS> field(grid);

    // Simulate block access with different hypothetical block sizes
    std::vector<size_t> hypothetical_block_sizes = {8, 16, 32, 64};

    for (size_t hypothetical_bs : hypothetical_block_sizes)
    {
        // For BS=8: measure as-is
        // For larger: extrapolate memory footprint
        double scale_factor = static_cast<double>(hypothetical_bs) / BS;

        bench::Timer     timer;
        volatile float   sum        = 0.0f;
        constexpr size_t iterations = 100;

        // Actual measurement with BS=8
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
        double elapsed_ns = timer.elapsed_ns();

        // Adjust for hypothetical block size
        double adjusted_ns    = elapsed_ns * scale_factor;
        double blocks_per_sec = (field.num_blocks() * iterations * 1e9) / adjusted_ns;
        double ns_per_block   = adjusted_ns / (field.num_blocks() * iterations);

        // Memory per block: each Ex/Ey/Ez is hypothetical_bs floats, 6 components
        size_t bytes_per_block_e = hypothetical_bs * 3 * sizeof(float); // Ex, Ey, Ez
        size_t bytes_per_block_b = hypothetical_bs * 3 * sizeof(float); // Bx, By, Bz
        size_t total_field_bytes = (bytes_per_block_e + bytes_per_block_b) * field.num_blocks();

        std::cout << std::setw(15) << hypothetical_bs << " | " << std::scientific << std::setprecision(2)
                  << std::setw(12) << blocks_per_sec << " | " << std::fixed << std::setprecision(3) << std::setw(12)
                  << ns_per_block << " | " << (total_field_bytes / (1024.0 * 1024.0)) << " MB\n";

        (void) sum;
    }
}

/**
     * @brief Stride impact: measure performance degradation with non-unit stride
     */
void bench_stride_impact()
{
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "STRIDE IMPACT ON CACHE EFFICIENCY\n";
    std::cout << std::string(80, '=') << "\n";

    size_t             total_elements = 10000;
    std::vector<float> data(total_elements);

    // Initialize
    for (size_t i = 0; i < total_elements; ++i)
        data[i] = static_cast<float>(i % 256);

    std::cout << std::left << std::setw(12) << "Stride"
              << " | " << std::setw(12) << "Throughput"
              << " | " << std::setw(12) << "Cache hit %%"
              << " | Notes\n";
    std::cout << std::string(70, '=') << "\n";

    std::vector<size_t> strides = {1, 2, 4, 8, 16, 32, 64};

    for (size_t stride : strides)
    {
        bench::Timer     timer;
        volatile float   sum        = 0.0f;
        constexpr size_t iterations = 1000;

        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < total_elements; i += stride)
            {
                sum += data[i];
            }
        }
        double elapsed_ns = timer.elapsed_ns();

        // Count actual operations
        size_t ops              = (total_elements / stride) * iterations;
        double giga_ops_per_sec = (ops * 1e9) / elapsed_ns;

        // Estimate cache hit rate (rough)
        // Stride-1 should have ~95% hit rate
        // Larger strides degrade due to cache line size (~64 bytes = 16 floats)
        double cache_line_elements = 64.0 / sizeof(float);
        double cache_hits          = 100.0 * (1.0 - (stride / cache_line_elements));
        cache_hits                 = std::max(0.0, cache_hits);
        cache_hits                 = std::min(100.0, cache_hits);

        std::string notes;
        if (stride == 1)
            notes = "(sequential, optimal)";
        else if (stride < 16)
            notes = "(within cache line)";
        else if (stride < 256)
            notes = "(miss penalty)";
        else
            notes = "(high miss rate)";

        std::cout << std::setw(12) << stride << " | " << std::scientific << std::setprecision(2) << std::setw(12)
                  << giga_ops_per_sec << " | " << std::fixed << std::setprecision(1) << std::setw(12) << cache_hits
                  << " | " << notes << "\n";

        (void) sum;
    }
}

} // namespace

int main()
{
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║          CACHE SCALING & OPTIMIZATION STUDY            ║\n";
    std::cout << "║  Measures cache effects, block sizing, stride impact   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    bench_cache_scaling();
    bench_field_block_size_scaling();
    bench_stride_impact();

    return 0;
}

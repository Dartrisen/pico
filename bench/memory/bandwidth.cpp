#include "bench/core/timer.hpp"
#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_system.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

    /**
     * @brief Sequential read benchmark: measure peak sequential bandwidth
     *
     * Baseline for comparison: how fast can we read memory sequentially?
     */
    void bench_sequential_bandwidth()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "SEQUENTIAL MEMORY BANDWIDTH (baseline)\n";
        std::cout << std::string(80, '=') << "\n";

        std::vector<size_t> sizes_mb = {1, 4, 8, 32, 64, 256};

        for (size_t mb : sizes_mb)
        {
            size_t             num_elements = (mb * 1024 * 1024) / sizeof(float);
            std::vector<float> data(num_elements);

            // Initialize
            for (size_t i = 0; i < num_elements; ++i)
                data[i] = static_cast<float>(i);

            // Warmup
            volatile float sum = 0.0f;
            for (size_t i = 0; i < num_elements; ++i)
                sum += data[i];

            // Benchmark: sequential read
            bench::Timer timer;
            sum = 0.0f;
            for (size_t iter = 0; iter < 10; ++iter)
            {
                for (size_t i = 0; i < num_elements; ++i)
                    sum += data[i];
            }
            double elapsed_sec = timer.elapsed_sec();

            double bytes_transferred = num_elements * 10 * sizeof(float);
            double bandwidth_gbps    = (bytes_transferred / elapsed_sec) / 1e9;

            std::cout << "Size: " << std::setw(4) << mb << " MB | Bandwidth: " << std::fixed << std::setprecision(2)
                      << std::setw(6) << bandwidth_gbps << " GB/s\n";
        }
    }

    /**
     * @brief Random access benchmark: measure memory latency via random reads
     *
     * Shows cache miss penalty
     */
    void bench_random_bandwidth()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "RANDOM MEMORY ACCESS (latency-bound)\n";
        std::cout << std::string(80, '=') << "\n";

        std::vector<size_t> sizes_kb = {16, 64, 256, 1024, 4096};

        for (size_t kb : sizes_kb)
        {
            size_t              num_elements = (kb * 1024) / sizeof(float);
            std::vector<float>  data(num_elements);
            std::vector<size_t> indices(num_elements);

            // Initialize with random permutation
            for (size_t i = 0; i < num_elements; ++i)
            {
                data[i]    = static_cast<float>(i);
                indices[i] = i;
            }
            std::random_shuffle(indices.begin(), indices.end());

            // Warmup
            volatile float sum = 0.0f;
            for (size_t i = 0; i < num_elements; ++i)
                sum += data[indices[i]];

            // Benchmark
            bench::Timer timer;
            sum                         = 0.0f;
            constexpr size_t iterations = 100;
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < num_elements; ++i)
                    sum += data[indices[i]];
            }
            double elapsed_sec = timer.elapsed_sec();

            double ops_per_sec = (num_elements * iterations) / elapsed_sec;

            std::cout << "Size: " << std::setw(5) << kb << " KB | Accesses/sec: " << std::scientific
                      << std::setprecision(3) << std::setw(10) << ops_per_sec << " | "
                      << "Latency: " << std::fixed << std::setprecision(3) << std::setw(6)
                      << (1.0 / (ops_per_sec / 1e9)) << " ns\n";

            (void) sum; // suppress warning
        }
    }

    /**
     * @brief Particle position array: measure SOA bandwidth
     *
     * Structure-of-Arrays: particles laid out sequentially
     */
    void bench_particle_bandwidth()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "PARTICLE POSITION ARRAY BANDWIDTH (SOA)\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t    BS            = 8;
        std::vector<size_t> num_particles = {1000, 10000, 50000};

        particle::ParticleSystem<BS> particles(num_particles.back());
        for (size_t np : num_particles)
        {
            particles.set_active(np);

            // Initialize positions
            size_t idx = 0;
            for (auto& pb : particles)
            {
                for (size_t i = 0; i < pb.activeCount; ++i)
                {
                    pb.position_x[i] = 10.0f + static_cast<float>(idx++);
                }
            }

            // Benchmark: read all positions sequentially
            bench::Timer     timer;
            volatile float   sum        = 0.0f;
            constexpr size_t iterations = 1000;
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
            double elapsed_sec = timer.elapsed_sec();

            // Bytes: np * 4 bytes * iterations
            double bytes          = np * 4 * iterations * 1e-9;
            double bandwidth_gbps = bytes / elapsed_sec;

            std::cout << "Particles: " << std::setw(6) << np << " | Bandwidth: " << std::fixed << std::setprecision(2)
                      << std::setw(6) << bandwidth_gbps << " GB/s\n";
        }
    }

    /**
     * @brief Field grid: measure strided access bandwidth
     *
     * Fields are laid out in a grid; particles access scattered locations
     */
    void bench_field_bandwidth()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "FIELD GRID ACCESS (strided, scattered)\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t    BS         = 8;
        std::vector<size_t> grid_sizes = {32, 64, 128, 256};

        for (size_t gs : grid_sizes)
        {
            Grid grid(gs, gs);

            FieldSystem<BS> field(grid);

            // Benchmark: linear scan of field blocks
            bench::Timer     timer;
            volatile float   sum        = 0.0f;
            constexpr size_t iterations = 100;
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
            double elapsed_sec = timer.elapsed_sec();

            size_t total_floats   = field.num_blocks() * BS * iterations;
            double bytes          = total_floats * 4 * 1e-9;
            double bandwidth_gbps = bytes / elapsed_sec;

            std::cout << "Grid: " << std::setw(3) << gs << "x" << std::setw(3) << gs << " | Bandwidth: " << std::fixed
                      << std::setprecision(2) << std::setw(6) << bandwidth_gbps << " GB/s\n";

            (void) sum;
        }
    }

} // namespace

int main()
{
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║        MEMORY BANDWIDTH CHARACTERIZATION               ║\n";
    std::cout << "║  Sequential, random, and access pattern analysis       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    bench_sequential_bandwidth();
    bench_random_bandwidth();
    bench_particle_bandwidth();
    bench_field_bandwidth();

    return 0;
}

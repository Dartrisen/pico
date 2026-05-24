#include "data/field/include/field_em.hpp"
#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"
#include "data/particle/include/particle_system.hpp"
#include "kernels/pusher/boris.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

/**
 * @brief Simple RAII timer for performance measurement
 */
class PerfTimer
{
public:
    using Clock = std::chrono::high_resolution_clock;

    PerfTimer() : start_(Clock::now()) {}

    double elapsed_ms() const
    {
        auto end      = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }

    double elapsed_us() const
    {
        auto end      = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count();
    }

private:
    Clock::time_point start_;
};

/**
 * @brief Benchmark result structure
 */
struct BenchmarkResult
{
    const char* name;
    double      time_ms;
    size_t      particles;
    double      throughput_particles_per_ms;
    double      throughput_gparticles_per_sec;

    void print() const
    {
        std::cout << std::left << std::setw(50) << name << " | Time: " << std::fixed << std::setprecision(3)
                  << std::setw(8) << time_ms << " ms"
                  << " | Particles: " << std::setw(8) << particles << " | Throughput: " << std::setw(8)
                  << throughput_gparticles_per_sec << " Gp/s\n";
    }
};

/**
 * @brief Test 1: Heavy particle push - Boris algorithm with varying block counts
 *
 * This benchmark stresses:
 * - Sequential memory access patterns (position, momentum, weight)
 * - Cache efficiency with different dataset sizes
 * - Impact of block size and alignment on performance
 */
void benchmark_boris_push_heavy_load()
{
    std::cout << "\n=== Test 1: Boris Pusher - Heavy Load (Real Data Structures) ===\n";

    std::vector<size_t> num_particles_list = {800, 4000, 8000, 40000};
    constexpr size_t    bs                 = 8;

    Grid grid(64, 64);

    for (size_t num_particles : num_particles_list)
    {
        std::cout << "\n--- " << num_particles << " particles (BS=" << bs << ") ---\n";

        // Use real ParticleSystem
        particle::ParticleSystem<bs> particles(num_particles);
        particles.set_active(num_particles);

        // Initialize particles with realistic data
        for (auto& pb : particles)
        {
            for (size_t i = 0; i < pb.activeCount; ++i)
            {
                pb.position_x[i] = static_cast<float>(i) * 0.1f;
                pb.momentum_x[i] = 1.0f;
                pb.momentum_y[i] = 0.5f;
                pb.momentum_z[i] = 0.1f;
                // mass, charge, weight already initialized by ParticleSystem
            }
        }

        // Create field scratch
        FieldScratch<bs> fs;
        fs.clear();

        constexpr int iterations = 100;
        float         dt         = 0.01f;

        // Warmup
        for (int it = 0; it < 5; ++it)
        {
            for (auto& pb : particles)
            {
                kernels::pusher::BorisPusher<bs>::push_block(pb, fs, dt);
            }
        }

        // Benchmark
        PerfTimer timer;
        for (int it = 0; it < iterations; ++it)
        {
            for (auto& pb : particles)
            {
                kernels::pusher::BorisPusher<bs>::push_block(pb, fs, dt);
            }
        }
        double elapsed_ms = timer.elapsed_ms();

        BenchmarkResult result{"Boris Push (Real)", elapsed_ms / iterations, num_particles,
                               num_particles / (elapsed_ms / iterations),
                               (num_particles / (elapsed_ms / iterations)) / 1e6};

        result.print();
    }
}

/**
 * @brief Test 2: Memory bandwidth test - Sequential reads and writes
 *
 * This benchmark measures:
 * - Raw memory bandwidth of particle data
 * - Effects of cache line conflicts
 * - Impact of alignment and access pattern
 */
void benchmark_memory_bandwidth()
{
    std::cout << "\n=== Test 2: Memory Bandwidth - Sequential Access ===\n";

    constexpr size_t num_particles = 10000000; // 10M particles
    constexpr size_t bs            = 64;

    // Create aligned arrays like ParticleBlock does
    std::vector<float> position_x(num_particles);
    std::vector<float> momentum_x(num_particles);
    std::vector<float> momentum_y(num_particles);
    std::vector<float> momentum_z(num_particles);
    std::vector<float> weight(num_particles);

    // Initialize
    for (size_t i = 0; i < num_particles; ++i)
    {
        position_x[i] = i * 0.1f;
        momentum_x[i] = 1.0f;
        momentum_y[i] = 0.5f;
        momentum_z[i] = 0.1f;
        weight[i]     = 1.0f;
    }

    // Test 1: Read-only sequential
    {
        PerfTimer timer;
        float     sum = 0.0f;
        for (int rep = 0; rep < 10; ++rep)
        {
            for (size_t i = 0; i < num_particles; ++i)
            {
                sum += position_x[i] + momentum_x[i] + momentum_y[i] + momentum_z[i];
            }
        }
        volatile float sink           = sum; // Prevent optimization
        double         elapsed_ms     = timer.elapsed_ms();
        double         bytes_accessed = num_particles * 4 * sizeof(float) * 10;
        double         bandwidth_gb_s = (bytes_accessed / (1024 * 1024 * 1024)) / (elapsed_ms / 1000.0);

        std::cout << "Read-only sequential (4 arrays): " << std::fixed << std::setprecision(2) << bandwidth_gb_s
                  << " GB/s\n";
    }

    // Test 2: Write-intensive
    {
        PerfTimer timer;
        for (int rep = 0; rep < 10; ++rep)
        {
            for (size_t i = 0; i < num_particles; ++i)
            {
                momentum_x[i] += position_x[i] * 0.01f;
                momentum_y[i] += position_x[i] * 0.01f;
                momentum_z[i] += position_x[i] * 0.01f;
            }
        }
        double elapsed_ms     = timer.elapsed_ms();
        double bytes_accessed = num_particles * 5 * sizeof(float) * 10;
        double bandwidth_gb_s = (bytes_accessed / (1024 * 1024 * 1024)) / (elapsed_ms / 1000.0);

        std::cout << "Read-write mixed (5 arrays): " << std::fixed << std::setprecision(2) << bandwidth_gb_s
                  << " GB/s\n";
    }
}

/**
 * @brief Test 3: Cache efficiency - Working set scalability
 *
 * Measures how performance degrades as working set grows
 * This is critical for understanding memory layout impact
 */
void benchmark_cache_efficiency()
{
    std::cout << "\n=== Test 3: Cache Efficiency - Working Set Scaling ===\n";

    std::vector<size_t> working_set_sizes = {
            1000,    // ~1KB per particle (4 floats * 4 bytes)
            10000,   // ~100KB
            100000,  // ~1MB (L3 cache ~ 8MB)
            1000000, // ~10MB
            10000000 // ~100MB
    };

    std::cout << std::left << std::setw(20) << "Working Set" << std::setw(20) << "Bandwidth (GB/s)" << std::setw(20)
              << "Latency (cycles/iter)\n";
    std::cout << std::string(60, '-') << "\n";

    for (size_t size : working_set_sizes)
    {
        std::vector<float> data(size);
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = static_cast<float>(i) * 0.1f;
        }

        // Sequential read bandwidth
        PerfTimer     timer;
        constexpr int iterations = 1000;
        float         sum        = 0.0f;

        for (int it = 0; it < iterations; ++it)
        {
            for (size_t i = 0; i < size; ++i)
            {
                sum += data[i];
            }
        }
        volatile float sink = sum; // Prevent optimization

        double elapsed_us     = timer.elapsed_us();
        double bytes          = size * sizeof(float) * iterations;
        double bandwidth_gb_s = (bytes / (1024 * 1024 * 1024)) / (elapsed_us / 1e6);

        std::cout << std::left << std::setw(20) << size << " elements" << std::setw(20) << std::fixed
                  << std::setprecision(2) << bandwidth_gb_s << std::setw(20) << std::fixed << std::setprecision(1)
                  << (elapsed_us / iterations) << "\n";
    }
}

/**
 * @brief Mock unaligned particle block (no 64-byte alignment)
 * Used to benchmark impact of struct alignment
 */
#pragma pack(1)
template <size_t BLOCK_SIZE>
struct UnalignedParticleBlock
{
    float    momentum_x[BLOCK_SIZE];
    float    momentum_y[BLOCK_SIZE];
    float    momentum_z[BLOCK_SIZE];
    float    position_x[BLOCK_SIZE];
    float    weight[BLOCK_SIZE];
    uint16_t activeCount = 0;
};
#pragma pack()

/**
 * @brief Test 4: Struct Alignment Impact - Aligned vs Unaligned
 *
 * Directly compares the performance of accessing momentum components in:
 * - Real ParticleBlock with 64-byte alignment
 * - Mock UnalignedParticleBlock without alignment
 *
 * This shows the real-world impact of struct alignment on memory layout efficiency
 */
void benchmark_alignment_impact()
{
    std::cout << "\n=== Test 4: Struct Alignment Impact (Real Structs) ===\n";

    constexpr size_t bs            = 64;
    constexpr size_t num_blocks    = 1000; // 64K particles total
    constexpr size_t num_particles = num_blocks * bs;

    // --- Test 1: Aligned ParticleBlock ---
    {
        std::cout << "\nAligned ParticleBlock (64-byte alignment):\n";

        std::vector<particle::ParticleBlock<bs>> aligned_blocks;
        for (size_t b = 0; b < num_blocks; ++b)
        {
            particle::ParticleBlock<bs> pb;
            pb.activeCount = bs;
            for (size_t i = 0; i < bs; ++i)
            {
                pb.momentum_x[i] = 1.0f;
                pb.momentum_y[i] = 0.5f;
                pb.momentum_z[i] = 0.1f;
            }
            aligned_blocks.push_back(pb);
        }

        // Benchmark: Sequential momentum component access
        {
            PerfTimer     timer;
            float         sum        = 0.0f;
            constexpr int iterations = 1000;

            for (int it = 0; it < iterations; ++it)
            {
                for (const auto& pb : aligned_blocks)
                {
                    const auto& mx = pb.momentum_x;
                    const auto& my = pb.momentum_y;
                    const auto& mz = pb.momentum_z;
                    for (size_t i = 0; i < pb.activeCount; ++i)
                    {
                        sum += mx[i] + my[i] + mz[i];
                    }
                }
            }
            volatile float sink = sum; // Prevent optimization

            double elapsed_ms     = timer.elapsed_ms();
            size_t total_accesses = num_particles * 3 * iterations;
            double throughput     = total_accesses / elapsed_ms;

            std::cout << "  Momentum access (X+Y+Z): " << std::fixed << std::setprecision(2) << throughput << " Mp/s  ("
                      << std::fixed << std::setprecision(3) << elapsed_ms / iterations << " ms/iter)\n";
        }

        // Benchmark: Strided access (simulate gather with gaps)
        {
            PerfTimer     timer;
            float         sum        = 0.0f;
            constexpr int iterations = 1000;

            for (int it = 0; it < iterations; ++it)
            {
                for (const auto& pb : aligned_blocks)
                {
                    const auto& mx = pb.momentum_x;
                    for (size_t i = 0; i < pb.activeCount; i += 2)
                    {
                        sum += mx[i];
                    }
                }
            }
            volatile float sink = sum; // Prevent optimization

            double elapsed_ms     = timer.elapsed_ms();
            size_t total_accesses = (num_particles / 2) * iterations;
            double throughput     = total_accesses / elapsed_ms;

            std::cout << "  Strided momentum access: " << std::fixed << std::setprecision(2) << throughput << " Mp/s\n";
        }
    }

    // --- Test 2: Unaligned ParticleBlock ---
    {
        std::cout << "\nUnaligned ParticleBlock (no alignment, pragma pack(1)):\n";

        std::vector<UnalignedParticleBlock<bs>> unaligned_blocks;
        for (size_t b = 0; b < num_blocks; ++b)
        {
            UnalignedParticleBlock<bs> pb;
            pb.activeCount = bs;
            for (size_t i = 0; i < bs; ++i)
            {
                pb.momentum_x[i] = 1.0f;
                pb.momentum_y[i] = 0.5f;
                pb.momentum_z[i] = 0.1f;
            }
            unaligned_blocks.push_back(pb);
        }

        // Benchmark: Sequential momentum component access
        {
            PerfTimer     timer;
            float         sum        = 0.0f;
            constexpr int iterations = 1000;

            for (int it = 0; it < iterations; ++it)
            {
                for (const auto& pb : unaligned_blocks)
                {
                    const auto& mx = pb.momentum_x;
                    const auto& my = pb.momentum_y;
                    const auto& mz = pb.momentum_z;
                    for (size_t i = 0; i < pb.activeCount; ++i)
                    {
                        sum += mx[i] + my[i] + mz[i];
                    }
                }
            }
            volatile float sink = sum; // Prevent optimization

            double elapsed_ms     = timer.elapsed_ms();
            size_t total_accesses = num_particles * 3 * iterations;
            double throughput     = total_accesses / elapsed_ms;

            std::cout << "  Momentum access (X+Y+Z): " << std::fixed << std::setprecision(2) << throughput << " Mp/s  ("
                      << std::fixed << std::setprecision(3) << elapsed_ms / iterations << " ms/iter)\n";
        }

        // Benchmark: Strided access
        {
            PerfTimer     timer;
            float         sum        = 0.0f;
            constexpr int iterations = 1000;

            for (int it = 0; it < iterations; ++it)
            {
                for (const auto& pb : unaligned_blocks)
                {
                    const auto& mx = pb.momentum_x;
                    for (size_t i = 0; i < pb.activeCount; i += 2)
                    {
                        sum += mx[i];
                    }
                }
            }
            volatile float sink = sum; // Prevent optimization

            double elapsed_ms     = timer.elapsed_ms();
            size_t total_accesses = (num_particles / 2) * iterations;
            double throughput     = total_accesses / elapsed_ms;

            std::cout << "  Strided momentum access: " << std::fixed << std::setprecision(2) << throughput << " Mp/s\n";
        }
    }
}

/**
 * @brief Test 5: Component gather efficiency
 *
 * Tests the efficiency of gathering specific momentum components
 * This exercises the component() function and memory layout
 */
void benchmark_component_gather()
{
    std::cout << "\n=== Test 5: Component Gather Efficiency (Real Data Structures) ===\n";

    constexpr size_t bs            = 128;
    constexpr size_t num_particles = 1280000; // 10000 blocks of 128

    // Use real ParticleSystem
    particle::ParticleSystem<bs> particles(num_particles);
    particles.set_active(num_particles);

    // Initialize particle data
    for (auto& pb : particles)
    {
        for (size_t i = 0; i < pb.activeCount; ++i)
        {
            pb.momentum_x[i] = 1.0f;
            pb.momentum_y[i] = 0.5f;
            pb.momentum_z[i] = 0.1f;
        }
    }

    // Test gathering X component
    {
        PerfTimer     timer;
        float         sum        = 0.0f;
        constexpr int iterations = 1000;

        for (int it = 0; it < iterations; ++it)
        {
            for (const auto& pb : particles)
            {
                const auto& mom_x = pb.component(particle::MomentumComp::X);
                for (size_t i = 0; i < pb.activeCount; ++i)
                {
                    sum += mom_x[i];
                }
            }
        }
        volatile float sink = sum; // Prevent optimization

        double elapsed_ms     = timer.elapsed_ms();
        size_t total_accesses = num_particles * iterations;
        double throughput     = total_accesses / elapsed_ms;

        std::cout << "Component gather X: " << std::fixed << std::setprecision(2) << throughput << " Mp/s\n";
    }

    // Test gathering all components (SoA access pattern)
    {
        PerfTimer     timer;
        float         sum        = 0.0f;
        constexpr int iterations = 1000;

        for (int it = 0; it < iterations; ++it)
        {
            for (const auto& pb : particles)
            {
                const auto& mom_x = pb.component(particle::MomentumComp::X);
                const auto& mom_y = pb.component(particle::MomentumComp::Y);
                const auto& mom_z = pb.component(particle::MomentumComp::Z);
                for (size_t i = 0; i < pb.activeCount; ++i)
                {
                    sum += mom_x[i] + mom_y[i] + mom_z[i];
                }
            }
        }
        volatile float sink = sum; // Prevent optimization

        double elapsed_ms     = timer.elapsed_ms();
        size_t total_accesses = num_particles * iterations * 3;
        double throughput     = total_accesses / elapsed_ms;

        std::cout << "Component gather X+Y+Z: " << std::fixed << std::setprecision(2) << throughput << " Mp/s\n";
    }
}

int main()
{
    std::cout << "========================================\n";
    std::cout << "PIC Engine Performance Benchmarks\n";
    std::cout << "System under heavy load\n";
    std::cout << "========================================\n";

    benchmark_boris_push_heavy_load();
    benchmark_memory_bandwidth();
    benchmark_cache_efficiency();
    benchmark_alignment_impact();
    benchmark_component_gather();

    std::cout << "\n========================================\n";
    std::cout << "Benchmarks Complete\n";
    std::cout << "========================================\n";

    return 0;
}

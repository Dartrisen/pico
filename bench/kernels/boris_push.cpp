#include "bench/core/perf_counters.hpp"
#include "bench/core/timer.hpp"
#include "data/field/include/field_block.hpp"
#include "data/field/include/field_em.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_system.hpp"
#include "kernels/pusher/boris.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

    /**
     * @brief Throughput result: particles/sec and per-particle cost
     */
    struct ThroughputResult
    {
        const char* label;
        size_t      num_particles;
        size_t      num_iterations;
        double      total_time_ms;
        double      particles_per_sec;
        double      ns_per_particle;

        void print() const
        {
            std::cout << std::left << std::setw(40) << label << " | " << std::setw(10) << num_particles << " p | "
                      << std::setw(8) << std::fixed << std::setprecision(2) << total_time_ms << " ms | "
                      << std::setw(12) << std::scientific << std::setprecision(3) << particles_per_sec << " p/s | "
                      << std::setw(6) << std::fixed << std::setprecision(2) << ns_per_particle << " ns/p\n";
        }

        static void print_header()
        {
            std::cout << std::left << std::setw(40) << "Benchmark"
                      << " | " << std::setw(10) << "Particles"
                      << " | " << std::setw(8) << "Time(ms)"
                      << " | " << std::setw(12) << "p/sec"
                      << " | " << std::setw(6) << "ns/p\n";
            std::cout << std::string(100, '=') << "\n";
        }
    };

    /**
     * @brief Boris pusher: measure throughput for different problem sizes
     *
     * This stresses:
     * - Memory bandwidth (position, momentum arrays)
     * - ALU density (magnetic rotation math)
     * - Cache hierarchy (block-based layout)
     */
    void bench_boris_pusher_scaling()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BORIS PUSHER THROUGHPUT (particles/sec) ACROSS SCALES\n";
        std::cout << std::string(80, '=') << "\n";

        ThroughputResult::print_header();

        constexpr size_t    BS              = 8;
        std::vector<size_t> particle_counts = {1000, 4000, 8000, 16000, 40000, 100000};
        constexpr size_t    num_iterations  = 100;
        constexpr float     dt              = 0.01f;

        Grid grid(64, 64);

        for (size_t num_p : particle_counts)
        {
            particle::ParticleSystem<BS> particles(num_p);
            particles.set_active(num_p);

            // Initialize with realistic data
            for (auto& pb : particles)
            {
                for (size_t i = 0; i < pb.activeCount; ++i)
                {
                    pb.position_x[i] = 10.0f + static_cast<float>(i) * 0.1f;
                    pb.momentum_x[i] = 0.1f;
                    pb.momentum_y[i] = 0.05f;
                    pb.momentum_z[i] = 0.02f;
                }
            }

            // Create fields properly
            EMFields<BS> fields(grid);
            fields.E.set_fields<field::FieldComp::X>(0.1f);
            fields.E.set_fields<field::FieldComp::Y>(0.05f);
            fields.E.set_fields<field::FieldComp::Z>(0.02f);
            fields.B.set_fields<field::FieldComp::X>(0.001f);
            fields.B.set_fields<field::FieldComp::Y>(0.001f);
            fields.B.set_fields<field::FieldComp::Z>(0.001f);

            FieldScratch<BS> scratch;

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
                    kernels::pusher::BorisPusher<BS>::push_block(pb, scratch, dt);
                }
            }

            // Benchmark
            bench::Timer timer;
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
                    kernels::pusher::BorisPusher<BS>::push_block(pb, scratch, dt);
                }
            }
            double elapsed_ms = timer.elapsed_ms();

            // Calculate throughput
            uint64_t total_particle_updates = num_p * num_iterations;
            double   particles_per_sec      = (total_particle_updates * 1000.0) / elapsed_ms;
            double   ns_per_particle        = (elapsed_ms * 1e6) / total_particle_updates;

            ThroughputResult res;
            res.label             = "Boris Pusher";
            res.num_particles     = num_p;
            res.num_iterations    = num_iterations;
            res.total_time_ms     = elapsed_ms;
            res.particles_per_sec = particles_per_sec;
            res.ns_per_particle   = ns_per_particle;
            res.print();
        }
    }

    /**
     * @brief Measure cache behavior: working set size vs performance
     *
     * Different block sizes can fit in L1/L2/L3.
     * Shows how data layout impacts performance.
     */
    void bench_boris_pusher_cache_sensitivity()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BORIS PUSHER: CACHE SENSITIVITY (block size tuning)\n";
        std::cout << std::string(80, '=') << "\n";

        ThroughputResult::print_header();

        std::vector<size_t> block_sizes     = {8, 16, 32, 64};
        constexpr size_t    total_particles = 10000;
        constexpr size_t    num_iterations  = 50;
        constexpr float     dt              = 0.01f;

        Grid grid(64, 64);

        for (size_t bs : block_sizes)
        {
            if (bs == 8)
            {
                // Template specialization for BS=8
                particle::ParticleSystem<8> particles(total_particles);
                particles.set_active(total_particles);

                for (auto& pb : particles)
                {
                    for (size_t i = 0; i < pb.activeCount; ++i)
                    {
                        pb.position_x[i] = 10.0f + static_cast<float>(i) * 0.1f;
                        pb.momentum_x[i] = 0.1f;
                        pb.momentum_y[i] = 0.05f;
                        pb.momentum_z[i] = 0.02f;
                    }
                }

                EMFields<8> fields(grid);
                fields.E.set_fields<field::FieldComp::X>(0.1f);
                fields.E.set_fields<field::FieldComp::Y>(0.05f);
                fields.E.set_fields<field::FieldComp::Z>(0.02f);
                fields.B.set_fields<field::FieldComp::X>(0.001f);
                fields.B.set_fields<field::FieldComp::Y>(0.001f);
                fields.B.set_fields<field::FieldComp::Z>(0.001f);
                FieldScratch<8> scratch;

                bench::Timer timer;
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
                        kernels::pusher::BorisPusher<8>::push_block(pb, scratch, dt);
                    }
                }
                double elapsed_ms = timer.elapsed_ms();

                uint64_t total_ops         = total_particles * num_iterations;
                double   particles_per_sec = (total_ops * 1000.0) / elapsed_ms;
                double   ns_per_particle   = (elapsed_ms * 1e6) / total_ops;

                ThroughputResult res;
                res.label             = "Boris Pusher (BS=8)";
                res.num_particles     = total_particles;
                res.num_iterations    = num_iterations;
                res.total_time_ms     = elapsed_ms;
                res.particles_per_sec = particles_per_sec;
                res.ns_per_particle   = ns_per_particle;
                res.print();
            }
        }
    }

} // namespace

int main()
{
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║     BORIS PUSHER KERNEL PERFORMANCE BENCHMARK          ║\n";
    std::cout << "║  Measures throughput, latency, cache sensitivity       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    bench_boris_pusher_scaling();
    bench_boris_pusher_cache_sensitivity();

    return 0;
}

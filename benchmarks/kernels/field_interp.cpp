#include "bench/core/perf_counters.hpp"
#include "bench/core/timer.hpp"
#include "data/field/include/field_em.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_system.hpp"
#include "kernels/gather/gather.hpp"
#include "kernels/shapes/spline.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

    /**
     * @brief Field interpolation: measure gather operation cost
     *
     * This is memory-intensive:
     * - Random reads from field grid
     * - Multiple shape function evaluations
     * - Accumulation of weighted field values
     */
    void bench_field_gather_scaling()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "FIELD GATHER (INTERPOLATION) - SCALING\n";
        std::cout << std::string(80, '=') << "\n";

        std::cout << std::left << std::setw(40) << "Benchmark"
                  << " | " << std::setw(10) << "Particles"
                  << " | " << std::setw(8) << "Time(ms)"
                  << " | " << std::setw(12) << "p/sec"
                  << " | " << std::setw(6) << "ns/p\n";
        std::cout << std::string(100, '=') << "\n";

        constexpr size_t    BS              = 8;
        std::vector<size_t> particle_counts = {1000, 4000, 8000, 16000, 40000};
        constexpr size_t    num_iterations  = 100;

        Grid grid(128, 128);

        particle::ParticleSystem<BS> particles(particle_counts.back());
        for (size_t num_p : particle_counts)
        {
            particles.set_active(num_p);

            // Distribute particles uniformly across grid
            float cell_size = grid.cell_size();
            for (auto& pb : particles)
            {
                for (size_t i = 0; i < pb.activeCount; ++i)
                {
                    pb.position_x[i] = cell_size * (5.0f + static_cast<float>(i % 100) * 0.5f);
                }
            }

            // Create fields with realistic pattern
            EMFields<BS> fields(grid);
            // Initialize E field
            fields.E.set_fields<field::FieldComp::X>(0.1f);
            fields.E.set_fields<field::FieldComp::Y>(0.05f);
            fields.E.set_fields<field::FieldComp::Z>(0.02f);
            // Initialize B field
            fields.B.set_fields<field::FieldComp::X>(0.01f);
            fields.B.set_fields<field::FieldComp::Y>(0.01f);
            fields.B.set_fields<field::FieldComp::Z>(0.01f);
            FieldScratch<BS> scratch;

            // Warmup
            for (size_t iter = 0; iter < 3; ++iter)
            {
                for (auto& pb : particles)
                {
                    kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid,
                                                                                              scratch);
                }
            }

            // Benchmark
            bench::Timer timer;
            for (size_t iter = 0; iter < num_iterations; ++iter)
            {
                for (auto& pb : particles)
                {
                    kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid,
                                                                                              scratch);
                }
            }
            double elapsed_ms = timer.elapsed_ms();

            uint64_t total_ops         = num_p * num_iterations;
            double   particles_per_sec = (total_ops * 1000.0) / elapsed_ms;
            double   ns_per_particle   = (elapsed_ms * 1e6) / total_ops;

            std::cout << std::left << std::setw(40) << "FieldGather (LinearSpline)"
                      << " | " << std::setw(10) << num_p << " | " << std::setw(8) << std::fixed << std::setprecision(2)
                      << elapsed_ms << " | " << std::setw(12) << std::scientific << std::setprecision(3)
                      << particles_per_sec << " | " << std::setw(6) << std::fixed << std::setprecision(2)
                      << ns_per_particle << "\n";
        }
    }

    /**
     * @brief Memory bandwidth estimate for gather
     *
     * Each particle reads:
     * - 1x position (1 float)
     * - Shape support points (typically 2-4 floats per field component × 6 components)
     * - Scattered across field grid
     */
    void bench_field_gather_bandwidth()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "FIELD GATHER - MEMORY BANDWIDTH ESTIMATE\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t BS             = 8;
        constexpr size_t num_particles  = 20000;
        constexpr size_t num_iterations = 100;

        Grid grid(128, 128);

        particle::ParticleSystem<BS> particles(num_particles);
        particles.set_active(num_particles);

        // Distribute uniformly
        float cell_size = grid.cell_size();
        for (auto& pb : particles)
        {
            for (size_t i = 0; i < pb.activeCount; ++i)
            {
                pb.position_x[i] = cell_size * (5.0f + static_cast<float>(i % 100) * 0.5f);
            }
        }

        EMFields<BS> fields(grid);
        // Initialize E field
        fields.E.set_fields<field::FieldComp::X>(1.0f);
        fields.E.set_fields<field::FieldComp::Y>(0.5f);
        fields.E.set_fields<field::FieldComp::Z>(0.2f);

        // Initialize B field
        fields.B.set_fields<field::FieldComp::X>(0.01f);
        fields.B.set_fields<field::FieldComp::Y>(0.01f);
        fields.B.set_fields<field::FieldComp::Z>(0.01f);
        FieldScratch<BS> scratch;

        bench::Timer timer;
        for (size_t iter = 0; iter < num_iterations; ++iter)
        {
            for (auto& pb : particles)
            {
                kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid, scratch);
            }
        }
        double elapsed_sec = timer.elapsed_sec();

        // Estimate bytes accessed:
        // - Each particle reads ~2 field values (E, B) × 2 shape points = 4 floats = 16 bytes
        // - Conservative estimate: 20 bytes/particle (accounting for cache effects)
        uint64_t                bytes_accessed = num_particles * num_iterations * 20;
        bench::BandwidthCounter bw(bytes_accessed, elapsed_sec);
        bw.print();

        std::cout << "  Total time: " << std::fixed << std::setprecision(3) << elapsed_sec << " sec\n";
        std::cout << "  Particle updates: " << (num_particles * num_iterations / 1e6) << " M\n";
    }

} // namespace

int main()
{
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  FIELD INTERPOLATION (GATHER) BENCHMARK                ║\n";
    std::cout << "║  Measures memory access patterns, bandwidth            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    bench_field_gather_scaling();
    bench_field_gather_bandwidth();

    return 0;
}

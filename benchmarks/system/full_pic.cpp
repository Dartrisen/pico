#include "bench/core/perf_counters.hpp"
#include "bench/core/timer.hpp"
#include "data/field/include/field_em.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_system.hpp"
#include "kernels/gather/gather.hpp"
#include "kernels/pusher/boris.hpp"
#include "kernels/shapes/spline.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

    /**
     * @brief Full PIC cycle: gather -> push -> deposit (simplified)
     *
     * Measures end-to-end performance with realistic workload:
     * - Field interpolation (gather)
     * - Particle push
     * - No deposit for now
     *
     * This is the "beating heart" of the PIC algorithm
     */
    void bench_full_pic_cycle()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "FULL PIC CYCLE (gather + push)\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t    BS              = 512;
        std::vector<size_t> particle_counts = {1000, 5000, 10000, 20000, 50000};
        constexpr size_t    num_timesteps   = 50;
        constexpr float     dt              = 0.01f;

        Grid grid(64, 64);

        std::cout << std::left << std::setw(15) << "Particles"
                  << " | " << std::setw(12) << "Time(ms)"
                  << " | " << std::setw(12) << "p/ts/sec"
                  << " | " << std::setw(12) << "Throughput\n";
        std::cout << std::string(80, '=') << "\n";

        particle::ParticleSystem<BS> particles(particle_counts.back());
        for (size_t num_p : particle_counts)
        {
            particles.set_active(num_p);

            // Initialize particles
            float  cell_size = grid.cell_size();
            size_t idx       = 0;
            for (auto& pb : particles)
            {
                for (size_t i = 0; i < pb.activeCount; ++i)
                {
                    pb.position_x[i] = cell_size * (5.0f + static_cast<float>(idx % 50) * 0.5f);
                    pb.momentum_x[i] = 0.1f;
                    pb.momentum_y[i] = 0.05f;
                    pb.momentum_z[i] = 0.02f;
                    ++idx;
                }
            }

            // Create fields
            EMFields<BS> fields(grid);
            fields.E.set_fields<field::FieldComp::X>(0.1f);
            fields.E.set_fields<field::FieldComp::Y>(0.05f);
            fields.E.set_fields<field::FieldComp::Z>(0.02f);
            fields.B.set_fields<field::FieldComp::X>(0.001f);
            fields.B.set_fields<field::FieldComp::Y>(0.001f);
            fields.B.set_fields<field::FieldComp::Z>(0.001f);
            FieldScratch<BS> scratch;

            // Warmup
            for (size_t step = 0; step < 3; ++step)
            {
                for (auto& pb : particles)
                {
                    // Gather
                    kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid,
                                                                                              scratch);
                    // Push
                    kernels::pusher::BorisPusher<BS>::push_block(pb, scratch, dt);
                }
            }

            // Benchmark
            bench::Timer timer;
            for (size_t step = 0; step < num_timesteps; ++step)
            {
                for (auto& pb : particles)
                {
                    // Gather field values
                    kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid,
                                                                                              scratch);
                    // Push particles
                    kernels::pusher::BorisPusher<BS>::push_block(pb, scratch, dt);

                    // Note: deposit omitted for now (would accumulate current on grid)
                }
            }
            double elapsed_ms = timer.elapsed_ms();

            uint64_t total_particle_steps     = num_p * num_timesteps;
            double   particles_per_ts_per_sec = (total_particle_steps * 1000.0) / elapsed_ms;

            // Throughput in Gparticles/sec
            double throughput_gp_per_sec = particles_per_ts_per_sec / 1e9;

            std::cout << std::setw(15) << num_p << " | " << std::fixed << std::setprecision(2) << std::setw(12)
                      << elapsed_ms << " | " << std::scientific << std::setprecision(3) << std::setw(12)
                      << particles_per_ts_per_sec << " | " << std::setw(12) << throughput_gp_per_sec << " Gp/s\n";
        }
    }

    /**
     * @brief Weak scaling: increase particles, measure time growth
     *
     * Ideal: time should stay roughly constant as we increase particles
     * and timesteps proportionally
     */
    void bench_weak_scaling()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "WEAK SCALING (particles x timesteps fixed ratio)\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t BS = 8;
        constexpr float  dt = 0.01f;

        // Scaling: keep particles/cell roughly constant
        // Grid is fixed at 64x64, so 4096 cells
        // Start with 10 particles/cell = 40,960 total
        std::vector<size_t> particle_counts = {10000, 40960, 100000};
        std::vector<size_t> timestep_counts = {100, 50, 25}; // Adjust for fair comparison

        Grid grid(64, 64);

        std::cout << std::left << std::setw(15) << "Particles"
                  << " | " << std::setw(12) << "Timesteps"
                  << " | " << std::setw(12) << "Time(ms)"
                  << " | " << std::setw(12) << "Efficiency\n";
        std::cout << std::string(80, '=') << "\n";

        double baseline_time = 0.0;

        for (size_t idx = 0; idx < particle_counts.size(); ++idx)
        {
            size_t num_p  = particle_counts[idx];
            size_t num_ts = timestep_counts[idx];

            particle::ParticleSystem<BS> particles(num_p);
            particles.set_active(num_p);

            // Initialize
            float  cell_size = grid.cell_size();
            size_t pidx      = 0;
            for (auto& pb : particles)
            {
                for (size_t i = 0; i < pb.activeCount; ++i)
                {
                    pb.position_x[i] = cell_size * (5.0f + static_cast<float>(pidx % 50) * 0.5f);
                    pb.momentum_x[i] = 0.1f;
                    pb.momentum_y[i] = 0.05f;
                    pb.momentum_z[i] = 0.02f;
                    ++pidx;
                }
            }

            EMFields<BS> fields(grid);
            fields.E.set_fields<field::FieldComp::X>(0.1f);
            fields.E.set_fields<field::FieldComp::Y>(0.05f);
            fields.E.set_fields<field::FieldComp::Z>(0.02f);
            fields.B.set_fields<field::FieldComp::X>(0.001f);
            fields.B.set_fields<field::FieldComp::Y>(0.001f);
            fields.B.set_fields<field::FieldComp::Z>(0.001f);
            FieldScratch<BS> scratch;

            bench::Timer timer;
            for (size_t step = 0; step < num_ts; ++step)
            {
                for (auto& pb : particles)
                {
                    kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid,
                                                                                              scratch);
                    kernels::pusher::BorisPusher<BS>::push_block(pb, scratch, dt);
                }
            }
            double elapsed_ms = timer.elapsed_ms();

            if (idx == 0)
                baseline_time = elapsed_ms;

            double efficiency = (baseline_time / elapsed_ms) * 100.0;

            std::cout << std::setw(15) << num_p << " | " << std::setw(12) << num_ts << " | " << std::fixed
                      << std::setprecision(2) << std::setw(12) << elapsed_ms << " | " << std::setw(12) << efficiency
                      << " %\n";
        }
    }

    /**
     * @brief Strong scaling: increase threads/workers for fixed problem
     *
     * Note: This is a theoretical benchmark since pico is single-threaded.
     * We measure the raw compute cost and estimate speedup potential.
     */
    void bench_algorithmic_breakdown()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "ALGORITHMIC BREAKDOWN (per operation cost)\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t BS             = 8;
        constexpr size_t num_particles  = 20000;
        constexpr size_t num_iterations = 100;
        constexpr float  dt             = 0.01f;

        Grid grid(64, 64);

        particle::ParticleSystem<BS> particles(num_particles);
        particles.set_active(num_particles);

        float  cell_size = grid.cell_size();
        size_t idx       = 0;
        for (auto& pb : particles)
        {
            for (size_t i = 0; i < pb.activeCount; ++i)
            {
                pb.position_x[i] = cell_size * (5.0f + static_cast<float>(idx % 50) * 0.5f);
                pb.momentum_x[i] = 0.1f;
                pb.momentum_y[i] = 0.05f;
                pb.momentum_z[i] = 0.02f;
                ++idx;
            }
        }

        EMFields<BS> fields(grid);
        fields.E.set_fields<field::FieldComp::X>(0.1f);
        fields.E.set_fields<field::FieldComp::Y>(0.05f);
        fields.E.set_fields<field::FieldComp::Z>(0.02f);
        fields.B.set_fields<field::FieldComp::X>(0.001f);
        fields.B.set_fields<field::FieldComp::Y>(0.001f);
        fields.B.set_fields<field::FieldComp::Z>(0.001f);
        FieldScratch<BS> scratch;

        std::cout << std::left << std::setw(30) << "Operation"
                  << " | " << std::setw(12) << "Time(ms)"
                  << " | " << std::setw(12) << "%% of Total\n";
        std::cout << std::string(80, '=') << "\n";

        // Measure gather
        bench::Timer gather_timer;
        for (size_t iter = 0; iter < num_iterations; ++iter)
        {
            for (auto& pb : particles)
            {
                kernels::gather::FieldGather<kernels::shapes::SplineShape<1>, BS>::gather(pb, fields, grid, scratch);
            }
        }
        double gather_time = gather_timer.elapsed_ms();

        // Measure push
        bench::Timer push_timer;
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
        double push_time = push_timer.elapsed_ms();

        double total_time = gather_time + push_time;

        std::cout << std::setw(30) << "Gather (field interpolation)"
                  << " | " << std::fixed << std::setprecision(2) << std::setw(12) << gather_time << " | "
                  << std::setprecision(1) << std::setw(12) << (100.0 * gather_time / total_time) << " %\n";

        std::cout << std::setw(30) << "Push (Boris)"
                  << " | " << std::setw(12) << push_time << " | " << std::setw(12) << (100.0 * push_time / total_time)
                  << " %\n";

        std::cout << std::setw(30) << "TOTAL"
                  << " | " << std::setw(12) << total_time << " | " << std::setw(12) << "100.0"
                  << " %\n";
    }

} // namespace

int main()
{
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║         FULL PIC SYSTEM PERFORMANCE BENCHMARK          ║\n";
    std::cout << "║  End-to-end cycle, scaling, algorithmic breakdown      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    bench_full_pic_cycle();
    bench_weak_scaling();
    bench_algorithmic_breakdown();

    return 0;
}

#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/ParticleSortLocalityVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/Injectors.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "engine/modules/sorter/ParticleSorter.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

// Scrambles particle x-positions across blocks to introduce spatial cache misses
template <typename Engine>
void scramble_particle_positions(Engine& engine, double domain_length)
{
    auto&                                 particles = engine.particles();
    std::mt19937                          rng(1337);
    std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(domain_length));

    for (auto& block : particles)
    {
        for (std::size_t i = 0; i < block.activeCount; ++i)
        {
            block.position_x[i] = dist(rng);
        }
    }
}

int main()
{
    constexpr std::size_t grid_cells = 2048;
    constexpr double      dx         = 0.05;
    constexpr double      dt         = 0.001;
    constexpr std::size_t ppc        = 1000; // ~2.04M total particles
    constexpr std::size_t nsteps     = 100;
    constexpr std::size_t BS         = 64;

    assert(dx > 0.95 * dt && "CFL condition violated.");

    Grid grid(grid_cells, dx, /*guards=*/4);

    using Shape     = kernels::shapes::SplineShape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::NoInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    // 1. Initialize Engine & Thermal Particles
    EngineT engine_instance{grid, ppc};
    auto&   particles = engine_instance.particles();
    particles.init_positions_uniform(grid);
    particles.init_velocities_thermal(/*v_th=*/2.0f, 0.0f, 0.0f, 0.0f, /*seed=*/42);

    const double domain_length = grid.physical_size() * grid.cell_size();
    scramble_particle_positions(engine_instance, domain_length);

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    PICApp app(std::move(wrapper), dt);

    // 2. Measure Execution on Unsorted (Scrambled) System via PICApp
    const auto start_unsorted = std::chrono::high_resolution_clock::now();
    app.run(nsteps);
    const auto   end_unsorted = std::chrono::high_resolution_clock::now();
    const double unsorted_ms  = std::chrono::duration<double, std::milli>(end_unsorted - start_unsorted).count();

    // 3. Sort Block Particles using pico::modules::sorter::ParticleSorter Kernel
    pico::modules::sorter::ParticleSorter<BS> sorter;
    for (auto& block : concrete_wrapper->engine().particles())
    {
        sorter.sort_block(block, grid);
    }

    // 4. Measure Execution on Sorted System via PICApp
    const auto start_sorted = std::chrono::high_resolution_clock::now();
    app.run(nsteps);
    const auto   end_sorted = std::chrono::high_resolution_clock::now();
    const double sorted_ms  = std::chrono::duration<double, std::milli>(end_sorted - start_sorted).count();

    // 5. Verification Analysis & Reporting
    pico::diagnostics::ParticleSortLocalityVerifier verifier(/*minimum_speedup=*/1.02);
    const auto                                      res = verifier.verify(unsorted_ms, sorted_ms);

    pico::ui::VerificationReport report("Particle Sort & Cache Locality Verification", res.passed);

    report.add_row("Total Active Particles", std::to_string(concrete_wrapper->engine().particles().active_particles()));
    report.add_fixed_row("Unsorted Runtime", res.unsorted_time_ms, 2, "ms");
    report.add_fixed_row("Sorted Runtime", res.sorted_time_ms, 2, "ms");
    report.add_fixed_row("Locality Speedup", res.speedup, 3, "x");
    report.add_pct_row("Runtime Reduction", res.improvement_pct, res.passed);

    report.print();
    return report.passed() ? 0 : 1;
}

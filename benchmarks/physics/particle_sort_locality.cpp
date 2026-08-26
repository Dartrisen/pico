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

template <typename Engine>
void scramble_block_particles(Engine& engine, double dx, float range = 24.0f)
{
    auto&                                 particles = engine.particles();
    std::mt19937                          rng(1337);
    std::uniform_real_distribution<float> dist(-range * static_cast<float>(dx), range * static_cast<float>(dx));

    for (auto& block : particles)
    {
        for (std::size_t i = 0; i < block.activeCount; ++i)
        {
            block.position_x[i] += dist(rng);
        }
    }
}

int main()
{
    constexpr std::size_t grid_cells = 131072;
    constexpr double      dx         = 0.05;
    constexpr double      dt         = 0.001;
    constexpr std::size_t ppc        = 64;
    constexpr std::size_t nsteps     = 500;
    constexpr std::size_t BS         = 64;

    assert(dt <= 0.95 * dx && "CFL condition violated.");

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

    auto create_engine = [&]()
    {
        EngineT engine{grid, ppc};

        auto& particles = engine.particles();

        particles.init_positions_uniform(grid);
        particles.init_velocities_thermal(
                /*v_th=*/0.001f, 0.0f, 0.0f, 0.0f,
                /*seed=*/42);

        scramble_block_particles(engine, dx, /*range=*/24.0f);

        return engine;
    };

    // ------------------------------------------------------------
    // 1. SORTING OFF
    // ------------------------------------------------------------
    double unsorted_ms = 0.0;
    {
        EngineT engine = create_engine();

        engine.set_sort_frequency(0);
        engine.set_locality_threshold(0.0);

        auto   wrapper = std::make_unique<EngineWrapper<EngineT>>(std::move(engine));
        PICApp app(std::move(wrapper), dt);

        const auto start = std::chrono::high_resolution_clock::now();
        app.run(nsteps);
        const auto end = std::chrono::high_resolution_clock::now();
        unsorted_ms    = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // ------------------------------------------------------------
    // 2. GLOBAL SORTING ON
    // ------------------------------------------------------------
    double      sorted_ms        = 0.0;
    std::size_t active_particles = 0;
    {
        EngineT engine   = create_engine();
        active_particles = engine.particles().active_particles();

        engine.set_sort_frequency(100);
        engine.set_locality_threshold(0.0);

        auto   wrapper = std::make_unique<EngineWrapper<EngineT>>(std::move(engine));
        PICApp app(std::move(wrapper), dt);

        const auto start = std::chrono::high_resolution_clock::now();
        app.run(nsteps);
        const auto end = std::chrono::high_resolution_clock::now();
        sorted_ms      = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // 3. Verification Analysis & Reporting
    pico::diagnostics::ParticleSortLocalityVerifier verifier(/*minimum_speedup=*/1.02);
    const auto                                      res = verifier.verify(unsorted_ms, sorted_ms);

    pico::ui::VerificationReport report("Particle Sort & Cache Locality Verification", res.passed);

    report.add_row("Total Active Particles", std::to_string(active_particles));
    report.add_fixed_row("Unsorted Runtime", res.unsorted_time_ms, 2, "ms");
    report.add_fixed_row("Sorted Runtime", res.sorted_time_ms, 2, "ms");
    report.add_fixed_row("Locality Speedup", res.speedup, 3, "x");
    report.add_pct_row("Runtime Reduction", res.improvement_pct, res.passed);

    report.print();
    return report.passed() ? 0 : 1;
}

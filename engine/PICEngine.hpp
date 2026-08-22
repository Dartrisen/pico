#pragma once

#include "EngineBase.hpp"
#include "data/field/include/field_em.hpp"
#include "data/field/include/field_system.hpp"
#include "data/particle/include/particle_system.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/SilverMuller.hpp"
#include "engine/modules/boundary/Thermalizing.hpp"
#include "engine/modules/concepts.hpp"
#include "engine/modules/injector/Injectors.hpp"
#include "engine/modules/sorter/ParticleSorter.hpp"
#include "engine/perf/PipelineProfiler.hpp"

#include <cstdint>
#include <utility>

template <class FieldSolverT, class GatherT, class PusherT, class DepositT, class FieldBoundaryT, class ParticleBoundaryT,
          class FieldInjectorT = pico::modules::injector::NoInjector<64>, std::size_t BLOCK_SIZE = 64>
    requires pico::modules::FieldSolver<FieldSolverT, EMFields<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::Gather<GatherT, particle::ParticleBlock<BLOCK_SIZE>, EMFields<BLOCK_SIZE>, FieldScratch<BLOCK_SIZE>> &&
             pico::modules::Pusher<PusherT, particle::ParticleBlock<BLOCK_SIZE>, FieldScratch<BLOCK_SIZE>> &&
             pico::modules::Deposit<DepositT, particle::ParticleBlock<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::FieldBoundary<FieldBoundaryT, EMFields<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::ParticleBoundary<ParticleBoundaryT, particle::ParticleBlock<BLOCK_SIZE>>
class PICEngine final : public EngineBase<PICEngine<FieldSolverT, GatherT, PusherT, DepositT, FieldBoundaryT, ParticleBoundaryT, FieldInjectorT, BLOCK_SIZE>>
{
public:
    // Move Semantics
    PICEngine(PICEngine&&) noexcept            = default;
    PICEngine& operator=(PICEngine&&) noexcept = default;
    PICEngine(const PICEngine&)                = delete;
    PICEngine& operator=(const PICEngine&)     = delete;

    // 1. Grid + PPC Constructor (Constant density n0, default injector)
    explicit PICEngine(const Grid& grid, std::size_t particles_per_cell = 10, float n0 = 1.0f)
            : PICEngine(grid, particles_per_cell, FieldBoundaryT{}, ParticleBoundaryT{}, FieldInjectorT{}, n0)
    {
    }

    // 2. Full Constructor with Boundaries and Field Injector
    PICEngine(const Grid& grid, std::size_t particles_per_cell, FieldBoundaryT field_boundary, ParticleBoundaryT particle_boundary,
              FieldInjectorT field_injector = FieldInjectorT{}, float n0 = 1.0f)
            : fields_(grid), current_(grid), particles_(grid.physical_size() * particles_per_cell), field_solver_{}, pusher_{}, gather_{}, deposit_{},
              field_boundary_(std::move(field_boundary)), particle_boundary_(std::move(particle_boundary)), field_injector_(std::move(field_injector)),
              particles_per_cell_(particles_per_cell)
    {
        particles_.set_active(grid.physical_size() * particles_per_cell);
        particles_.init_positions_uniform(grid);
        particles_.init_velocities_cold(0.0f, 0.0f, 0.0f);
        particles_.init_density_constant(n0);
    }

    // 3. Density Profile Constructor
    template <typename DensityFunc>
    PICEngine(const Grid& grid, std::size_t particles_per_cell, DensityFunc&& density_fn, FieldBoundaryT field_boundary = FieldBoundaryT{},
              ParticleBoundaryT particle_boundary = ParticleBoundaryT{}, FieldInjectorT field_injector = FieldInjectorT{})
            : fields_(grid), current_(grid), particles_(grid.physical_size() * particles_per_cell), field_solver_{}, pusher_{}, gather_{}, deposit_{},
              field_boundary_(std::move(field_boundary)), particle_boundary_(std::move(particle_boundary)), field_injector_(std::move(field_injector)),
              particles_per_cell_(particles_per_cell)
    {
        particles_.set_active(grid.physical_size() * particles_per_cell);
        particles_.init_positions_uniform(grid);
        particles_.init_velocities_cold(0.0f, 0.0f, 0.0f);
        particles_.init_density_profile(grid, std::forward<DensityFunc>(density_fn));
    }

    // 4. Constructor with Pre-constructed ParticleSystem
    PICEngine(const Grid& grid, particle::ParticleSystem<BLOCK_SIZE> particles, FieldBoundaryT field_boundary = FieldBoundaryT{},
              ParticleBoundaryT particle_boundary = ParticleBoundaryT{}, FieldInjectorT field_injector = FieldInjectorT{})
            : fields_(grid), current_(grid), particles_(std::move(particles)), field_solver_{}, pusher_{}, gather_{}, deposit_{}, field_boundary_(std::move(field_boundary)),
              particle_boundary_(std::move(particle_boundary)), field_injector_(std::move(field_injector)),
              particles_per_cell_(particles_.active_particles() / grid.physical_size())
    {
    }

    void advance_impl(double dt)
    {
        const Grid&  grid         = fields_.E.grid();
        const double current_time = static_cast<double>(step_counter_) * dt;

        if (particles_.active_particles() > 0 && (step_counter_ % sort_frequency_ == 0))
        {
            auto timer = profiler_.time_stage(pico::perf::Stage::Sorting);
            for (auto& block : particles_)
            {
                sorter_.sort_block(block, grid);
            }
        }

        {
            auto timer = profiler_.time_stage(pico::perf::Stage::Boundaries);
            field_boundary_.fill_field_guards(fields_, grid);
        }

        current_.zero_out();
        // clang-format off
        #pragma omp parallel
        // clang-format on
        {
            FieldSystem<BLOCK_SIZE> thread_current(grid);
            thread_current.zero_out();

            FieldScratch<BLOCK_SIZE> thread_scratch;

            std::uint64_t local_gather_ns  = 0;
            std::uint64_t local_push_ns    = 0;
            std::uint64_t local_deposit_ns = 0;
            // clang-format off
            #pragma omp for schedule(static)
            // clang-format on
            for (auto& block : particles_)
            {
                using clock = pico::perf::PipelineProfiler::clock;

                const auto t0 = clock::now();
                gather_.gather_block(block, fields_, grid, thread_scratch);

                const auto t1 = clock::now();
                pusher_.push_block(block, thread_scratch, dt);
                particle_boundary_.apply(block, grid);

                const auto t2 = clock::now();
                deposit_.deposit_block(block, thread_current, grid, dt, particles_per_cell_);

                const auto t3 = clock::now();

                local_gather_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                local_push_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
                local_deposit_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
            }

            profiler_.add_nanoseconds(pico::perf::Stage::Gather, local_gather_ns);
            profiler_.add_nanoseconds(pico::perf::Stage::Pusher, local_push_ns);
            profiler_.add_nanoseconds(pico::perf::Stage::Deposit, local_deposit_ns);
            // clang-format off
            #pragma omp critical
            // clang-format on
            {
                current_.accumulate(thread_current);
            }
        }

        {
            auto timer = profiler_.time_stage(pico::perf::Stage::Boundaries);
            field_boundary_.fold_currents(current_, grid);
        }

        {
            auto timer = profiler_.time_stage(pico::perf::Stage::FieldSolver);
            field_solver_.solve(fields_, current_, dt);
            field_injector_.inject(fields_, current_time, dt);
        }

        ++step_counter_;
    }

    // Diagnostics & Profiling Interface
    const pico::perf::PipelineProfiler& profiler() const noexcept
    {
        return profiler_;
    }

    void reset_profiler() noexcept
    {
        profiler_.reset();
        step_counter_ = 0;
    }

    void print_perf_report() const
    {
        const double total_t = profiler_.total_seconds();

        std::cout << "\n--- Engine Profiler Summary (" << step_counter_ << " steps) ---\n";
        for (std::size_t i = 0; i < static_cast<std::size_t>(pico::perf::Stage::Count); ++i)
        {
            const auto   stage = static_cast<pico::perf::Stage>(i);
            const double t     = profiler_.seconds(stage);
            const double pct   = (total_t > 0.0) ? (t / total_t) * 100.0 : 0.0;

            std::cout << std::left << std::setw(36) << pico::perf::STAGE_NAMES[i] << ": " << std::right << std::setw(12) << pico::perf::PipelineProfiler::format_time(t) << " ("
                      << std::fixed << std::setprecision(1) << std::setw(5) << pct << "%)\n";
        }
        std::cout << "---------------------------------------------\n";
    }

    // State Accessors
    particle::ParticleSystem<BLOCK_SIZE>& particles() noexcept
    {
        return particles_;
    }
    const particle::ParticleSystem<BLOCK_SIZE>& particles() const noexcept
    {
        return particles_;
    }

    EMFields<BLOCK_SIZE>& fields() noexcept
    {
        return fields_;
    }
    const EMFields<BLOCK_SIZE>& fields() const noexcept
    {
        return fields_;
    }

    FieldSystem<BLOCK_SIZE>& current() noexcept
    {
        return current_;
    }
    const FieldSystem<BLOCK_SIZE>& current() const noexcept
    {
        return current_;
    }

    std::size_t particles_per_cell() const noexcept
    {
        return particles_per_cell_;
    }

private:
    EMFields<BLOCK_SIZE>                 fields_;
    FieldSystem<BLOCK_SIZE>              current_;
    particle::ParticleSystem<BLOCK_SIZE> particles_;

    FieldSolverT                                      field_solver_;
    PusherT                                           pusher_;
    GatherT                                           gather_;
    DepositT                                          deposit_;
    FieldBoundaryT                                    field_boundary_;
    ParticleBoundaryT                                 particle_boundary_;
    FieldInjectorT                                    field_injector_;
    pico::modules::sorter::ParticleSorter<BLOCK_SIZE> sorter_;

    std::size_t particles_per_cell_{10};

    pico::perf::PipelineProfiler profiler_{};
    std::size_t                  step_counter_{0};
    std::size_t                  sort_frequency_{50};
};

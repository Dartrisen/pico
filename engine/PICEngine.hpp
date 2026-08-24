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
#include "engine/perf/LocalityProfiler.hpp"
#include "engine/perf/PipelineProfiler.hpp"

#include <cstdint>
#include <omp.h>
#include <utility>
#include <vector>

/**
 * @brief High-performance parallel Particle-in-Cell execution engine.
 * @tparam FieldSolverT Electromagnetic field solver module.
 * @tparam GatherT Interpolation module from fields to particles.
 * @tparam PusherT Particle trajectory integrator.
 * @tparam DepositT Current deposition module from particles to grid.
 * @tparam FieldBoundaryT Field boundary condition handler.
 * @tparam ParticleBoundaryT Particle boundary condition handler.
 * @tparam FieldInjectorT External field source injector.
 * @tparam BLOCK_SIZE Number of particles per block unit for SIMD layout.
 */
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
    PICEngine(PICEngine&&) noexcept            = default;
    PICEngine& operator=(PICEngine&&) noexcept = default;
    PICEngine(const PICEngine&)                = delete;
    PICEngine& operator=(const PICEngine&)     = delete;

    explicit PICEngine(const Grid& grid, std::size_t particles_per_cell = 10, float n0 = 1.0f)
            : PICEngine(grid, particles_per_cell, FieldBoundaryT{}, ParticleBoundaryT{}, FieldInjectorT{}, n0)
    {
    }

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

    PICEngine(const Grid& grid, particle::ParticleSystem<BLOCK_SIZE> particles, FieldBoundaryT field_boundary = FieldBoundaryT{},
              ParticleBoundaryT particle_boundary = ParticleBoundaryT{}, FieldInjectorT field_injector = FieldInjectorT{})
            : fields_(grid), current_(grid), particles_(std::move(particles)), field_solver_{}, pusher_{}, gather_{}, deposit_{}, field_boundary_(std::move(field_boundary)),
              particle_boundary_(std::move(particle_boundary)), field_injector_(std::move(field_injector)),
              particles_per_cell_(particles_.active_particles() / grid.physical_size())
    {
    }

    /**
     * @brief Advances the particle and field state by a single time step dt.
     * @param dt Time step duration.
     */
    void advance_impl(double dt)
    {
        const Grid&  grid         = fields_.E.grid();
        const double current_time = static_cast<double>(step_counter_) * dt;

        const bool stride_degraded = (locality_threshold_ > 0.0) && (locality_metrics_.mean_stride() > locality_threshold_);
        const bool periodic_sort   = (sort_frequency_ > 0) && (step_counter_ % sort_frequency_ == 0);

        if (particles_.active_particles() > 0 && (periodic_sort || stride_degraded))
        {
            execute_stage(pico::perf::Stage::Sorting, [&] { sorter_.sort(particles_, grid); });
        }
        execute_stage(pico::perf::Stage::Boundaries, [&] { field_boundary_.fill_field_guards(fields_, grid); });

        current_.zero_out();
        ensure_thread_buffers(grid);

        // Compile-time dispatch based on Pusher capabilities
        if constexpr (pico::modules::AsyncPusher<PusherT>)
        {
            advance_staged_gpu(grid, dt);
        }
        else
        {
            advance_fused_cpu(grid, dt);
        }

        // Accumulate thread-local currents into global current grid
        for (const auto& tc : thread_currents_)
        {
            current_.accumulate(tc);
        }

        locality_metrics_.reset();
        for (const auto& tm : thread_metrics_)
        {
            locality_metrics_.accumulate(tm);
        }

        execute_stage(pico::perf::Stage::Boundaries, [&] { field_boundary_.fold_currents(current_, grid); });
        execute_stage(pico::perf::Stage::FieldSolver,
                      [&]
                      {
                          field_solver_.solve(fields_, current_, dt);
                          field_injector_.inject(fields_, current_time, dt);
                      });
        ++step_counter_;
    }

    void enable_stage_profiling(bool enable) noexcept
    {
        enable_stage_profiling_ = enable;
    }
    void enable_locality_diagnostics(bool enable) noexcept
    {
        enable_locality_diagnostics_ = enable;
    }
    [[nodiscard]] double mean_cell_stride() const noexcept
    {
        return locality_metrics_.mean_stride();
    }
    [[nodiscard]] const pico::diagnostics::LocalityMetrics& locality_metrics() const noexcept
    {
        return locality_metrics_;
    }
    void set_sort_frequency(std::size_t frequency) noexcept
    {
        sort_frequency_ = frequency;
    }
    void set_locality_threshold(double max_stride) noexcept
    {
        locality_threshold_ = max_stride;
    }
    [[nodiscard]] const pico::perf::PipelineProfiler& profiler() const noexcept
    {
        return profiler_;
    }

    void reset_profiler() noexcept
    {
        profiler_.reset();
        step_counter_ = 0;
    }

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
    [[nodiscard]] std::size_t particles_per_cell() const noexcept
    {
        return particles_per_cell_;
    }

private:
    template <typename StageFunc>
    void execute_stage(pico::perf::Stage stage, StageFunc&& func)
    {
        if (enable_stage_profiling_)
        {
            auto timer = profiler_.time_stage(stage);
            std::forward<StageFunc>(func)();
        }
        else
        {
            std::forward<StageFunc>(func)();
        }
    }

    void ensure_thread_buffers(const Grid& grid)
    {
        const std::size_t nthreads = static_cast<std::size_t>(omp_get_max_threads());

        if (thread_currents_.size() < nthreads)
        {
            thread_currents_.reserve(nthreads);
            thread_scratch_.reserve(nthreads);
            thread_metrics_.reserve(nthreads);

            for (std::size_t i = thread_currents_.size(); i < nthreads; ++i)
            {
                thread_currents_.emplace_back(grid);
                thread_scratch_.emplace_back();
                thread_metrics_.emplace_back();
            }
        }
    }

    // CPU pushers
    void advance_fused_cpu(const Grid& grid, double dt)
    {
        // clang-format off
        #pragma omp parallel
        // clang-format on
        {
            const int tid            = omp_get_thread_num();
            auto&     thread_current = thread_currents_[tid];
            auto&     thread_scratch = thread_scratch_[tid];
            auto&     thread_metrics = thread_metrics_[tid];

            thread_current.zero_out();
            thread_metrics.reset();

            // clang-format off
            #pragma omp for schedule(static)
            // clang-format on
            for (auto& block : particles_)
            {
                step_particle_block_fused(block, thread_scratch, thread_current, thread_metrics, grid, dt);
            }
        }
    }

    // GPU pushers
    void advance_staged_gpu(const Grid& grid, double dt)
    {
        for (auto& tc : thread_currents_)
            tc.zero_out();
        for (auto& tm : thread_metrics_)
            tm.reset();

        // clang-format off
        #pragma omp parallel
        // clang-format on
        {
            const int tid            = omp_get_thread_num();
            auto&     thread_scratch = thread_scratch_[tid];
            // clang-format off
            #pragma omp for schedule(static)
            // clang-format on
            for (auto& block : particles_)
            {
                gather_.gather_block(block, fields_, grid, thread_scratch);
                pusher_.push_block(block, thread_scratch, dt);
            }
        }

        execute_stage(pico::perf::Stage::Pusher, [&] { pusher_.sync(); });
        // clang-format off
        #pragma omp parallel
        // clang-format on
        {
            const int tid            = omp_get_thread_num();
            auto&     thread_current = thread_currents_[tid];
            auto&     thread_metrics = thread_metrics_[tid];
            // clang-format off
            #pragma omp for schedule(static)
            // clang-format on
            for (auto& block : particles_)
            {
                particle_boundary_.apply(block, grid);
                deposit_.deposit_block(block, thread_current, grid, dt, particles_per_cell_);

                if (enable_locality_diagnostics_)
                {
                    pico::diagnostics::LocalityProfiler<particle::ParticleBlock<BLOCK_SIZE>>::process_block(block, grid, thread_metrics);
                }
            }
        }
    }

    void step_particle_block_fused(auto& block, auto& scratch, auto& current, auto& thread_metrics, const Grid& grid, double dt)
    {
        const uint64_t t0 = enable_stage_profiling_ ? pico::perf::read_cpu_ticks() : 0;
        gather_.gather_block(block, fields_, grid, scratch);

        const uint64_t t1 = enable_stage_profiling_ ? pico::perf::read_cpu_ticks() : 0;
        pusher_.push_block(block, scratch, dt);
        particle_boundary_.apply(block, grid);

        const uint64_t t2 = enable_stage_profiling_ ? pico::perf::read_cpu_ticks() : 0;
        deposit_.deposit_block(block, current, grid, dt, particles_per_cell_);

        if (enable_stage_profiling_)
        {
            const uint64_t t3 = pico::perf::read_cpu_ticks();
            profiler_.add_ticks(pico::perf::Stage::Gather, t1 - t0);
            profiler_.add_ticks(pico::perf::Stage::Pusher, t2 - t1);
            profiler_.add_ticks(pico::perf::Stage::Deposit, t3 - t2);
        }
        if (enable_locality_diagnostics_)
        {
            pico::diagnostics::LocalityProfiler<particle::ParticleBlock<BLOCK_SIZE>>::process_block(block, grid, thread_metrics);
        }
    }

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

    std::vector<FieldSystem<BLOCK_SIZE>>            thread_currents_;
    std::vector<FieldScratch<BLOCK_SIZE>>           thread_scratch_;
    std::vector<pico::diagnostics::LocalityMetrics> thread_metrics_;

    std::size_t particles_per_cell_{10};

    pico::perf::PipelineProfiler profiler_{};
    std::size_t                  step_counter_{0};
    std::size_t                  sort_frequency_{0};

    pico::diagnostics::LocalityMetrics locality_metrics_{};
    double                             locality_threshold_{0.0};

    bool enable_locality_diagnostics_{true};
    bool enable_stage_profiling_{true};
};

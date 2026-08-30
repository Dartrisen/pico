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

#include <cmath>
#include <concepts>
#include <cstdint>
#include <omp.h>
#include <stdexcept>
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
    requires pico::modules::FieldSolver<FieldSolverT, EMFields<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>, FieldBoundaryT> &&
             pico::modules::Gather<GatherT, particle::ParticleBlock<BLOCK_SIZE>, EMFields<BLOCK_SIZE>, FieldScratch<BLOCK_SIZE>> &&
             pico::modules::Pusher<PusherT, particle::ParticleBlock<BLOCK_SIZE>, FieldScratch<BLOCK_SIZE>> &&
             pico::modules::Deposit<DepositT, particle::ParticleBlock<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::FieldBoundary<FieldBoundaryT, EMFields<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::ParticleBoundary<ParticleBoundaryT, particle::ParticleBlock<BLOCK_SIZE>>
class PICEngine final : public EngineBase<PICEngine<FieldSolverT, GatherT, PusherT, DepositT, FieldBoundaryT, ParticleBoundaryT, FieldInjectorT, BLOCK_SIZE>>
{
public:
    using ParticleSystemType = particle::ParticleSystem<BLOCK_SIZE>;
    using ParticleBlockType  = particle::ParticleBlock<BLOCK_SIZE>;

    PICEngine(PICEngine&&) noexcept            = default;
    PICEngine& operator=(PICEngine&&) noexcept = default;
    PICEngine(const PICEngine&)                = delete;
    PICEngine& operator=(const PICEngine&)     = delete;

    explicit PICEngine(const Grid& grid, FieldBoundaryT field_boundary = FieldBoundaryT{}, ParticleBoundaryT particle_boundary = ParticleBoundaryT{},
                       FieldInjectorT field_injector = FieldInjectorT{})
            : fields_(grid), current_(grid), field_solver_{}, pusher_{}, gather_{}, deposit_{}, field_boundary_(std::move(field_boundary)),
              particle_boundary_(std::move(particle_boundary)), field_injector_(std::move(field_injector))
    {
    }

    /**
     * @brief Convenience constructor for a single uniform cold species.
     */
    PICEngine(const Grid& grid, std::size_t particles_per_cell, float n0 = 1.0f, float base_charge = -1.0f, float base_mass = 1.0f) : PICEngine(grid)
    {
        add_species_uniform(particles_per_cell, n0, base_charge, base_mass);
    }

    /**
     * @brief Convenience constructor for a single uniform thermal species (Maxwellian velocity distribution).
     */
    PICEngine(const Grid& grid, std::size_t particles_per_cell, float v_th, float n0, float base_charge = -1.0f, float base_mass = 1.0f) : PICEngine(grid)
    {
        add_species_thermal(particles_per_cell, v_th, n0, base_charge, base_mass);
    }

    ParticleSystemType& add_species(ParticleSystemType species, std::size_t particles_per_cell)
    {
        if (particles_per_cell == 0)
        {
            throw std::invalid_argument("PICEngine: particles_per_cell must be greater than zero");
        }

        species_.push_back(std::move(species));
        species_ppc_.push_back(particles_per_cell);
        return species_.back();
    }

    /**
     * @brief Add a species with uniform spatial density and cold velocity.
     */
    ParticleSystemType& add_species_uniform(std::size_t particles_per_cell, float n0 = 1.0f, float base_charge = -1.0f, float base_mass = 1.0f, float vx = 0.0f, float vy = 0.0f,
                                            float vz = 0.0f)
    {
        return add_species(particles_per_cell, [n0](double) { return n0; }, base_charge, base_mass, vx, vy, vz);
    }

    /**
     * @brief Add a species with a spatial density profile function and uniform drift velocity.
     */
    template <typename DensityFunc>
        requires std::invocable<DensityFunc, double>
    ParticleSystemType& add_species(std::size_t particles_per_cell, DensityFunc&& density_fn, float base_charge = -1.0f, float base_mass = 1.0f, float vx = 0.0f, float vy = 0.0f,
                                    float vz = 0.0f)
    {
        if (particles_per_cell == 0)
        {
            throw std::invalid_argument("PICEngine: particles_per_cell must be greater than zero");
        }

        const Grid&       grid            = fields_.E.grid();
        const std::size_t total_particles = grid.physical_size() * particles_per_cell;

        species_.emplace_back(total_particles, base_charge, base_mass);

        auto& sp = species_.back();
        sp.set_active(total_particles);
        sp.init_positions_uniform(grid);
        sp.init_density_profile(grid, std::forward<DensityFunc>(density_fn));
        sp.init_velocities_cold(vx, vy, vz);

        species_ppc_.push_back(particles_per_cell);
        return sp;
    }

    /**
     * @brief Add a species with uniform spatial density and thermal Maxwellian velocity distribution.
     */
    ParticleSystemType& add_species_thermal(std::size_t particles_per_cell, float v_th, float n0 = 1.0f, float base_charge = -1.0f, float base_mass = 1.0f, float vx_drift = 0.0f,
                                            float vy_drift = 0.0f, float vz_drift = 0.0f, uint32_t seed = 1337)
    {
        return add_species_thermal(particles_per_cell, [n0](double) { return n0; }, v_th, base_charge, base_mass, vx_drift, vy_drift, vz_drift, seed);
    }

    /**
     * @brief Add a species with a spatial density profile and thermal Maxwellian velocity distribution.
     */
    template <typename DensityFunc>
        requires std::invocable<DensityFunc, double>
    ParticleSystemType& add_species_thermal(std::size_t particles_per_cell, DensityFunc&& density_fn, float v_th, float base_charge = -1.0f, float base_mass = 1.0f,
                                            float vx_drift = 0.0f, float vy_drift = 0.0f, float vz_drift = 0.0f, uint32_t seed = 1337)
    {
        if (particles_per_cell == 0)
        {
            throw std::invalid_argument("PICEngine: particles_per_cell must be greater than zero");
        }

        const Grid&       grid            = fields_.E.grid();
        const std::size_t total_particles = grid.physical_size() * particles_per_cell;

        species_.emplace_back(total_particles, base_charge, base_mass);

        auto& sp = species_.back();
        sp.set_active(total_particles);
        sp.init_positions_uniform(grid);
        sp.init_density_profile(grid, std::forward<DensityFunc>(density_fn));
        sp.init_velocities_thermal(v_th, vx_drift, vy_drift, vz_drift, seed);

        species_ppc_.push_back(particles_per_cell);
        return sp;
    }

    /**
     * @brief Add a species with spatial density and custom velocity profile functions.
     */
    template <typename DensityFunc, typename VelFunc>
        requires std::invocable<DensityFunc, double> && std::invocable<VelFunc, double>
    ParticleSystemType& add_species_profile(std::size_t particles_per_cell, DensityFunc&& density_fn, VelFunc&& vel_fn, float base_charge = -1.0f, float base_mass = 1.0f)
    {
        if (particles_per_cell == 0)
        {
            throw std::invalid_argument("PICEngine: particles_per_cell must be greater than zero");
        }

        const Grid&       grid            = fields_.E.grid();
        const std::size_t total_particles = grid.physical_size() * particles_per_cell;

        species_.emplace_back(total_particles, base_charge, base_mass);

        auto& sp = species_.back();
        sp.set_active(total_particles);
        sp.init_positions_uniform(grid);
        sp.init_density_profile(grid, std::forward<DensityFunc>(density_fn));
        sp.init_velocities_profile(std::forward<VelFunc>(vel_fn));

        species_ppc_.push_back(particles_per_cell);
        return sp;
    }

    /**
     * @brief Add a thermal species with spatially varying drift profile function.
     */
    template <typename DensityFunc, typename DriftFunc>
        requires std::invocable<DensityFunc, double> && std::invocable<DriftFunc, double>
    ParticleSystemType& add_species_thermal_profile(std::size_t particles_per_cell, DensityFunc&& density_fn, float v_th, DriftFunc&& drift_fn, float base_charge = -1.0f,
                                                    float base_mass = 1.0f, uint32_t seed = 1337)
    {
        if (particles_per_cell == 0)
        {
            throw std::invalid_argument("PICEngine: particles_per_cell must be greater than zero");
        }

        const Grid&       grid            = fields_.E.grid();
        const std::size_t total_particles = grid.physical_size() * particles_per_cell;

        species_.emplace_back(total_particles, base_charge, base_mass);

        auto& sp = species_.back();
        sp.set_active(total_particles);
        sp.init_positions_uniform(grid);
        sp.init_density_profile(grid, std::forward<DensityFunc>(density_fn));
        sp.init_velocities_thermal(v_th, std::forward<DriftFunc>(drift_fn), seed);

        species_ppc_.push_back(particles_per_cell);
        return sp;
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

        for (auto& species : species_)
        {
            if (species.active_particles() > 0 && (periodic_sort || stride_degraded))
            {
                execute_stage(pico::perf::Stage::Sorting, [&] { sorter_.sort(species, grid); });
            }
        }

        execute_stage(pico::perf::Stage::Boundaries, [&] { field_boundary_.fill_field_guards(fields_, grid); });

        current_.zero_out();
        ensure_thread_buffers(grid);
        // clang-format off
        #pragma omp parallel
        // clang-format on
        {
            const int tid = omp_get_thread_num();

            auto& thread_current = thread_currents_[tid];
            auto& thread_scratch = thread_scratch_[tid];
            auto& thread_metrics = thread_metrics_[tid];

            thread_current.zero_out();
            thread_metrics.reset();

            for (std::size_t s = 0; s < species_.size(); ++s)
            {
                auto&             species = species_[s];
                const std::size_t ppc     = species_ppc_[s];

                // clang-format off
                #pragma omp for schedule(static) nowait
                // clang-format on
                for (auto& block : species)
                {
                    step_particle_block(block, species, ppc, thread_scratch, thread_current, thread_metrics, grid, dt);
                }
            }
        }

        for (const auto& thread_current : thread_currents_)
        {
            current_.accumulate(thread_current);
        }

        locality_metrics_.reset();
        for (const auto& thread_metric : thread_metrics_)
        {
            locality_metrics_.accumulate(thread_metric);
        }

        execute_stage(pico::perf::Stage::Boundaries, [&] { field_boundary_.fold_currents(current_, grid); });

        execute_stage(pico::perf::Stage::FieldSolver,
                      [&]
                      {
                          field_solver_.solve(fields_, current_, field_boundary_, dt);
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

    void set_sort_frequency(std::size_t frequency) noexcept
    {
        sort_frequency_ = frequency;
    }

    void set_locality_threshold(double max_stride) noexcept
    {
        locality_threshold_ = max_stride;
    }

    [[nodiscard]] double mean_cell_stride() const noexcept
    {
        return locality_metrics_.mean_stride();
    }

    [[nodiscard]] const pico::diagnostics::LocalityMetrics& locality_metrics() const noexcept
    {
        return locality_metrics_;
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

    [[nodiscard]] const std::vector<ParticleSystemType>& species() const noexcept
    {
        return species_;
    }

    [[nodiscard]] std::size_t num_species() const noexcept
    {
        return species_.size();
    }

    [[nodiscard]] std::size_t total_particles() const noexcept
    {
        std::size_t total = 0;
        for (const auto& sp : species_)
        {
            total += sp.active_particles();
        }
        return total;
    }

    [[nodiscard]] EMFields<BLOCK_SIZE>& fields() noexcept
    {
        return fields_;
    }

    [[nodiscard]] const EMFields<BLOCK_SIZE>& fields() const noexcept
    {
        return fields_;
    }

    [[nodiscard]] FieldSystem<BLOCK_SIZE>& current() noexcept
    {
        return current_;
    }

    [[nodiscard]] const FieldSystem<BLOCK_SIZE>& current() const noexcept
    {
        return current_;
    }

    [[nodiscard]] const FieldInjectorT& injector() const noexcept
    {
        return field_injector_;
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

    void step_particle_block(ParticleBlockType& block, const ParticleSystemType& species, std::size_t ppc, FieldScratch<BLOCK_SIZE>& scratch, FieldSystem<BLOCK_SIZE>& current,
                             pico::diagnostics::LocalityMetrics& thread_metrics, const Grid& grid, double dt)
    {
        const float dt_f = static_cast<float>(dt);

        const uint64_t t0 = enable_stage_profiling_ ? pico::perf::read_cpu_ticks() : 0;
        gather_.gather_block(block, fields_, grid, scratch);

        const uint64_t t1 = enable_stage_profiling_ ? pico::perf::read_cpu_ticks() : 0;
        pusher_.push_block(block, scratch, dt_f, species.base_charge() / species.base_mass());

        particle_boundary_.apply(block, grid);

        const uint64_t t2 = enable_stage_profiling_ ? pico::perf::read_cpu_ticks() : 0;
        deposit_.deposit_block(block, current, grid, dt_f, static_cast<float>(ppc), species.base_charge());

        if (enable_stage_profiling_)
        {
            const uint64_t t3 = pico::perf::read_cpu_ticks();
            profiler_.add_ticks(pico::perf::Stage::Gather, t1 - t0);
            profiler_.add_ticks(pico::perf::Stage::Pusher, t2 - t1);
            profiler_.add_ticks(pico::perf::Stage::Deposit, t3 - t2);
        }

        if (enable_locality_diagnostics_)
        {
            pico::diagnostics::LocalityProfiler<ParticleBlockType>::process_block(block, grid, thread_metrics);
        }
    }

    EMFields<BLOCK_SIZE>    fields_;
    FieldSystem<BLOCK_SIZE> current_;

    std::vector<ParticleSystemType> species_;
    std::vector<std::size_t>        species_ppc_;

    FieldSolverT      field_solver_;
    PusherT           pusher_;
    GatherT           gather_;
    DepositT          deposit_;
    FieldBoundaryT    field_boundary_;
    ParticleBoundaryT particle_boundary_;
    FieldInjectorT    field_injector_;

    pico::modules::sorter::ParticleSorter<BLOCK_SIZE> sorter_;

    std::vector<FieldSystem<BLOCK_SIZE>>            thread_currents_;
    std::vector<FieldScratch<BLOCK_SIZE>>           thread_scratch_;
    std::vector<pico::diagnostics::LocalityMetrics> thread_metrics_;

    pico::perf::PipelineProfiler profiler_{};

    std::size_t step_counter_{0};
    std::size_t sort_frequency_{0};

    pico::diagnostics::LocalityMetrics locality_metrics_{};
    double                             locality_threshold_{0.0};

    bool enable_locality_diagnostics_{true};
    bool enable_stage_profiling_{true};
};

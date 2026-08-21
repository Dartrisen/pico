#pragma once

#include "EngineBase.hpp"
#include "data/field/include/field_em.hpp"
#include "data/field/include/field_system.hpp"
#include "data/particle/include/particle_system.hpp"
#include "engine/modules/boundary/PeriodicBoundary.hpp"
#include "engine/modules/boundary/SilverMuller.hpp"
#include "engine/modules/boundary/Thermalizing.hpp"
#include "engine/modules/concepts.hpp"

#include <cstdint>

template <class FieldSolverT, class GatherT, class PusherT, class DepositT, class FieldBoundaryT,
          class ParticleBoundaryT, std::size_t BLOCK_SIZE = 64>
    requires pico::modules::FieldSolver<FieldSolverT, EMFields<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::Gather<GatherT, particle::ParticleBlock<BLOCK_SIZE>, EMFields<BLOCK_SIZE>,
                                   FieldScratch<BLOCK_SIZE>> &&
             pico::modules::Pusher<PusherT, particle::ParticleBlock<BLOCK_SIZE>, FieldScratch<BLOCK_SIZE>> &&
             pico::modules::Deposit<DepositT, particle::ParticleBlock<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::FieldBoundary<FieldBoundaryT, EMFields<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::ParticleBoundary<ParticleBoundaryT, particle::ParticleBlock<BLOCK_SIZE>>
class PICEngine final
        : public EngineBase<
                  PICEngine<FieldSolverT, GatherT, PusherT, DepositT, FieldBoundaryT, ParticleBoundaryT, BLOCK_SIZE>>
{
public:
    // Grid + PPC constructor (Default-constructs template boundary types)
    explicit PICEngine(const Grid& grid, std::size_t particles_per_cell = 10)
            : PICEngine(grid, particles_per_cell, FieldBoundaryT{}, ParticleBoundaryT{})
    {
    }

    // full constructor accepting explicit boundary instances
    PICEngine(const Grid& grid, std::size_t particles_per_cell, FieldBoundaryT field_boundary,
              ParticleBoundaryT particle_boundary)
            : fields_(grid), current_(grid), scratch_(), particles_(grid.physical_size() * particles_per_cell),
              field_solver_{}, pusher_{}, gather_{}, deposit_{}, field_boundary_(std::move(field_boundary)),
              particle_boundary_(std::move(particle_boundary)),
              particles_per_cell_(static_cast<uint32_t>(particles_per_cell))
    {
        particles_.set_active(grid.physical_size() * particles_per_cell);
        particles_.init_positions_uniform(grid);
        particles_.init_velocities_cold(0.01f, 0.0f, 0.0f);
    }

    // constructor with pre-constructed ParticleSystem
    PICEngine(const Grid& grid, particle::ParticleSystem<BLOCK_SIZE> particles,
              FieldBoundaryT    field_boundary    = FieldBoundaryT{},
              ParticleBoundaryT particle_boundary = ParticleBoundaryT{})
            : fields_(grid), current_(grid), scratch_(), particles_(std::move(particles)), field_solver_{}, pusher_{},
              gather_{}, deposit_{}, field_boundary_(std::move(field_boundary)),
              particle_boundary_(std::move(particle_boundary)), particles_per_cell_(10)
    {
    }

    void advance_impl(double dt)
    {
        const Grid& grid = fields_.E.grid();

        // fill field guard cells prior to gathering
        field_boundary_.fill_field_guards(fields_, grid);

        // reset grid currents
        current_.zero_out();

        // block-by-block PIC iteration
        for (auto& block : particles_)
        {
            gather_.gather_block(block, fields_, grid, scratch_);
            pusher_.push_block(block, scratch_, dt);

            // kinetic particle boundary policy post-push
            particle_boundary_.apply(block, grid);

            deposit_.deposit_block(block, current_, grid, dt, particles_per_cell_);
        }

        // fold guard cell currents into physical mesh
        field_boundary_.fold_currents(current_, grid);

        // solve field equations
        field_solver_.solve(fields_, current_, dt);
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

private:
    EMFields<BLOCK_SIZE>                 fields_;
    FieldSystem<BLOCK_SIZE>              current_;
    FieldScratch<BLOCK_SIZE>             scratch_;
    particle::ParticleSystem<BLOCK_SIZE> particles_;

    FieldSolverT      field_solver_;
    PusherT           pusher_;
    GatherT           gather_;
    DepositT          deposit_;
    FieldBoundaryT    field_boundary_;
    ParticleBoundaryT particle_boundary_;

    uint32_t particles_per_cell_;
};
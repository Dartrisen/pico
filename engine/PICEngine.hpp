#pragma once

#include "EngineBase.hpp"
#include "data/field/include/field_em.hpp"
#include "data/field/include/field_system.hpp"
#include "data/particle/include/particle_system.hpp"
#include "engine/modules/boundary/PeriodicBoundary.hpp"
#include "engine/modules/concepts.hpp"

#include <cstdint>

template <class FieldSolverT, class GatherT, class PusherT, class DepositT, class BoundaryT, std::size_t BLOCK_SIZE>
    requires pico::modules::FieldSolver<FieldSolverT, EMFields<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::Gather<GatherT, particle::ParticleBlock<BLOCK_SIZE>, EMFields<BLOCK_SIZE>,
                                   FieldScratch<BLOCK_SIZE>> &&
             pico::modules::Pusher<PusherT, particle::ParticleBlock<BLOCK_SIZE>, FieldScratch<BLOCK_SIZE>> &&
             pico::modules::Deposit<DepositT, particle::ParticleBlock<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>>
class PICEngine final : public EngineBase<PICEngine<FieldSolverT, GatherT, PusherT, DepositT, BoundaryT, BLOCK_SIZE>>
{
public:
    explicit PICEngine(const Grid& grid, size_t particles_per_cell = 10)
            : fields_(grid), current_(grid), scratch_(), particles_(grid.physical_size() * particles_per_cell),
              field_solver_{}, pusher_{}, gather_{}, deposit_{}, boundary_{}
    {
        particles_.set_active(grid.physical_size() * particles_per_cell);
        particles_.init_positions_uniform(grid);
        particles_.init_velocities_cold(0.01f, 0.0f, 0.0f);
        particles_per_cell_ = particles_per_cell;
    }

    PICEngine(const Grid& grid, particle::ParticleSystem<BLOCK_SIZE> particles)
            : fields_(grid), current_(grid), scratch_(), particles_(std::move(particles)), field_solver_{}, pusher_{},
              gather_{}, deposit_{}, boundary_{}
    {
    }

    void advance_impl(double dt)
    {
        const Grid& grid = fields_.E.grid();

        // Step 1: Fill field guard cells before gathering
        boundary_.fill_field_guards(fields_, grid);

        // Step 2: Reset currents
        current_.zero_out();

        // Step 3: Block-by-block gather -> push -> deposit
        for (auto& block : particles_)
        {
            gather_.gather_block(block, fields_, grid, scratch_);
            pusher_.push_block(block, scratch_, dt);
            deposit_.deposit_block(block, current_, grid, dt, particles_per_cell_);
        }

        // Step 4: Fold guard currents into physical boundary cells
        boundary_.fold_currents(current_, grid);

        // Step 5: Advance Maxwell field equations
        field_solver_.solve(fields_, current_, dt);
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

private:
    EMFields<BLOCK_SIZE>                 fields_;
    FieldSystem<BLOCK_SIZE>              current_;
    FieldScratch<BLOCK_SIZE>             scratch_;
    particle::ParticleSystem<BLOCK_SIZE> particles_;

    FieldSolverT field_solver_;
    PusherT      pusher_;
    GatherT      gather_;
    DepositT     deposit_;
    BoundaryT    boundary_;

    uint32_t particles_per_cell_;
};
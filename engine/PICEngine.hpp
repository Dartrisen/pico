#pragma once

#include "EngineBase.hpp"
#include "data/field/include/field_em.hpp"
#include "data/field/include/field_system.hpp"
#include "data/particle/include/particle_system.hpp"
#include "engine/modules/concepts.hpp"

template <class FieldSolverT, class GatherT, class PusherT, size_t BLOCK_SIZE>
    requires pico::modules::FieldSolver<FieldSolverT, EMFields<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>> &&
             pico::modules::Gather<GatherT, particle::ParticleSystem<BLOCK_SIZE>, EMFields<BLOCK_SIZE>,
                                   FieldScratch<BLOCK_SIZE>> &&
             pico::modules::Pusher<PusherT, particle::ParticleSystem<BLOCK_SIZE>, FieldScratch<BLOCK_SIZE>>
class PICEngine final : public EngineBase<PICEngine<FieldSolverT, GatherT, PusherT, BLOCK_SIZE>>
{
public:
    explicit PICEngine(const Grid& grid)
            : fields_(grid), current_(grid), scratch_(), particles_(grid.size()), field_solver_{}, pusher_{}, gather_{}
    {
        particles_.set_active(grid.size());
    }
    void advance_impl(double dt)
    {
        const Grid& grid = fields_.E.grid();

        GatherT::gather(particles_, fields_, grid, scratch_);
        pusher_.push(particles_, scratch_, dt);
        field_solver_.solve(fields_, current_, dt);
    }

private:
    // Simulation state
    EMFields<BLOCK_SIZE>                 fields_;
    FieldSystem<BLOCK_SIZE>              current_;
    FieldScratch<BLOCK_SIZE>             scratch_;
    particle::ParticleSystem<BLOCK_SIZE> particles_;

    // Physics modules
    FieldSolverT field_solver_;
    PusherT      pusher_;
    GatherT      gather_;
};

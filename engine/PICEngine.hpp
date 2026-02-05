#pragma once
#include "engine/modules/concepts.hpp"
#include "field/include/field_em.hpp"
#include "field/include/field_system.hpp"
#include "EngineBase.hpp"

template<
    class FieldSolverT,
    //class Pusher,
    //class CollisionModel,
    //class CurrentDeposit
    size_t BLOCK_SIZE
>
requires pico::modules::FieldSolver<FieldSolverT, EMFields<BLOCK_SIZE>, FieldSystem<BLOCK_SIZE>>
         //pico::modules::Pusher<Pusher, Particles, Fields> &&
         //pico::modules::Collision<CollisionModel, Particles> &&
         //pico::modules::Deposit<CurrentDeposit, Particles, Fields>
class PICEngine final
    //: public EngineBase<PICEngine<FieldSolver, Pusher, CollisionModel, CurrentDeposit>>
    : public EngineBase<PICEngine<FieldSolverT, BLOCK_SIZE>>
{
public:
    explicit PICEngine(const Grid& grid)
            : fields_(grid)
            , current_(grid)
            , field_solver_{}   // stateless solver
        {}
    void advance_impl(double dt) {
        field_solver_.solve(fields_, current_, dt);
        //pusher_.push(particles_, fields_, dt);
        //collision_.apply(particles_, dt);
        //deposit_.deposit(particles_, fields_);
    }
private:
    // Simulation state
    // Grid grid_;
    EMFields<BLOCK_SIZE> fields_;
    FieldSystem<BLOCK_SIZE> current_;
    // Particles<BLOCK_SIZE> particles_;

    // Physics modules
    FieldSolverT field_solver_;
    //Pusher pusher_;
    //CollisionModel collision_;
    //CurrentDeposit deposit_;
};

// app/app.cpp
#include "app/EngineWrapper.hpp"
#include "app/IEngine.hpp"
#include "app/PICApp.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

int main(int argc, char** argv)
{
    // ---- runtime config ----
    Grid        grid(256, 0.1);
    double      dt     = 1e-3;
    std::size_t nsteps = 1000;

    // ---- compile-time config ----
    constexpr std::size_t BS = 64;

    // ---- compile-time modules choices ----
    using Shape     = kernels::shapes::Shape<1>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using Boundary  = pico::modules::boundary::PeriodicBoundaryHandler<BS>;
    using BoundaryP = pico::modules::boundary::ThermalizingParticleBoundary<BS>;
    // using Coll  = module::collision::NoCollision;

    using EngineT = PICEngine<Field, Gather, Push, Dep, Boundary, BoundaryP, BS>;

    // ---- bridge abstraction: runtime → compile-time ----
    auto engine = std::make_unique<EngineWrapper<EngineT>>(EngineT{grid, 10});

    // ---- high-level app ----
    PICApp app(std::move(engine), dt);
    app.run(nsteps);
    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}

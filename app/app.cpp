// app/app.cpp
#include "app/PICApp.hpp"
#include "app/IEngine.hpp"
#include "app/EngineWrapper.hpp"

#include "engine/PICEngine.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "engine/modules/gather/Gather.hpp"

#include "kernels/shapes/shape.hpp"

#include "data/grid/include/grid.hpp"

int main(int argc, char **argv)
{
    // ---- runtime config ----
    Grid grid(256, 0.1);
    double dt = 1e-3;
    std::size_t nsteps = 1000;

    // ---- compile-time config ----
    constexpr std::size_t BS = 64;

    // ---- compile-time modules choices ----
    using Shape = kernels::shapes::Shape<1>;
    using Field = pico::modules::field::YeeMaxwell<BS>;
    using Push = pico::modules::pusher::BorisPusher<BS>;
    using Gather = pico::modules::gather::Gather<Shape, BS>;
    // using Coll  = module::collision::NoCollision;
    // using Dep   = module::deposit::Esirkepov<BS>;

    using EngineT = PICEngine<
        Field,
        Gather,
        Push,
        // Coll,
        // Dep,
        BS>;

    // ---- bridge abstraction: runtime → compile-time ----
    auto engine = std::make_unique<EngineWrapper<EngineT>>(EngineT{grid});

    // ---- high-level app ----
    PICApp app(std::move(engine), dt);
    app.run(nsteps);
    std::cout << "Simulation completed successfully." << std::endl;
    return 0;
}

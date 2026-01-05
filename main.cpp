#include <iostream>
#include <format>

#include "particle_system.hpp"
#include "field_system.hpp"
#include "operators.hpp"

int main()
{
    constexpr size_t BS = 8;

    Grid grid(64, 1.0f);
    EMFields<BS> fields(grid);
    FieldSystem<BS> J(grid);

    // // Uniform E field
    fields.E.set_field_x(1.0f);

    ParticleSystem<BS> particles(1);
    particles.set_active(1);  // one active particle

    auto* b = particles.Blocks();
    b[0].position_x[0] = 0.1f;
    b[0].momentum_x[0] = 0.0f;
    b[0].momentum_y[0] = 0.0f;
    b[0].momentum_z[0] = 0.0f;

    MaxwellSolver<BS> maxwell;

    const float dt = 0.1f;

    for (int step = 0; step < 100; ++step) {
        J.for_each_block([](auto& block, size_t /*b*/) {
            for (size_t i = 0; i < BS; ++i) {
                block.field_x[i] = 0.0f;
                block.field_y[i] = 0.0f;
                block.field_z[i] = 0.0f;
            }
        });
        particles.advance(fields, J, grid, dt);
        maxwell(fields, J, grid, dt);

        std::cout << std::format("{:2} x={:.4f}\n", step, b[0].position_x[0]);
    }
    return 0;
}

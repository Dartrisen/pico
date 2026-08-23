#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_block.hpp"
#include "kernels/pusher/relativistic_boris.hpp"

#include <cmath>
#include <gtest/gtest.h>

class RelativisticBorisPusherTest : public ::testing::Test
{
protected:
    static constexpr size_t BS = 32;

    particle::ParticleBlock<BS> create_particle(float px, float py, float pz, float x = 0.0f, float m = 1.0f, float q = 1.0f)
    {
        particle::ParticleBlock<BS> block{};
        block.activeCount   = 1;
        block.mass[0]       = m;
        block.charge[0]     = q;
        block.momentum_x[0] = px;
        block.momentum_y[0] = py;
        block.momentum_z[0] = pz;
        block.position_x[0] = x;
        return block;
    }

    FieldScratch<BS> create_field(float ex, float ey, float ez, float bx, float by, float bz)
    {
        FieldScratch<BS> scratch{};
        for (size_t i = 0; i < BS; ++i)
        {
            scratch.Ex[i] = ex;
            scratch.Ey[i] = ey;
            scratch.Ez[i] = ez;
            scratch.Bx[i] = bx;
            scratch.By[i] = by;
            scratch.Bz[i] = bz;
        }
        return scratch;
    }

    void simulate(particle::ParticleBlock<BS>& block, FieldScratch<BS>& scratch, float dt, int steps)
    {
        for (int i = 0; i < steps; ++i)
        {
            kernels::pusher::RelativisticBorisPusher<BS>::push_block(block, scratch, dt);
        }
    }

    float get_kinetic_energy(const particle::ParticleBlock<BS>& block)
    {
        const float px    = block.momentum_x[0];
        const float py    = block.momentum_y[0];
        const float pz    = block.momentum_z[0];
        const float m     = block.mass[0];
        const float gamma = std::sqrt(1.0f + (px * px + py * py + pz * pz) / (m * m));
        return (gamma - 1.0f) * m;
    }
};

TEST_F(RelativisticBorisPusherTest, VelocityCappingUnderContinuousAcceleration)
{
    auto block   = create_particle(0.1f, 0.0f, 0.0f);
    auto scratch = create_field(50.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    simulate(block, scratch, 0.05f, 200);

    const float px    = block.momentum_x[0];
    const float py    = block.momentum_y[0];
    const float pz    = block.momentum_z[0];
    const float m     = block.mass[0];
    const float gamma = std::sqrt(1.0f + (px * px + py * py + pz * pz) / (m * m));
    const float vx    = px / (gamma * m);

    EXPECT_LT(vx, 1.0f);
    EXPECT_GT(vx, 0.95f);
}

TEST_F(RelativisticBorisPusherTest, ZeroFieldMomentumConservation)
{
    auto block   = create_particle(0.6f, 0.3f, -0.2f, 2.0f);
    auto scratch = create_field(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    const float init_px = block.momentum_x[0];
    const float init_py = block.momentum_y[0];
    const float init_pz = block.momentum_z[0];
    const float init_x  = block.position_x[0];
    const float dt      = 0.02f;

    simulate(block, scratch, dt, 1);

    EXPECT_NEAR(block.momentum_x[0], init_px, 1e-6f);
    EXPECT_NEAR(block.momentum_y[0], init_py, 1e-6f);
    EXPECT_NEAR(block.momentum_z[0], init_pz, 1e-6f);

    const float gamma      = std::sqrt(1.0f + (init_px * init_px + init_py * init_py + init_pz * init_pz) / (block.mass[0] * block.mass[0]));
    const float expected_x = init_x + dt * init_px / (gamma * block.mass[0]);

    EXPECT_NEAR(block.position_x[0], expected_x, 1e-6f);
}

TEST_F(RelativisticBorisPusherTest, EnergyConservationAfterFieldRemoval)
{
    auto block          = create_particle(0.2f, 0.1f, 0.0f);
    auto active_scratch = create_field(2.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.5f);
    auto zero_scratch   = create_field(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    const float dt = 0.02f;

    simulate(block, active_scratch, dt, 50);
    const float energy_before = get_kinetic_energy(block);

    simulate(block, zero_scratch, dt, 50);
    const float energy_after = get_kinetic_energy(block);

    EXPECT_NEAR(energy_before, energy_after, 1e-5f);
}
#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_block.hpp"
#include "kernels/pusher/relativistic_boris.hpp"

#include <cmath>
#include <gtest/gtest.h>

class RelativisticBorisPusherTest : public ::testing::Test
{
protected:
    static constexpr size_t BS = 32;
};

TEST_F(RelativisticBorisPusherTest, VelocityCappingUnderContinuousAcceleration)
{
    particle::ParticleBlock<BS> block{};
    block.activeCount = 1;

    block.mass[0]       = 1.0f;
    block.charge[0]     = 1.0f;
    block.momentum_x[0] = 0.1f;
    block.momentum_y[0] = 0.0f;
    block.momentum_z[0] = 0.0f;
    block.position_x[0] = 0.0f;

    FieldScratch<BS> scratch{};
    // Apply a massive and continuous electric field to push past classical limits
    scratch.Ex[0] = 50.0f;
    scratch.Ey[0] = 0.0f;
    scratch.Ez[0] = 0.0f;
    scratch.Bx[0] = 0.0f;
    scratch.By[0] = 0.0f;
    scratch.Bz[0] = 0.0f;

    const float dt = 0.05f;
    const float c  = 1.0f;

    // Simulate over 200 steps
    for (int step = 0; step < 200; ++step)
    {
        kernels::pusher::RelativisticBorisPusher<BS>::push_block(block, scratch, dt);
    }

    // Extract final velocity: v = p / (gamma * m)
    const float px = block.momentum_x[0];
    const float py = block.momentum_y[0];
    const float pz = block.momentum_z[0];
    const float m  = block.mass[0];

    const float p_sq  = px * px + py * py + pz * pz;
    const float gamma = std::sqrt(1.0f + p_sq / (m * m));
    const float vx    = px / (gamma * m);

    // Velocity must strictly remain bounded below c
    EXPECT_LT(vx, c);
    // With heavy acceleration, it should push close to the speed of light limit
    EXPECT_GT(vx, 0.95f);
}

TEST_F(RelativisticBorisPusherTest, ZeroFieldMomentumConservation)
{
    particle::ParticleBlock<BS> block{};
    block.activeCount = 1;

    block.mass[0]       = 1.0f;
    block.charge[0]     = 1.0f;
    block.momentum_x[0] = 0.6f;
    block.momentum_y[0] = 0.3f;
    block.momentum_z[0] = -0.2f;
    block.position_x[0] = 2.0f;

    FieldScratch<BS> scratch{};
    // Zero out all fields
    for (size_t i = 0; i < BS; ++i)
    {
        scratch.Ex[i] = 0.0f;
        scratch.Ey[i] = 0.0f;
        scratch.Ez[i] = 0.0f;
        scratch.Bx[i] = 0.0f;
        scratch.By[i] = 0.0f;
        scratch.Bz[i] = 0.0f;
    }

    const float initial_px = block.momentum_x[0];
    const float initial_py = block.momentum_y[0];
    const float initial_pz = block.momentum_z[0];
    const float initial_x  = block.position_x[0];
    const float dt         = 0.02f;

    kernels::pusher::RelativisticBorisPusher<BS>::push_block(block, scratch, dt);

    // Momentum must be completely unchanged in zero fields
    EXPECT_NEAR(block.momentum_x[0], initial_px, 1e-6f);
    EXPECT_NEAR(block.momentum_y[0], initial_py, 1e-6f);
    EXPECT_NEAR(block.momentum_z[0], initial_pz, 1e-6f);

    // Check position progression matches relativistic free-streaming: x += dt * (p / (gamma * m))
    const float p_sq       = initial_px * initial_px + initial_py * initial_py + initial_pz * initial_pz;
    const float gamma      = std::sqrt(1.0f + p_sq / (block.mass[0] * block.mass[0]));
    const float expected_x = initial_x + dt * initial_px / (gamma * block.mass[0]);

    EXPECT_NEAR(block.position_x[0], expected_x, 1e-6f);
}

TEST_F(RelativisticBorisPusherTest, EnergyConservationAfterFieldRemoval)
{
    particle::ParticleBlock<BS> block{};
    block.activeCount = 1;

    block.mass[0]       = 1.0f;
    block.charge[0]     = 1.0f;
    block.momentum_x[0] = 0.2f;
    block.momentum_y[0] = 0.1f;
    block.momentum_z[0] = 0.0f;
    block.position_x[0] = 0.0f;

    FieldScratch<BS> scratch{};
    // Phase 1: Active electromagnetic field
    scratch.Ex[0] = 2.0f;
    scratch.Ey[0] = 1.0f;
    scratch.Ez[0] = 0.0f;
    scratch.Bx[0] = 0.0f;
    scratch.By[0] = 0.0f;
    scratch.Bz[0] = 1.5f;

    const float dt = 0.02f;

    // Push while the field is active (particle accelerates/moves)
    for (int step = 0; step < 50; ++step)
    {
        kernels::pusher::RelativisticBorisPusher<BS>::push_block(block, scratch, dt);
    }

    // Helper lambda to calculate relativistic kinetic energy: E_k = (gamma - 1) * m
    auto get_kinetic_energy = [](const particle::ParticleBlock<BS>& b, float m_val)
    {
        const float px    = b.momentum_x[0];
        const float py    = b.momentum_y[0];
        const float pz    = b.momentum_z[0];
        const float p_sq  = px * px + py * py + pz * pz;
        const float gamma = std::sqrt(1.0f + p_sq / (m_val * m_val));
        return (gamma - 1.0f) * m_val;
    };

    const float energy_before = get_kinetic_energy(block, block.mass[0]);

    // Phase 2: Turn off the field completely (field is gone)
    scratch.Ex[0] = 0.0f;
    scratch.Ey[0] = 0.0f;
    scratch.Ez[0] = 0.0f;
    scratch.Bx[0] = 0.0f;
    scratch.By[0] = 0.0f;
    scratch.Bz[0] = 0.0f;

    // Push further with zero field (particle free-streams)
    for (int step = 0; step < 50; ++step)
    {
        kernels::pusher::RelativisticBorisPusher<BS>::push_block(block, scratch, dt);
    }

    const float energy_after = get_kinetic_energy(block, block.mass[0]);

    // Kinetic energy must be locked and perfectly conserved after the field is removed
    EXPECT_NEAR(energy_before, energy_after, 1e-5f);
}
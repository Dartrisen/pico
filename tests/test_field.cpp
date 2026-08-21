// #include "data/field/include/field_system.hpp"
// #include "data/grid/include/grid.hpp"
// #include "kernels/field/maxwell_yee.hpp"

// #include <cmath>
// #include <gtest/gtest.h>

// TEST(YeeMaxwellTest, StandingWaveConvergence)
// {
//     constexpr std::size_t N     = 500;
//     constexpr std::size_t G     = 2;
//     constexpr double      dx    = 0.01;
//     constexpr double      dt    = 0.001;
//     constexpr double      omega = 2.0 * M_PI;
//     constexpr double      k     = omega;

//     Grid grid(N, dx, G);

//     // Pass Grid directly into FieldSystem constructor
//     FieldSystem<64> E(grid);
//     FieldSystem<64> B(grid);
//     FieldSystem<64> J(grid);

//     for (std::size_t i = 0; i < grid.total_size(); ++i)
//     {
//         double x_E   = (static_cast<double>(i) - static_cast<double>(G)) * dx;
//         E.field_y(i) = static_cast<float>(std::sin(k * x_E));
//         B.field_z(i) = 0.0f;
//     }

//     kernels::field::YeeMaxwell<64>::advance_magnetic_field(B, E, grid, static_cast<float>(0.5 * dt));
//     kernels::field::YeeMaxwell<64>::advance_electric_field(E, B, J, grid, static_cast<float>(dt));

//     double max_error_E = 0.0;
//     for (std::size_t i = G; i < G + N; ++i)
//     {
//         double x_E     = (static_cast<double>(i) - static_cast<double>(G)) * dx;
//         double exact_E = std::sin(k * x_E) * std::cos(omega * dt);
//         double error   = std::abs(E.field_y(i) - exact_E);
//         max_error_E    = std::max(max_error_E, error);
//     }

//     EXPECT_LT(max_error_E, 1e-3);
// }

#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "kernels/field/maxwell_yee.hpp"

#include <cmath>
#include <gtest/gtest.h>

TEST(YeeMaxwellTest, StandingWaveConvergence)
{
    constexpr std::size_t N     = 500;
    constexpr std::size_t G     = 2;
    constexpr double      dx    = 0.01;
    constexpr double      dt    = 0.001;
    constexpr double      omega = 2.0 * M_PI;
    constexpr double      k     = omega;

    Grid grid(N, dx, G);

    FieldSystem<64> E(grid);
    FieldSystem<64> B(grid);
    FieldSystem<64> J(grid);

    for (std::size_t i = 0; i < grid.total_size(); ++i)
    {
        double x_E   = (static_cast<double>(i) - static_cast<double>(G)) * dx;
        E.field_y(i) = static_cast<float>(std::sin(k * x_E));
        B.field_z(i) = 0.0f;
    }

    kernels::field::YeeMaxwell<64>::advance_magnetic_field(B, E, grid, static_cast<float>(0.5 * dt));
    kernels::field::YeeMaxwell<64>::advance_electric_field(E, B, J, grid, static_cast<float>(dt));

    double max_error_E = 0.0;
    for (std::size_t i = G; i < G + N; ++i)
    {
        double x_E     = (static_cast<double>(i) - static_cast<double>(G)) * dx;
        double exact_E = std::sin(k * x_E) * std::cos(omega * dt);
        double error   = std::abs(E.field_y(i) - exact_E);
        max_error_E    = std::max(max_error_E, error);
    }

    EXPECT_LT(max_error_E, 1e-3);
}

TEST(YeeMaxwellTest, CurrentSourceDrive)
{
    constexpr std::size_t N  = 100;
    constexpr std::size_t G  = 2;
    constexpr double      dx = 0.1;
    constexpr float       dt = 0.01f;

    Grid grid(N, dx, G);

    FieldSystem<64> E(grid);
    FieldSystem<64> B(grid);
    FieldSystem<64> J(grid);

    // Uniform current density across domain
    for (std::size_t i = 0; i < grid.total_size(); ++i)
    {
        J.field_x(i) = 0.5f;
        J.field_y(i) = 1.0f;
        J.field_z(i) = 2.0f;
    }

    // With zero initial B field, dE/dt = -J
    kernels::field::YeeMaxwell<64>::advance_electric_field(E, B, J, grid, dt);

    for (std::size_t i = G; i < G + N; ++i)
    {
        EXPECT_NEAR(E.field_x(i), -dt * 0.5f, 1e-6f);
        EXPECT_NEAR(E.field_y(i), -dt * 1.0f, 1e-6f);
        EXPECT_NEAR(E.field_z(i), -dt * 2.0f, 1e-6f);
    }
}

TEST(YeeMaxwellTest, EnergyConservationInFreeSpace)
{
    constexpr std::size_t N  = 400;
    constexpr std::size_t G  = 2;
    constexpr double      dx = 0.05;
    constexpr float       dt = 0.002f;

    Grid grid(N, dx, G);

    FieldSystem<64> E(grid);
    FieldSystem<64> B(grid);
    FieldSystem<64> J(grid);

    // Initialize pulse centered at cell 200
    const double x_center = 200.0 * dx;
    for (std::size_t i = 0; i < grid.total_size(); ++i)
    {
        double x_E = (static_cast<double>(i) - static_cast<double>(G)) * dx;
        double x_B = x_E + 0.5 * dx;

        E.field_y(i) = static_cast<float>(std::exp(-std::pow((x_E - x_center) / 1.0, 2.0)));
        B.field_z(i) = static_cast<float>(std::exp(-std::pow((x_B - x_center) / 1.0, 2.0)));
    }

    auto compute_total_energy = [&]()
    {
        double energy = 0.0;
        for (std::size_t i = G; i < G + N; ++i)
        {
            double ey = E.field_y(i);
            double bz = B.field_z(i);
            energy += 0.5 * (ey * ey + bz * bz) * dx;
        }
        return energy;
    };

    const double initial_energy = compute_total_energy();

    // Propagate wave for 100 steps (stays away from domain boundaries)
    for (int step = 0; step < 100; ++step)
    {
        kernels::field::YeeMaxwell<64>::advance_magnetic_field(B, E, grid, dt);
        kernels::field::YeeMaxwell<64>::advance_electric_field(E, B, J, grid, dt);
    }

    const double final_energy = compute_total_energy();
    const double rel_diff     = std::abs(final_energy - initial_energy) / initial_energy;

    EXPECT_LT(rel_diff, 1e-3);
}
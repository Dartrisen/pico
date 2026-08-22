#pragma once

#include "field_em.hpp"
#include "field_system.hpp"
#include "grid.hpp"
#include "particle_block.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <tuple>

constexpr size_t CACHE_LINE = 64;

enum ParticleValue
{
    mass   = 1,
    charge = -1,
    weight = 1
};

namespace particle
{

template <size_t BLOCK_SIZE>
class ParticleSystem
{
public:
    // ---- Constructors & Destructor ----
    explicit ParticleSystem(size_t maxParticles);
    ParticleSystem(size_t maxParticles, float charge, float mass);
    ~ParticleSystem();

    ParticleSystem(const ParticleSystem&)            = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;
    ParticleSystem(ParticleSystem&&) noexcept;
    ParticleSystem& operator=(ParticleSystem&&) noexcept;

    // ---- Accessors & Iterators ----
    size_t                  active_particles() const;
    static constexpr size_t block_size();
    size_t                  max_particles() const;
    size_t                  num_blocks() const;

    auto begin() noexcept
    {
        return blocks_;
    }
    auto end() noexcept
    {
        return blocks_ + numBlocks_;
    }
    auto begin() const noexcept
    {
        return blocks_;
    }
    auto end() const noexcept
    {
        return blocks_ + numBlocks_;
    }

    // ---- Density Initializers ----
    void init_density_constant(float n0 = 1.0f, float base_charge = -1.0f, float base_mass = 1.0f);

    template <typename DensityFunc>
    void init_density_profile(const Grid& grid, DensityFunc&& density_fn, float base_charge = -1.0f, float base_mass = 1.0f);

    // ---- Position Initializers ----
    void init_positions_uniform(double domain_length);
    void init_positions_uniform(const Grid& grid);

    // ---- Velocity Initializers ----
    void init_velocities_cold(float vx = 0.0f, float vy = 0.0f, float vz = 0.0f);

    template <typename VelFunc>
    void init_velocities_profile(VelFunc&& vel_fn);

    void init_velocities_thermal(float v_th, float v_drift_x = 0.0f, float v_drift_y = 0.0f, float v_drift_z = 0.0f, uint32_t seed = 42);

    template <typename DriftFunc>
    void init_velocities_thermal(float v_th, DriftFunc&& drift_func, uint32_t seed = 42);

    void init_velocities_wave(float v0, double k, bool longitudinal_only = true);

    // ---- Mutators & Modifiers ----
    void set_active(size_t n);
    void set_block_value(particle::ParticleBlock<BLOCK_SIZE>& block, ParticleValue value);

private:
    size_t maxParticles_{0};
    size_t numBlocks_{0};
    size_t activeParticles_{0};

    particle::ParticleBlock<BLOCK_SIZE>* blocks_{nullptr};
};

} // namespace particle

#include "data/particle/src/particle_system.inl"

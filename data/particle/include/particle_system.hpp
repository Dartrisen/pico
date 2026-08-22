#pragma once

#include "field_em.hpp"
#include "field_system.hpp"
#include "grid.hpp"
#include "particle_block.hpp"

#include <algorithm>
#include <iostream>

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
    ParticleSystem(size_t maxParticles);
    explicit ParticleSystem(size_t maxParticles, float charge, float mass);
    ~ParticleSystem();

    void set_active(size_t n);

    // ---- Particle State Initializers ----
    void init_positions_uniform(double domain_length);
    void init_positions_uniform(const Grid& grid);
    void init_velocities_cold(float vx = 0.0f, float vy = 0.0f, float vz = 0.0f);
    void init_velocities_thermal(float v_th, float v_drift_x = 0.0f, float v_drift_y = 0.0f, float v_drift_z = 0.0f, uint32_t seed = 42);
    void init_velocities_wave(float v0, double k, bool longitudinal_only = true);
    template <typename VelFunc>
    void init_velocities_profile(VelFunc&& vel_fn);

    // ---- Density & Weight Initializers ----
    void init_density_constant(float n0 = 1.0f, float base_charge = -1.0f, float base_mass = 1.0f);

    template <typename DensityFunc>
    void init_density_profile(const Grid& grid, DensityFunc&& density_fn, float base_charge = -1.0f, float base_mass = 1.0f);

    static constexpr size_t block_size();
    size_t                  num_blocks() const;
    size_t                  max_particles() const;
    size_t                  active_particles() const;

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
    void set_block_value(particle::ParticleBlock<BLOCK_SIZE>& block, ParticleValue value);

    ParticleSystem(const ParticleSystem&)            = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;
    ParticleSystem(ParticleSystem&&) noexcept;
    ParticleSystem& operator=(ParticleSystem&&) noexcept;

private:
    size_t maxParticles_;
    size_t numBlocks_;
    size_t activeParticles_ = 0;

    particle::ParticleBlock<BLOCK_SIZE>* blocks_ = nullptr;
};

} // namespace particle

#include "data/particle/src/particle_system.inl"

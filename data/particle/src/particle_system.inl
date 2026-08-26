#pragma once

#include <algorithm>
#include <cmath>
#include <new>
#include <random>
#include <span>
#include <tuple>

namespace particle
{

// ============================================================================
// Constructors & Destructor
// ============================================================================

template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>::ParticleSystem(size_t maxParticles, float charge, float mass)
        : maxParticles_(maxParticles), numBlocks_((maxParticles + BLOCK_SIZE - 1) / BLOCK_SIZE), base_charge_(charge), base_mass_(mass)
{
    if (numBlocks_ == 0)
    {
        blocks_          = nullptr;
        activeParticles_ = 0;
        return;
    }

    const size_t blocksBytes = numBlocks_ * sizeof(particle::ParticleBlock<BLOCK_SIZE>);
    blocks_                  = static_cast<particle::ParticleBlock<BLOCK_SIZE>*>(::operator new[](blocksBytes, std::align_val_t(CACHE_LINE)));
    std::uninitialized_value_construct_n(blocks_, numBlocks_);

    for (auto& block : std::span(blocks_, numBlocks_))
    {
        std::fill(block.weight.begin(), block.weight.end(), 1.0f);
    }
}

template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>::ParticleSystem(ParticleSystem&& other) noexcept
        : maxParticles_(other.maxParticles_), numBlocks_(other.numBlocks_), activeParticles_(other.activeParticles_), base_charge_(other.base_charge_),
          base_mass_(other.base_mass_), blocks_(other.blocks_)
{
    other.blocks_          = nullptr;
    other.maxParticles_    = 0;
    other.numBlocks_       = 0;
    other.activeParticles_ = 0;
}

template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>& ParticleSystem<BLOCK_SIZE>::operator=(ParticleSystem&& other) noexcept
{
    if (this != &other)
    {
        if (blocks_)
        {
            std::destroy_n(blocks_, numBlocks_);
            ::operator delete[](blocks_, std::align_val_t(CACHE_LINE));
        }

        maxParticles_    = other.maxParticles_;
        numBlocks_       = other.numBlocks_;
        activeParticles_ = other.activeParticles_;
        base_charge_     = other.base_charge_;
        base_mass_       = other.base_mass_;
        blocks_          = other.blocks_;

        other.blocks_          = nullptr;
        other.maxParticles_    = 0;
        other.numBlocks_       = 0;
        other.activeParticles_ = 0;
    }
    return *this;
}

template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>::~ParticleSystem()
{
    if (blocks_)
    {
        std::destroy_n(blocks_, numBlocks_);
        ::operator delete[](blocks_, std::align_val_t(CACHE_LINE));
        blocks_ = nullptr;
    }
}

// ============================================================================
// Accessors & System Information
// ============================================================================

template <size_t BLOCK_SIZE>
size_t ParticleSystem<BLOCK_SIZE>::active_particles() const
{
    return activeParticles_;
}

template <size_t BLOCK_SIZE>
constexpr size_t ParticleSystem<BLOCK_SIZE>::block_size()
{
    return BLOCK_SIZE;
}

template <size_t BLOCK_SIZE>
size_t ParticleSystem<BLOCK_SIZE>::max_particles() const
{
    return maxParticles_;
}

template <size_t BLOCK_SIZE>
size_t ParticleSystem<BLOCK_SIZE>::num_blocks() const
{
    return numBlocks_;
}

// ============================================================================
// Density Initializers
// ============================================================================

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_density_constant(float n0)
{
    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        std::fill(block.weight.begin(), block.weight.begin() + block.activeCount, n0);
    }
}

template <size_t BLOCK_SIZE>
template <typename DensityFunc>
void ParticleSystem<BLOCK_SIZE>::init_density_profile(const Grid&, DensityFunc&& density_fn)
{
    if (activeParticles_ == 0)
        return;

    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        for (size_t i = 0; i < block.activeCount; ++i)
        {
            block.weight[i] = static_cast<float>(density_fn(block.position_x[i]));
        }
    }
}

// ============================================================================
// Position Initializers
// ============================================================================

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_positions_uniform(double domain_length)
{
    if (activeParticles_ == 0)
        return;

    const double dx         = domain_length / static_cast<double>(activeParticles_);
    size_t       global_idx = 0;

    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        for (size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            block.position_x[i] = static_cast<float>((global_idx + 0.5) * dx);
        }
    }
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_positions_uniform(const Grid& grid)
{
    init_positions_uniform(grid.physical_size() * grid.cell_size());
}

// ============================================================================
// Velocity Initializers
// ============================================================================

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_velocities_cold(float vx, float vy, float vz)
{
    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        std::fill(block.momentum_x.begin(), block.momentum_x.end(), base_mass_ * vx);
        std::fill(block.momentum_y.begin(), block.momentum_y.end(), base_mass_ * vy);
        std::fill(block.momentum_z.begin(), block.momentum_z.end(), base_mass_ * vz);
    }
}

template <size_t BLOCK_SIZE>
template <typename VelFunc>
void ParticleSystem<BLOCK_SIZE>::init_velocities_profile(VelFunc&& vel_fn)
{
    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        for (size_t i = 0; i < block.activeCount; ++i)
        {
            const double x0   = block.position_x[i];
            auto [vx, vy, vz] = vel_fn(x0);

            block.momentum_x[i] = base_mass_ * static_cast<float>(vx);
            block.momentum_y[i] = base_mass_ * static_cast<float>(vy);
            block.momentum_z[i] = base_mass_ * static_cast<float>(vz);
        }
    }
}

// Anisotropic Thermal Distribution (handles custom drift lambda)
template <size_t BLOCK_SIZE>
template <typename DriftFunc>
void ParticleSystem<BLOCK_SIZE>::init_velocities_thermal(float v_th_x, float v_th_y, float v_th_z, DriftFunc&& drift_func, uint32_t seed)
{
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> u_dist(1e-7f, 1.0f - 1e-7f);

    auto sample_normal = [&](float v_th) -> float
    {
        const float u1 = u_dist(gen);
        const float u2 = u_dist(gen);
        return v_th * std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * std::numbers::pi_v<float> * u2);
    };

    init_velocities_profile(
            [&](double x)
            {
                const auto [vx_drift, vy_drift, vz_drift] = drift_func(x);
                return std::tuple{vx_drift + sample_normal(v_th_x), vy_drift + sample_normal(v_th_y), vz_drift + sample_normal(v_th_z)};
            });
}

// Isotropic Thermal Distribution with custom drift lambda (Delegates to anisotropic)
template <size_t BLOCK_SIZE>
template <typename DriftFunc>
void ParticleSystem<BLOCK_SIZE>::init_velocities_thermal(float v_th, DriftFunc&& drift_func, uint32_t seed)
{
    init_velocities_thermal(v_th, v_th, v_th, std::forward<DriftFunc>(drift_func), seed);
}

// Constant Anisotropic Thermal Drift Overload
template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_velocities_thermal(float v_th_x, float v_th_y, float v_th_z, float v_drift_x, float v_drift_y, float v_drift_z, uint32_t seed)
{
    init_velocities_thermal(v_th_x, v_th_y, v_th_z, [=](double) { return std::tuple{v_drift_x, v_drift_y, v_drift_z}; }, seed);
}

// Constant Isotropic Thermal Drift Overload
template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_velocities_thermal(float v_th, float v_drift_x, float v_drift_y, float v_drift_z, uint32_t seed)
{
    init_velocities_thermal(v_th, v_th, v_th, v_drift_x, v_drift_y, v_drift_z, seed);
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_velocities_wave(float v0, double k, bool longitudinal_only)
{
    init_velocities_profile(
            [v0, k, longitudinal_only](double x)
            {
                const float v_wave = v0 * std::sin(static_cast<float>(k * x));
                const float v_y    = longitudinal_only ? 0.0f : static_cast<float>(v0 * std::cos(k * x));
                return std::tuple{v_wave, v_y, 0.0f};
            });
}

// ============================================================================
// Mutators & State Adjustments
// ============================================================================

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::set_active(size_t n)
{
    activeParticles_ = std::min(n, maxParticles_);
    size_t remaining = activeParticles_;
    for (size_t b = 0; b < numBlocks_; ++b)
    {
        uint32_t cnt           = static_cast<uint32_t>(std::min(remaining, static_cast<size_t>(BLOCK_SIZE)));
        blocks_[b].activeCount = static_cast<uint16_t>(cnt);
        remaining              = (remaining > BLOCK_SIZE) ? (remaining - BLOCK_SIZE) : 0;
    }
}

} // namespace particle

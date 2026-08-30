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
void ParticleSystem<BLOCK_SIZE>::init_velocities_cold(float px, float py, float pz)
{
    const float p_sq      = px * px + py * py + pz * pz;
    const float inv_gamma = 1.0f / std::sqrt(1.0f + p_sq);

    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        std::fill(block.momentum_x.begin(), block.momentum_x.end(), px);
        std::fill(block.momentum_y.begin(), block.momentum_y.end(), py);
        std::fill(block.momentum_z.begin(), block.momentum_z.end(), pz);
        std::fill(block.inv_gamma.begin(), block.inv_gamma.end(), inv_gamma);
    }
}

template <size_t BLOCK_SIZE>
template <typename VelFunc>
void ParticleSystem<BLOCK_SIZE>::init_velocities_profile(VelFunc&& vel_fn)
{
    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto&        block = blocks_[b];
        const size_t count = block.activeCount;

        for (size_t i = 0; i < count; ++i)
        {
            const double x0   = block.position_x[i];
            auto [px, py, pz] = vel_fn(x0);

            const float fx = static_cast<float>(px);
            const float fy = static_cast<float>(py);
            const float fz = static_cast<float>(pz);

            block.momentum_x[i] = fx;
            block.momentum_y[i] = fy;
            block.momentum_z[i] = fz;

            const float p_sq   = fx * fx + fy * fy + fz * fz;
            block.inv_gamma[i] = 1.0f / std::sqrt(1.0f + p_sq);
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
// Relativistic Velocity Initializers (Maxwell-Jüttner & Lorentz Drift Boost)
// ============================================================================

// Isotropic Maxwell-Jüttner Thermal Sampler (Zenitani 2015 Rejection Algorithm)
template <size_t BLOCK_SIZE>
template <typename DriftFunc>
void ParticleSystem<BLOCK_SIZE>::init_velocities_rel_thermal(float theta, DriftFunc&& drift_func, uint32_t seed)
{
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> u_dist(1e-7f, 1.0f - 1e-7f);

    // Samples rest-frame specific momentum u' = gamma' * v' from Maxwell-Jüttner distribution
    auto sample_maxwell_juttner = [&](float th) -> std::tuple<float, float, float, float>
    {
        float x1, u_mag;
        while (true)
        {
            const float u1 = u_dist(gen), u2 = u_dist(gen), u3 = u_dist(gen);
            const float u4 = u_dist(gen), u5 = u_dist(gen), u6 = u_dist(gen), u7 = u_dist(gen);

            x1             = -th * std::log(u1 * u2 * u3);
            const float x2 = -th * std::log(u4 * u5 * u6 * u7);

            u_mag = std::sqrt(x1 * (x1 + 2.0f));
            if (u_dist(gen) * (x1 + x2) <= u_mag)
            {
                break; // Accepted
            }
        }

        const float gamma_prime = 1.0f + x1;
        const float cos_theta   = 2.0f * u_dist(gen) - 1.0f;
        const float sin_theta   = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
        const float phi         = 2.0f * std::numbers::pi_v<float> * u_dist(gen);

        return {u_mag * sin_theta * std::cos(phi), u_mag * sin_theta * std::sin(phi), u_mag * cos_theta, gamma_prime};
    };

    init_velocities_profile(
            [&](double x)
            {
                // Retrieve drift specific momentum u_d = gamma_d * v_d
                const auto [udx, udy, udz] = drift_func(x);
                const float ud_sq          = udx * udx + udy * udy + udz * udz;

                // Rest-frame thermal momentum u' and gamma'
                const auto [upx, upy, upz, gamma_prime] = sample_maxwell_juttner(theta);

                if (ud_sq == 0.0f)
                {
                    return std::tuple{upx, upy, upz};
                }

                // Relativistic 4-velocity Lorentz boost: u_lab = u' + u_d * ( (u_d . u')/(gamma_d + 1) + gamma' )
                const float gamma_d   = std::sqrt(1.0f + ud_sq);
                const float u_dot_up  = udx * upx + udy * upy + udz * upz;
                const float boost_fac = (u_dot_up / (gamma_d + 1.0f)) + gamma_prime;

                return std::tuple{upx + udx * boost_fac, upy + udy * boost_fac, upz + udz * boost_fac};
            });
}

// Constant Relativistic Drift Overload (u_drift = gamma_drift * v_drift)
template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_velocities_rel_thermal(float theta, float u_drift_x, float u_drift_y, float u_drift_z, uint32_t seed)
{
    init_velocities_rel_thermal(theta, [=](double) { return std::tuple{u_drift_x, u_drift_y, u_drift_z}; }, seed);
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

#include "data/particle/include/particle_system.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <new>
#include <random>
#include <span>
#include <stdexcept>

namespace particle
{

template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>::ParticleSystem(size_t maxParticles) : maxParticles_(maxParticles), numBlocks_((maxParticles + BLOCK_SIZE - 1) / BLOCK_SIZE)
{
    if (numBlocks_ == 0)
        throw std::bad_alloc();

    const size_t blocksBytes = numBlocks_ * sizeof(particle::ParticleBlock<BLOCK_SIZE>);

    // aligned allocation for blocks_ (raw memory)
    blocks_ = static_cast<particle::ParticleBlock<BLOCK_SIZE>*>(::operator new[](blocksBytes, std::align_val_t(CACHE_LINE)));

    std::uninitialized_value_construct_n(blocks_, numBlocks_);

    // debug/info
    std::cout << "Allocated " << numBlocks_ << " blocks (" << blocksBytes << " bytes).\n";

    for (auto& block : std::span(blocks_, numBlocks_))
    {
        std::fill(block.mass.begin(), block.mass.end(), 1.0f);
        std::fill(block.charge.begin(), block.charge.end(), -1.0f);
        std::fill(block.weight.begin(), block.weight.end(), 1.0f);
    }
}

template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>::ParticleSystem(size_t maxParticles, float charge, float mass) : ParticleSystem(maxParticles)
{
    for (auto& block : std::span(blocks_, numBlocks_))
    {
        std::fill(block.mass.begin(), block.mass.end(), mass);
        std::fill(block.charge.begin(), block.charge.end(), charge);
    }
}

template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>::ParticleSystem(ParticleSystem&& other) noexcept
        : maxParticles_(other.maxParticles_), numBlocks_(other.numBlocks_), activeParticles_(other.activeParticles_), blocks_(other.blocks_)
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
        blocks_          = other.blocks_;

        other.blocks_          = nullptr;
        other.maxParticles_    = 0;
        other.numBlocks_       = 0;
        other.activeParticles_ = 0;
    }
    return *this;
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::set_block_value(particle::ParticleBlock<BLOCK_SIZE>& block, ParticleValue value)
{
    if (value == ParticleValue::mass)
    {
        std::fill(block.mass.begin(), block.mass.end(), mass);
    }
    else if (value == ParticleValue::charge)
    {
        std::fill(block.charge.begin(), block.charge.end(), charge);
    }
    else if (value == ParticleValue::weight)
    {
        std::fill(block.weight.begin(), block.weight.end(), weight);
    }
}

template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>::~ParticleSystem()
{
    if (blocks_)
    {
        // destroy objects first (calls destructors properly)
        std::destroy_n(blocks_, numBlocks_);
        // free aligned memory
        ::operator delete[](blocks_, std::align_val_t(CACHE_LINE));
        blocks_ = nullptr;
    }
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::set_active(size_t n)
{
    activeParticles_ = std::min(n, maxParticles_);
    size_t remaining = activeParticles_;
    for (size_t b = 0; b < numBlocks_; ++b)
    {
        uint32_t cnt           = static_cast<uint32_t>(std::min(remaining, (size_t) BLOCK_SIZE));
        blocks_[b].activeCount = static_cast<uint16_t>(cnt);
        remaining              = (remaining > BLOCK_SIZE) ? (remaining - BLOCK_SIZE) : 0;
    }
}

// ---- Particle State Initializers ----
template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_positions_uniform(double domain_length)
{
    if (activeParticles_ == 0)
        return;

    double dx         = domain_length / static_cast<double>(activeParticles_);
    size_t global_idx = 0;

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

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_velocities_cold(float vx, float vy, float vz)
{
    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        std::fill(block.momentum_x.begin(), block.momentum_x.end(), vx);
        std::fill(block.momentum_y.begin(), block.momentum_y.end(), vy);
        std::fill(block.momentum_z.begin(), block.momentum_z.end(), vz);
    }
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_velocities_thermal(float v_th, float v_drift_x, float v_drift_y, float v_drift_z, uint32_t seed)
{
    std::mt19937                    gen(seed);
    std::normal_distribution<float> dist(0.0f, v_th);

    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        for (size_t i = 0; i < block.activeCount; ++i)
        {
            block.momentum_x[i] = v_drift_x + dist(gen);
            block.momentum_y[i] = v_drift_y + dist(gen);
            block.momentum_z[i] = v_drift_z + dist(gen);
        }
    }
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::init_density_constant(float n0, float base_charge, float base_mass)
{
    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        for (size_t i = 0; i < block.activeCount; ++i)
        {
            block.charge[i] = base_charge * n0;
            block.mass[i]   = base_mass * n0;
        }
    }
}

template <size_t BLOCK_SIZE>
template <typename DensityFunc>
void ParticleSystem<BLOCK_SIZE>::init_density_profile(const Grid& grid, DensityFunc&& density_fn, float base_charge, float base_mass)
{
    size_t       global_idx = 0;
    const double dx_p       = grid.physical_size() / static_cast<double>(activeParticles_);

    for (size_t b = 0; b < numBlocks_; ++b)
    {
        auto& block = blocks_[b];
        for (size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            const double x0      = (static_cast<double>(global_idx) + 0.5) * dx_p;
            const float  local_n = static_cast<float>(density_fn(x0));

            block.charge[i] = base_charge * local_n;
            block.mass[i]   = base_mass * local_n;
        }
    }
}

template <size_t BLOCK_SIZE>
constexpr size_t ParticleSystem<BLOCK_SIZE>::block_size()
{
    return BLOCK_SIZE;
}

template <size_t BLOCK_SIZE>
size_t ParticleSystem<BLOCK_SIZE>::num_blocks() const
{
    return numBlocks_;
}

template <size_t BLOCK_SIZE>
size_t ParticleSystem<BLOCK_SIZE>::max_particles() const
{
    return maxParticles_;
}

template <size_t BLOCK_SIZE>
size_t ParticleSystem<BLOCK_SIZE>::active_particles() const
{
    return activeParticles_;
}

} // namespace particle
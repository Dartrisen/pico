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
    mass,
    charge,
    weight
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

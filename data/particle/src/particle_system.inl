#include <new>
#include <span>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "data/particle/include/particle_system.hpp"
namespace particle
{
    /**
     * @brief Construct ParticleSystem using default mass and charge values
     */
    template <size_t BLOCK_SIZE>
    ParticleSystem<BLOCK_SIZE>::ParticleSystem(size_t maxParticles)
        : maxParticles_(maxParticles),
          numBlocks_((maxParticles + BLOCK_SIZE - 1) / BLOCK_SIZE)
    {
        if (numBlocks_ == 0)
            throw std::bad_alloc();

        const size_t blocksBytes = numBlocks_ * sizeof(particle::ParticleBlock<BLOCK_SIZE>);

        // aligned allocation for blocks_ (raw memory)
        blocks_ = static_cast<particle::ParticleBlock<BLOCK_SIZE> *>(
            ::operator new[](blocksBytes, std::align_val_t(CACHE_LINE)));

        std::fill(blocks_, blocks_ + numBlocks_, particle::ParticleBlock<BLOCK_SIZE>{});

        // debug/info
        std::cout << "Allocated " << numBlocks_ << " blocks (" << blocksBytes << " bytes).\n";

        for (auto &block : std::span(blocks_, numBlocks_))
        {
            std::fill(block.mass.begin(), block.mass.end(), 1.0f);
            std::fill(block.charge.begin(), block.charge.end(), -1.0f);
            std::fill(block.weight.begin(), block.weight.end(), 1.0f);
        }
    }

    template <size_t BLOCK_SIZE>
    ParticleSystem<BLOCK_SIZE>::ParticleSystem(ParticleSystem &&other) noexcept
        : maxParticles_(other.maxParticles_),
          numBlocks_(other.numBlocks_),
          activeParticles_(other.activeParticles_),
          blocks_(other.blocks_)
    {
        other.blocks_ = nullptr;
        other.maxParticles_ = 0;
        other.numBlocks_ = 0;
        other.activeParticles_ = 0;
    }

    template <size_t BLOCK_SIZE>
    ParticleSystem<BLOCK_SIZE> &ParticleSystem<BLOCK_SIZE>::operator=(ParticleSystem &&other) noexcept
    {
        if (this != &other)
        {
            // free current memory
            if (blocks_)
                ::operator delete[](blocks_, std::align_val_t(CACHE_LINE));

            maxParticles_ = other.maxParticles_;
            numBlocks_ = other.numBlocks_;
            activeParticles_ = other.activeParticles_;
            blocks_ = other.blocks_;

            other.blocks_ = nullptr;
            other.maxParticles_ = 0;
            other.numBlocks_ = 0;
            other.activeParticles_ = 0;
        }
        return *this;
    }

    template <size_t BLOCK_SIZE>
    void ParticleSystem<BLOCK_SIZE>::set_block_value(particle::ParticleBlock<BLOCK_SIZE> &block, ParticleValue value)
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
            uint32_t cnt = static_cast<uint32_t>(std::min(remaining, (size_t)BLOCK_SIZE));
            blocks_[b].activeCount = static_cast<uint16_t>(cnt);
            remaining = (remaining > BLOCK_SIZE) ? (remaining - BLOCK_SIZE) : 0;
        }
    }

    template <size_t BLOCK_SIZE>
    constexpr size_t ParticleSystem<BLOCK_SIZE>::block_size() { return BLOCK_SIZE; }

    template <size_t BLOCK_SIZE>
    size_t ParticleSystem<BLOCK_SIZE>::num_blocks() const { return numBlocks_; }

    template <size_t BLOCK_SIZE>
    size_t ParticleSystem<BLOCK_SIZE>::max_particles() const { return maxParticles_; }

    template <size_t BLOCK_SIZE>
    size_t ParticleSystem<BLOCK_SIZE>::active_particles() const { return activeParticles_; }

    // template <size_t BLOCK_SIZE>
    // particle::ParticleBlock<BLOCK_SIZE> *ParticleSystem<BLOCK_SIZE>::Blocks() { return blocks_; }

} // namespace particle
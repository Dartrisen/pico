#pragma once

#include <algorithm>

#include "particle_block.hpp"

constexpr size_t CACHE_LINE = 64;

template <size_t BLOCK_SIZE>
class ParticleSystem {
public:
    ParticleSystem(size_t maxParticles)
      : maxParticles_(maxParticles),
        numBlocks_((maxParticles + BLOCK_SIZE - 1) / BLOCK_SIZE)
    {
        if (numBlocks_ == 0) throw std::bad_alloc();

        const size_t blocksBytes = numBlocks_ * sizeof(ParticleBlock<BLOCK_SIZE>);

        // aligned allocation for blocks_ (raw memory)
        blocks_ = static_cast<ParticleBlock<BLOCK_SIZE>*>(
            ::operator new[](blocksBytes, std::align_val_t(CACHE_LINE))
        );

        std::memset(blocks_, 0, blocksBytes);
    }

    ~ParticleSystem() {
        ::operator delete[](blocks_, std::align_val_t(CACHE_LINE));
        blocks_ = nullptr;
    }

    void set_active(size_t n) {
        activeParticles_ = std::min(n, maxParticles_);
        size_t remaining = activeParticles_;
        for (size_t b = 0; b < numBlocks_; ++b) {
            uint32_t cnt = static_cast<uint32_t>(std::min(remaining, (size_t)BLOCK_SIZE));
            blocks_[b].activeCount  = static_cast<uint16_t>(cnt);
            remaining = (remaining > BLOCK_SIZE) ? (remaining - BLOCK_SIZE) : 0;
        }
    }

    void update_positions(float dt) {
        size_t remaining = activeParticles_;
        for (size_t b = 0; b < numBlocks_ && remaining; ++b) {
            size_t cnt = std::min(remaining, (size_t)BLOCK_SIZE);
            auto &blk = blocks_[b];

            for (size_t i = 0; i < cnt; ++i) {
                const float w = blk.weight[i];
                blk.posX[i] += blk.momX[i] * dt * w;
            }

            remaining -= cnt;
        }
    }

    static constexpr size_t block_size() { return BLOCK_SIZE; }
    size_t num_blocks() const { return numBlocks_; }
    size_t max_particles() const { return maxParticles_; }
    size_t active_particles() const { return activeParticles_; }

    ParticleBlock<BLOCK_SIZE>* Blocks() { return blocks_; }

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;

private:
    float charge_ = -1.0f;
    float mass_ = 1.0f;

    size_t maxParticles_;
    size_t numBlocks_;
    size_t activeParticles_ = 0;

    ParticleBlock<BLOCK_SIZE>* blocks_ = nullptr;
};

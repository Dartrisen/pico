#include <new>
#include <span>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "particle/include/particle_system.hpp"
#include "operators.hpp"

/**
 * @brief Construct ParticleSystem using default mass and charge values
 */
template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>::ParticleSystem(size_t maxParticles)
  : maxParticles_(maxParticles),
    numBlocks_((maxParticles + BLOCK_SIZE - 1) / BLOCK_SIZE)
{
    if (numBlocks_ == 0) throw std::bad_alloc();

    const size_t blocksBytes = numBlocks_ * sizeof(particle::ParticleBlock<BLOCK_SIZE>);

    // aligned allocation for blocks_ (raw memory)
    blocks_ = static_cast<particle::ParticleBlock<BLOCK_SIZE>*>(
        ::operator new[](blocksBytes, std::align_val_t(CACHE_LINE))
    );

    // zero-initialize blocks memory (ParticleBlock is POD-like: only floats/ints)
    std::memset(blocks_, 0, blocksBytes);

    // debug/info
    std::cout << "Allocated " << numBlocks_ << " blocks (" << blocksBytes << " bytes).\n";

    for (auto& block : std::span(blocks_, numBlocks_)) {
        std::fill(block.mass.begin(), block.mass.end(), 1.0f);
        std::fill(block.charge.begin(), block.charge.end(), -1.0f);
        std::fill(block.weight.begin(), block.weight.end(), 1.0f);
    }
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::set_block_value(particle::ParticleBlock<BLOCK_SIZE>& block, ParticleValue value) {
    if (value == ParticleValue::mass) {
        std::fill(block.mass.begin(), block.mass.end(), mass);
    } else if (value == ParticleValue::charge) {
        std::fill(block.charge.begin(), block.charge.end(), charge);
    } else if (value == ParticleValue::weight) {
        std::fill(block.weight.begin(), block.weight.end(), weight);
    }
}

template <size_t BLOCK_SIZE>
ParticleSystem<BLOCK_SIZE>::~ParticleSystem() {
    if (blocks_) {
        ::operator delete[](blocks_, std::align_val_t(CACHE_LINE));
        blocks_ = nullptr;
    }
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::set_active(size_t n) {
    activeParticles_ = std::min(n, maxParticles_);
    size_t remaining = activeParticles_;
    for (size_t b = 0; b < numBlocks_; ++b) {
        uint32_t cnt = static_cast<uint32_t>(std::min(remaining, (size_t)BLOCK_SIZE));
        blocks_[b].activeCount  = static_cast<uint16_t>(cnt);
        remaining = (remaining > BLOCK_SIZE) ? (remaining - BLOCK_SIZE) : 0;
    }
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::update_positions(float dt) {
    size_t remaining = activeParticles_;
    for (size_t b = 0; b < numBlocks_ && remaining; ++b) {
        size_t cnt = std::min(remaining, (size_t)BLOCK_SIZE);
        auto &blk = blocks_[b];

        for (size_t i = 0; i < cnt; ++i) {
            const float w = blk.weight[i];
            blk.position_x[i] += blk.momentum_x[i] * dt * w;
        }

        remaining -= cnt;
    }
}

template <size_t BLOCK_SIZE>
void ParticleSystem<BLOCK_SIZE>::advance(EMFields<BLOCK_SIZE>& fields,
                                         FieldSystem<BLOCK_SIZE>& J,
                                         const Grid& grid,
                                         float dt)
{
    FieldGather<CIC, BLOCK_SIZE> gather;
    BorisPusher<BLOCK_SIZE> boris;
    CurrentDeposit<CIC, BLOCK_SIZE> deposit;

    for (auto& block : std::span(blocks_, numBlocks_)) {
        FieldScratch<BLOCK_SIZE> scratch;

        gather(block, fields, grid, scratch);
        boris(block, scratch, dt);
        deposit(block, J, grid, dt);
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

template <size_t BLOCK_SIZE>
particle::ParticleBlock<BLOCK_SIZE>* ParticleSystem<BLOCK_SIZE>::Blocks() { return blocks_; }

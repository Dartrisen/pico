#pragma once

#include <algorithm>
#include <iostream>

#include "particle_block.hpp"

constexpr size_t CACHE_LINE = 64;

template <size_t BLOCK_SIZE>
class ParticleSystem {
public:
    ParticleSystem(size_t maxParticles);
    ~ParticleSystem();

    void set_active(size_t n);

    void update_positions(float dt);

    static constexpr size_t block_size();
    size_t num_blocks() const;
    size_t max_particles() const;
    size_t active_particles() const;

    ParticleBlock<BLOCK_SIZE>* Blocks();

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

#include "particle/src/particle_system.inl"

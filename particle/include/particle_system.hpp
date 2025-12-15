#pragma once

#include <algorithm>
#include <iostream>

#include "particle_block.hpp"


constexpr size_t CACHE_LINE = 64;

enum ParticleValue {
    mass,
    charge,
    weight
};

template <size_t BLOCK_SIZE>
class ParticleSystem {
public:
    ParticleSystem(size_t maxParticles);
    explicit ParticleSystem(size_t maxParticles, float charge, float mass);
    ~ParticleSystem();

    void set_active(size_t n);

    void update_positions(float dt);

    static constexpr size_t block_size();
    size_t num_blocks() const;
    size_t max_particles() const;
    size_t active_particles() const;

    particle::ParticleBlock<BLOCK_SIZE>* Blocks();
    void set_block_value(particle::ParticleBlock<BLOCK_SIZE>& block, ParticleValue value);

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;

private:
    size_t maxParticles_;
    size_t numBlocks_;
    size_t activeParticles_ = 0;

    particle::ParticleBlock<BLOCK_SIZE>* blocks_ = nullptr;
};

#include "particle/src/particle_system.inl"

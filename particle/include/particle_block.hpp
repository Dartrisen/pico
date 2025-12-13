#pragma once
#include <cstddef>

template <size_t BLOCK_SIZE>
struct ParticleBlock {
    alignas(64) float position_x[BLOCK_SIZE];
    alignas(64) float momentum_x[BLOCK_SIZE];
    alignas(64) float weight[BLOCK_SIZE];
    alignas(64) float mass[BLOCK_SIZE];

    uint16_t activeCount = 0;
};

#pragma once
#include <cstddef>

template <size_t BLOCK_SIZE>
struct ParticleBlock {
    float posX[BLOCK_SIZE];
    float momX[BLOCK_SIZE];
    float weight[BLOCK_SIZE];

    uint16_t activeCount = 0;
};
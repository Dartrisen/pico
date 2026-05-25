#pragma once
#include "field_system.hpp"

template <size_t BLOCK_SIZE>
struct EMFields
{
    FieldSystem<BLOCK_SIZE> E;
    FieldSystem<BLOCK_SIZE> B;

    EMFields(const Grid& grid) : E(grid), B(grid) {}
};

template <size_t BLOCK_SIZE>
struct alignas(64) FieldScratch
{
    using ComponentArray = std::array<float, BLOCK_SIZE>;

    ComponentArray Ex{}, Ey{}, Ez{};
    ComponentArray Bx{}, By{}, Bz{};
};

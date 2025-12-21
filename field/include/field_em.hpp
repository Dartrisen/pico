#pragma once
#include "field_system.hpp"


template <size_t BLOCK_SIZE>
struct EMFields {
    FieldSystem<BLOCK_SIZE> E;
    FieldSystem<BLOCK_SIZE> B;

    EMFields(const Grid& grid)
        : E(grid), B(grid) {}
};

template <size_t BLOCK_SIZE>
struct FieldScratch {
    alignas(64) std::array<float, BLOCK_SIZE> Ex, Ey, Ez;
    alignas(64) std::array<float, BLOCK_SIZE> Bx, By, Bz;

    inline void clear() noexcept {
        Ex.fill(0.f); Ey.fill(0.f); Ez.fill(0.f);
        Bx.fill(0.f); By.fill(0.f); Bz.fill(0.f);
    }
};

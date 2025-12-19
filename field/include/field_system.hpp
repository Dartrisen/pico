#pragma once
#include <cstddef>
#include <cstdint>

#include "grid.hpp"
#include "field_block.hpp"

/**
 * @brief Cache-aligned, SIMD-friendly field block using Structure-of-Arrays
 * @tparam BLOCK_SIZE Number of fields per block (must be multiple of 8)
 */
template<size_t BLOCK_SIZE>
class FieldSystem {
public:
    FieldSystem(const Grid& grid);

    // Accessors
    field::FieldBlock<BLOCK_SIZE>& block(size_t b) noexcept;
    const field::FieldBlock<BLOCK_SIZE>& block(size_t b) const noexcept;

    // Global linear access (slow, but useful for debugging)
    float& field_x(size_t idx) {
        auto [b, i] = locate(idx);
        return blocks_[b].field_x[i];
    }

    // Fast block iteration
    template<typename Func>
    void for_each_block(Func&& f) {
        for (size_t b = 0; b < num_blocks_; ++b)
            f(blocks_[b], b);
    }

    size_t num_blocks() const noexcept { return num_blocks_; }
    const Grid& grid() const noexcept { return grid_; }

private:
    const Grid& grid_;
    size_t num_blocks_;
    std::vector<field::FieldBlock<BLOCK_SIZE>> blocks_;

    static constexpr std::pair<size_t,size_t> locate(size_t linear_idx) noexcept {
        return { linear_idx / BLOCK_SIZE, linear_idx % BLOCK_SIZE };
    }
};

#include "field/src/field_system.inl"

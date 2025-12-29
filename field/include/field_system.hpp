#pragma once
#include <cstddef>
#include <cstdint>
#include <cassert>

#include "grid.hpp"
#include "field_block.hpp"

/**
 * @brief Cache-aligned, SIMD-friendly field block using Structure-of-Arrays
 * @tparam BLOCK_SIZE Number of fields per block (must be multiple of 8)
 */
template<size_t BLOCK_SIZE>
class FieldSystem {
public:
    explicit FieldSystem(const Grid& grid);

    // ---- block access (FAST PATH) ----
    field::FieldBlock<BLOCK_SIZE>& block(size_t b) noexcept;
    const field::FieldBlock<BLOCK_SIZE>& block(size_t b) const noexcept;

    size_t num_blocks() const noexcept { return blocks_.size(); }
    const Grid& grid() const noexcept { return grid_; }

    // ---- scalar access (SLOW PATH) ----
    float& field(field::FieldComp c, size_t idx) noexcept {
        assert(idx < blocks_.size() * BLOCK_SIZE);
        const size_t b = idx / BLOCK_SIZE;
        const size_t i = idx % BLOCK_SIZE;
        return blocks_[b].component(c)[i];
    }

    const float& field(field::FieldComp c, size_t idx) const noexcept {
        assert(idx < blocks_.size() * BLOCK_SIZE);
        const size_t b = idx / BLOCK_SIZE;
        const size_t i = idx % BLOCK_SIZE;
        return blocks_[b].component(c)[i];
    }

    // Backward compatibility helpers
    float& field_x(size_t idx) noexcept {
        return field(field::FieldComp::X, idx);
    }

    const float& field_x(size_t idx) const noexcept {
        return field(field::FieldComp::X, idx);
    }

    float& operator[](size_t idx) noexcept {
        return field_x(idx);
    }

    const float& operator[](size_t idx) const noexcept {
        return field_x(idx);
    }

    // ---- block iteration ----
    template<typename Func>
    void for_each_block(Func&& f) noexcept {
        for (size_t b = 0; b < blocks_.size(); ++b)
            f(blocks_[b], b);
    }

    template<typename Func>
    void for_each_block(Func&& f) const noexcept {
        for (size_t b = 0; b < blocks_.size(); ++b)
            f(blocks_[b], b);
    }

    // ---- bulk operations ----
    void set_fields(field::FieldComp c, float value) noexcept {
        for_each_block([&](auto& blk, size_t) {
            std::fill(blk.component(c).begin(),
                      blk.component(c).end(),
                      value);
        });
    }

    // backward-compatible helper
    void set_field_x(float value) noexcept {
        set_fields(field::FieldComp::X, value);
    }

private:
    Grid grid_;
    std::vector<field::FieldBlock<BLOCK_SIZE>> blocks_;
};

#include "field/src/field_system.inl"

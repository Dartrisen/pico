#pragma once
#include "field_block.hpp"
#include "grid.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief Cache-aligned, SIMD-friendly field block using Structure-of-Arrays
 * @tparam BLOCK_SIZE Number of fields per block (must be multiple of 8)
 */
template <size_t BLOCK_SIZE>
class FieldSystem
{
public:
    explicit FieldSystem(const Grid& grid);

    // ---- block access (FAST PATH) ----
    field::FieldBlock<BLOCK_SIZE>&       block(size_t b) noexcept;
    const field::FieldBlock<BLOCK_SIZE>& block(size_t b) const noexcept;

    size_t num_blocks() const noexcept
    {
        return blocks_.size();
    }
    const Grid& grid() const noexcept
    {
        return grid_;
    }

    template <field::FieldComp C>
    float& field(size_t idx) noexcept
    {
        assert(idx < blocks_.size() * BLOCK_SIZE);
        const size_t b = idx / BLOCK_SIZE;
        const size_t i = idx % BLOCK_SIZE;
        return blocks_[b].template component<C>()[i];
    }

    template <field::FieldComp C>
    const float& field(size_t idx) const noexcept
    {
        assert(idx < blocks_.size() * BLOCK_SIZE);
        const size_t b = idx / BLOCK_SIZE;
        const size_t i = idx % BLOCK_SIZE;
        return blocks_[b].template component<C>()[i];
    }

    float& field_x(size_t idx) noexcept
    {
        return field<field::FieldComp::X>(idx);
    }

    const float& field_x(size_t idx) const noexcept
    {
        return field<field::FieldComp::X>(idx);
    }

    float& field_y(size_t idx) noexcept
    {
        return field<field::FieldComp::Y>(idx);
    }

    const float& field_y(size_t idx) const noexcept
    {
        return field<field::FieldComp::Y>(idx);
    }

    float& field_z(size_t idx) noexcept
    {
        return field<field::FieldComp::Z>(idx);
    }

    const float& field_z(size_t idx) const noexcept
    {
        return field<field::FieldComp::Z>(idx);
    }

    float& operator[](size_t idx) noexcept
    {
        return field_x(idx);
    }

    const float& operator[](size_t idx) const noexcept
    {
        return field_x(idx);
    }

    // ---- zero out system ----
    void zero_out() noexcept
    {
        set_fields<field::FieldComp::X>(0.0f);
        set_fields<field::FieldComp::Y>(0.0f);
        set_fields<field::FieldComp::Z>(0.0f);
    }

    // ---- block iteration ----
    template <typename Func>
    [[gnu::always_inline]] void for_each_block(Func&& f) noexcept
    {
        for (size_t b = 0; b < blocks_.size(); ++b)
            f(blocks_[b], b);
    }

    template <typename Func>
    [[gnu::always_inline]] void for_each_block(Func&& f) const noexcept
    {
        for (size_t b = 0; b < blocks_.size(); ++b)
            f(blocks_[b], b);
    }

    template <field::FieldComp C>
    void set_fields(float value) noexcept
    {
        for_each_block([&](auto& blk, size_t) { std::fill(blk.template component<C>().begin(), blk.template component<C>().end(), value); });
    }

    // backward-compatible helper
    void set_field_x(float value) noexcept
    {
        set_fields<field::FieldComp::X>(value);
    }

private:
    Grid                                       grid_;
    std::vector<field::FieldBlock<BLOCK_SIZE>> blocks_;
};

#include "data/field/src/field_system.inl"

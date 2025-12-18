#pragma once
#include <cstddef>
#include <cstdint>

namespace field {

/**
 * @brief Cache-aligned, SIMD-friendly field block using Structure-of-Arrays
 * @tparam BLOCK_SIZE Number of fields per block (must be multiple of 8)
 */
template <size_t BLOCK_SIZE>
struct FieldBlock {
    static_assert(BLOCK_SIZE % 8 == 0, "BLOCK_SIZE must be multiple of 8 for AVX");

    static constexpr size_t size_x = BLOCK_SIZE;

    // Field components
    alignas(64) std::array<float, size_x> field_x;
    alignas(64) std::array<float, size_x> field_y;
    alignas(64) std::array<float, size_x> field_z;

};

} // namespace field
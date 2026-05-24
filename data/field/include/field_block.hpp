#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace field
{

    /**
     * @brief Enum for field components
     */
    enum class FieldComp : uint8_t
    {
        X = 0,
        Y = 1,
        Z = 2
    };

    /**
     * @brief Cache-aligned, SIMD-friendly field block using Structure-of-Arrays
     * @tparam BLOCK_SIZE Number of fields per block (must be multiple of 8)
     */
    template <size_t BLOCK_SIZE>
    struct FieldBlock
    {
        std::array<float, BLOCK_SIZE> field_x;
        std::array<float, BLOCK_SIZE> field_y;
        std::array<float, BLOCK_SIZE> field_z;

        auto& component(FieldComp c) noexcept
        {
            switch (c)
            {
                case FieldComp::X:
                    return field_x;
                case FieldComp::Y:
                    return field_y;
                case FieldComp::Z:
                    return field_z;
            }
            __builtin_unreachable();
        }

        const auto& component(FieldComp c) const noexcept
        {
            switch (c)
            {
                case FieldComp::X:
                    return field_x;
                case FieldComp::Y:
                    return field_y;
                case FieldComp::Z:
                    return field_z;
            }
            __builtin_unreachable();
        }
    };

} // namespace field
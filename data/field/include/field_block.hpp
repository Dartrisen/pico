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
    struct alignas(64) FieldBlock
    {
        using ComponentArray = std::array<float, BLOCK_SIZE>;

        ComponentArray field_x;
        ComponentArray field_y;
        ComponentArray field_z;

        template <FieldComp C>
        ComponentArray& component() noexcept
        {
            if constexpr (C == FieldComp::X)
                return field_x;
            else if constexpr (C == FieldComp::Y)
                return field_y;
            else
                return field_z;
        }

        template <FieldComp C>
        const ComponentArray& component() const noexcept
        {
            if constexpr (C == FieldComp::X)
                return field_x;
            else if constexpr (C == FieldComp::Y)
                return field_y;
            else
                return field_z;
        }
    };

} // namespace field
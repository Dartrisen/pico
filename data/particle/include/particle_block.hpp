#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace particle
{

    enum class MomentumComp : uint8_t
    {
        X = 0,
        Y = 1,
        Z = 2
    };

    /**
     * @brief Cache-aligned, SIMD-friendly particle block using Structure-of-Arrays
     * @tparam BLOCK_SIZE Number of particles per block (must be multiple of 16 for AVX-512)
     */
    template <size_t BLOCK_SIZE>
    struct alignas(64) ParticleBlock
    {
        static constexpr size_t size_x = BLOCK_SIZE;

        // Particle positions
        std::array<float, size_x> position_x;

        // Particle momenta
        std::array<float, size_x> momentum_x;
        std::array<float, size_x> momentum_y;
        std::array<float, size_x> momentum_z;

        // Particle properties
        std::array<float, size_x> weight;
        std::array<float, size_x> mass;
        std::array<float, size_x> charge;

        uint32_t activeCount = 0;
        uint32_t blockId     = 0;

        /**
         * @brief Check if block is full
         */
        [[nodiscard]] bool isFull() const
        {
            return activeCount >= BLOCK_SIZE;
        }

        /**
         * @brief Check if block is empty
         */
        [[nodiscard]] bool isEmpty() const
        {
            return activeCount == 0;
        }

        /**
         * @brief Get available space in block
         */
        [[nodiscard]] size_t availableSpace() const
        {
            return BLOCK_SIZE - activeCount;
        }

        /**
         * @brief Remove particle (unsafely) at index by swapping with last active particle
         */
        void removeParticle(size_t index) noexcept
        {
            size_t last = --activeCount;

            position_x[index] = position_x[last];
            momentum_x[index] = momentum_x[last];
            momentum_y[index] = momentum_y[last];
            momentum_z[index] = momentum_z[last];
            weight[index]     = weight[last];
            mass[index]       = mass[last];
            charge[index]     = charge[last];
        }

        template <MomentumComp C>
        auto& component() noexcept
        {
            if constexpr (C == MomentumComp::X)
                return momentum_x;
            else if constexpr (C == MomentumComp::Y)
                return momentum_y;
            else
                return momentum_z;
        }

        template <MomentumComp C>
        const auto& component() const noexcept
        {
            if constexpr (C == MomentumComp::X)
                return momentum_x;
            else if constexpr (C == MomentumComp::Y)
                return momentum_y;
            else
                return momentum_z;
        }
    };

} // namespace particle

#pragma once
#include <cstddef>
#include <cstdint>
#include <array>


namespace particle {

enum class MomentumComp : uint8_t {
    X = 0, Y = 1, Z = 2
};

/**
 * @brief Cache-aligned, SIMD-friendly particle block using Structure-of-Arrays
 * @tparam BLOCK_SIZE Number of particles per block (must be multiple of 8)
 */
template <size_t BLOCK_SIZE>
struct ParticleBlock {
    static_assert(BLOCK_SIZE % 8 == 0, "BLOCK_SIZE must be multiple of 8 for AVX");

    static constexpr size_t size_x = BLOCK_SIZE;

    // Particle positions
    alignas(64) std::array<float, size_x> position_x;

    // Particle momenta
    alignas(64) std::array<float, size_x> momentum_x;
    alignas(64) std::array<float, size_x> momentum_y;
    alignas(64) std::array<float, size_x> momentum_z;

    // Particle properties
    alignas(64) std::array<float, size_x> weight;
    alignas(64) std::array<float, size_x> mass;
    alignas(64) std::array<float, size_x> charge;

    uint16_t activeCount = 0;
    uint32_t blockId = 0;

    /**
     * @brief Check if block is full
     */
    bool isFull() const { return activeCount >= BLOCK_SIZE; }

    /**
     * @brief Check if block is empty
     */
    bool isEmpty() const { return activeCount == 0; }

    /**
     * @brief Get available space in block
     */
    size_t availableSpace() const { return BLOCK_SIZE - activeCount; }

    /**
     * @brief Remove particle (unsafely) at index by swapping with last active particle
     */
    void removeParticle(size_t index) noexcept {
        size_t last = --activeCount;

        position_x[index] = position_x[last];
        momentum_x[index] = momentum_x[last];
        weight[index]     = weight[last];
        mass[index]       = mass[last];
        charge[index]     = charge[last];
    }

    auto& component(MomentumComp c) noexcept {
        switch (c) {
            case MomentumComp::X: return momentum_x;
            case MomentumComp::Y: return momentum_y;
            case MomentumComp::Z: return momentum_z;
        }
        __builtin_unreachable();
    }

    const auto& component(MomentumComp c) const noexcept {
        switch (c) {
            case MomentumComp::X: return momentum_x;
            case MomentumComp::Y: return momentum_y;
            case MomentumComp::Z: return momentum_z;
        }
        __builtin_unreachable();
    }
};

} // namespace particle

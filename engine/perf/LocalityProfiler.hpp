#pragma once

#include "data/grid/include/grid.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace pico::diagnostics
{

struct LocalityMetrics
{
    std::atomic<std::uint64_t> total_stride{0};
    std::atomic<std::uint64_t> total_pairs{0};

    LocalityMetrics() noexcept = default;

    // Custom Move Constructor to enable move semantics on std::atomic members
    LocalityMetrics(LocalityMetrics&& other) noexcept
            : total_stride(other.total_stride.load(std::memory_order_relaxed)), total_pairs(other.total_pairs.load(std::memory_order_relaxed))
    {
    }

    // Custom Move Assignment
    LocalityMetrics& operator=(LocalityMetrics&& other) noexcept
    {
        if (this != &other)
        {
            total_stride.store(other.total_stride.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_pairs.store(other.total_pairs.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    // Disable copy constructors
    LocalityMetrics(const LocalityMetrics&)            = delete;
    LocalityMetrics& operator=(const LocalityMetrics&) = delete;

    void reset() noexcept
    {
        total_stride.store(0, std::memory_order_relaxed);
        total_pairs.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] double mean_stride() const noexcept
    {
        const std::uint64_t pairs = total_pairs.load(std::memory_order_relaxed);
        if (pairs == 0)
            return 0.0;
        return static_cast<double>(total_stride.load(std::memory_order_relaxed)) / static_cast<double>(pairs);
    }
};

template <typename ParticleBlock>
class LocalityProfiler
{
public:
    static void process_block(const ParticleBlock& block, const Grid& grid, LocalityMetrics& metrics) noexcept
    {
        const std::size_t count = block.activeCount;
        if (count <= 1)
            return;

        const double  inv_dx       = 1.0 / grid.cell_size();
        std::uint64_t local_stride = 0;
        std::size_t   prev_cell    = static_cast<std::size_t>(block.position_x[0] * inv_dx);

        for (std::size_t i = 1; i < count; ++i)
        {
            const std::size_t curr_cell = static_cast<std::size_t>(block.position_x[i] * inv_dx);
            local_stride += (curr_cell >= prev_cell) ? (curr_cell - prev_cell) : (prev_cell - curr_cell);
            prev_cell = curr_cell;
        }

        metrics.total_stride.fetch_add(local_stride, std::memory_order_relaxed);
        metrics.total_pairs.fetch_add(count - 1, std::memory_order_relaxed);
    }

    template <typename ParticleContainer>
    static double compute_mean_cell_stride(const ParticleContainer& particles, const Grid& grid) noexcept
    {
        LocalityMetrics metrics;

        for (const auto& block : particles)
        {
            process_block(block, grid, metrics);
        }

        return metrics.mean_stride();
    }
};

} // namespace pico::diagnostics

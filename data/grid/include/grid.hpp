#pragma once

#include <cstddef>

class Grid
{
private:
    std::size_t num_cells_;   // Physical cell count N
    double      cell_size_;   // dx
    std::size_t guard_cells_; // G (e.g., 2)

public:
    Grid(std::size_t num_cells, double cell_size, std::size_t guard_cells = 2)
            : num_cells_(num_cells), cell_size_(cell_size), guard_cells_(guard_cells)
    {
    }

    [[nodiscard]] std::size_t physical_size() const noexcept
    {
        return num_cells_;
    }
    [[nodiscard]] std::size_t total_size() const noexcept
    {
        return num_cells_ + 2 * guard_cells_;
    }
    [[nodiscard]] std::size_t guard_cells() const noexcept
    {
        return guard_cells_;
    }
    [[nodiscard]] double cell_size() const noexcept
    {
        return cell_size_;
    }

    // Map physical index [0, N-1] to array storage index [G, N+G-1]
    [[nodiscard]] std::size_t physical_to_buffer(std::size_t i) const noexcept
    {
        return i + guard_cells_;
    }
};

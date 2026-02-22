#pragma once
#include <stdexcept>
#include <iostream>

/**
 * @brief Represents a one-dimensional computational grid
 */
struct Grid
{
public:
    // Constructor
    Grid(int nx_, double dx_) : nx(nx_), dx(dx_)
    {
        if (nx <= 0)
            throw std::invalid_argument("nx must be positive");
        if (dx <= 0.0)
            throw std::invalid_argument("dx must be positive");
    }

    // Returns the index if valid, otherwise throws
    int idx(int i) const
    {
        if (i < 0 || i >= nx)
        {
            throw std::out_of_range("Grid index out of bounds");
        }
        return i;
    }

    // Grid size
    size_t size() const noexcept
    {
        return nx;
    }

    // Cell size
    double cell_size() const noexcept
    {
        return dx;
    }

    // Get coordinate of cell center
    double x(int i) const
    {
        return idx(i) * dx + 0.5 * dx;
    }

private:
    int nx;
    double dx;
};

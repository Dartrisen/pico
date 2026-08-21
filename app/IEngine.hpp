#pragma once
#include <cstddef>

struct IEngine
{
    virtual ~IEngine()                            = default;
    virtual void        advance(double dt)        = 0;
    virtual void        print_perf_report() const = 0;
    virtual std::size_t total_particles() const   = 0;
    virtual std::size_t grid_cells() const        = 0;
};

#pragma once

#include "engine/perf/PipelineProfiler.hpp"

#include <cstddef>

struct IEngine
{
    virtual ~IEngine()                                                            = default;
    virtual void                                advance(double dt)                = 0;
    virtual std::size_t                         total_particles() const           = 0;
    virtual std::size_t                         grid_cells() const                = 0;
    virtual const pico::perf::PipelineProfiler& profiler() const                  = 0;
    virtual double                              mean_cell_stride() const noexcept = 0;
    virtual double                              injected_energy() const           = 0;
};

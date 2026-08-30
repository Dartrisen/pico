#pragma once

#include "IEngine.hpp"

#include <utility>

template <class EngineT>
class EngineWrapper final : public IEngine
{
public:
    explicit EngineWrapper(EngineT&& engine) : engine_(std::move(engine)) {}

    void advance(double dt) override
    {
        engine_.advance(dt);
    }

    std::size_t total_particles() const override
    {
        return engine_.total_particles();
    }

    std::size_t grid_cells() const override
    {
        return engine_.fields().E.grid().physical_size();
    }

    double injected_energy() const override
    {
        if constexpr (requires { engine_.injector().injected_energy(); })
        {
            return engine_.injector().injected_energy();
        }
        else
        {
            return 0.0;
        }
    }

    double mean_cell_stride() const noexcept override
    {
        return engine_.mean_cell_stride();
    }

    const pico::perf::PipelineProfiler& profiler() const override
    {
        return engine_.profiler();
    }

    EngineT& engine() noexcept
    {
        return engine_;
    }
    const EngineT& engine() const noexcept
    {
        return engine_;
    }

private:
    EngineT engine_;
};

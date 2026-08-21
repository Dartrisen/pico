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
        return engine_.particles().active_particles();
    }

    std::size_t grid_cells() const override
    {
        return engine_.fields().E.grid().physical_size();
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

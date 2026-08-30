#pragma once

#include "CycleClock.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace pico::perf
{
enum class Stage : std::size_t
{
    Gather = 0,
    Pusher,
    Deposit,
    FieldSolver,
    Boundaries,
    Sorting,
    Count
};

constexpr std::array<std::string_view, static_cast<std::size_t>(Stage::Count)> STAGE_NAMES = {
        "Field Gather", "Boris Pusher & Particle Boundary", "Current Deposit", "Field Solver (Yee Maxwell)", "Boundary Handlers (Field/Current)", "Particle Sorting"};

class PipelineProfiler
{
public:
    PipelineProfiler() = default;

    PipelineProfiler(PipelineProfiler&& other) noexcept
    {
        assign_from(other);
    }

    PipelineProfiler& operator=(PipelineProfiler&& other) noexcept
    {
        if (this != &other)
        {
            assign_from(other);
        }
        return *this;
    }

    PipelineProfiler(const PipelineProfiler&)            = delete;
    PipelineProfiler& operator=(const PipelineProfiler&) = delete;

    struct ScopedTimer
    {
        PipelineProfiler& profiler;
        Stage             stage;
        std::uint64_t     start_ticks;

        ScopedTimer(PipelineProfiler& p, Stage s) : profiler(p), stage(s), start_ticks(read_cpu_ticks()) {}

        ~ScopedTimer()
        {
            profiler.add_ticks(stage, read_cpu_ticks() - start_ticks);
        }
    };

    [[nodiscard]] ScopedTimer time_stage(Stage stage) noexcept
    {
        return ScopedTimer(*this, stage);
    }

    void add_nanoseconds(Stage stage, std::uint64_t ns) noexcept
    {
        accumulated_ns_[static_cast<std::size_t>(stage)].fetch_add(ns, std::memory_order_relaxed);
    }

    void add_ticks(Stage stage, std::uint64_t ticks) noexcept
    {
        const auto ns = static_cast<std::uint64_t>(ticks * ticks_to_nanoseconds_scale());
        add_nanoseconds(stage, ns);
    }

    void reset() noexcept
    {
        for (auto& ns : accumulated_ns_)
        {
            ns.store(0, std::memory_order_relaxed);
        }
    }

    // Granular Unit Getters
    std::uint64_t nanoseconds(Stage stage) const noexcept
    {
        return accumulated_ns_[static_cast<std::size_t>(stage)].load(std::memory_order_relaxed);
    }

    double microseconds(Stage stage) const noexcept
    {
        return nanoseconds(stage) * 1e-3;
    }
    double milliseconds(Stage stage) const noexcept
    {
        return nanoseconds(stage) * 1e-6;
    }
    double seconds(Stage stage) const noexcept
    {
        return nanoseconds(stage) * 1e-9;
    }

    double total_seconds() const noexcept
    {
        double sum = 0.0;
        for (const auto& ns : accumulated_ns_)
        {
            sum += ns.load(std::memory_order_relaxed) * 1e-9;
        }
        return sum;
    }

    static std::string format_ns(double ns)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);

        if (ns >= 1e9)
            ss << (ns * 1e-9) << " s";
        else if (ns >= 1e6)
            ss << (ns * 1e-6) << " ms";
        else if (ns >= 1000.0)
            ss << (ns * 1e-3) << " us";
        else
            ss << ns << " ns";

        return ss.str();
    }

    static std::string format_time(double seconds)
    {
        return format_ns(seconds * 1e9);
    }

private:
    void assign_from(const PipelineProfiler& other) noexcept
    {
        for (std::size_t i = 0; i < static_cast<std::size_t>(Stage::Count); ++i)
        {
            accumulated_ns_[i].store(other.accumulated_ns_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
    }

    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Stage::Count)> accumulated_ns_{};
};

} // namespace pico::perf
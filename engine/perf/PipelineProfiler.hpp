#pragma once

#include <array>
#include <atomic>
#include <chrono>
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
    using clock = std::chrono::high_resolution_clock;

    PipelineProfiler() = default;

    PipelineProfiler(PipelineProfiler&& other) noexcept
    {
        for (std::size_t i = 0; i < static_cast<std::size_t>(Stage::Count); ++i)
        {
            accumulated_ns_[i].store(other.accumulated_ns_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
    }

    PipelineProfiler& operator=(PipelineProfiler&& other) noexcept
    {
        if (this != &other)
        {
            for (std::size_t i = 0; i < static_cast<std::size_t>(Stage::Count); ++i)
            {
                accumulated_ns_[i].store(other.accumulated_ns_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }
        return *this;
    }

    PipelineProfiler(const PipelineProfiler&)            = delete;
    PipelineProfiler& operator=(const PipelineProfiler&) = delete;

    struct ScopedTimer
    {
        PipelineProfiler& profiler;
        Stage             stage;
        clock::time_point start;

        ScopedTimer(PipelineProfiler& p, Stage s) : profiler(p), stage(s), start(clock::now()) {}

        ~ScopedTimer()
        {
            const auto duration = clock::now() - start;
            const auto ns       = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
            profiler.add_nanoseconds(stage, static_cast<std::uint64_t>(ns));
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

    // Auto-scaling dynamic duration formatter
    static std::string format_time(double seconds)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);

        if (seconds >= 1.0)
        {
            ss << seconds << " s";
        }
        else if (seconds >= 1e-3)
        {
            ss << (seconds * 1e3) << " ms";
        }
        else if (seconds >= 1e-6)
        {
            ss << (seconds * 1e6) << " us";
        }
        else
        {
            ss << (seconds * 1e9) << " ns";
        }

        return ss.str();
    }

private:
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Stage::Count)> accumulated_ns_{};
};

} // namespace pico::perf
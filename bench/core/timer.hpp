#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace bench
{

    /**
     * @brief High-resolution timer with explicit nanosecond precision
     */
    class Timer
    {
    public:
        using Clock = std::chrono::high_resolution_clock;

        Timer() : start_(Clock::now()) {}

        void reset()
        {
            start_ = Clock::now();
        }

        // Return time in various units
        double elapsed_ns() const
        {
            auto end = Clock::now();
            return std::chrono::duration<double, std::nano>(end - start_).count();
        }

        double elapsed_us() const
        {
            auto end = Clock::now();
            return std::chrono::duration<double, std::micro>(end - start_).count();
        }

        double elapsed_ms() const
        {
            auto end = Clock::now();
            return std::chrono::duration<double, std::milli>(end - start_).count();
        }

        double elapsed_sec() const
        {
            auto end = Clock::now();
            return std::chrono::duration<double>(end - start_).count();
        }

        uint64_t elapsed_ns_int() const
        {
            auto end = Clock::now();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        }

    private:
        Clock::time_point start_;
    };

    /**
     * @brief Stopwatch for multiple measurements
     */
    class Stopwatch
    {
    public:
        Stopwatch() = default;

        void start()
        {
            timer_.reset();
        }

        void stop()
        {
            measurements_.push_back(timer_.elapsed_ns());
        }

        size_t count() const
        {
            return measurements_.size();
        }

        double min_ns() const;
        double max_ns() const;
        double mean_ns() const;
        double median_ns() const;
        double stddev_ns() const;

        void print_stats(const char* label) const;

    private:
        Timer               timer_;
        std::vector<double> measurements_;
    };

    // ========== inline implementations ==========

    inline double Stopwatch::min_ns() const
    {
        if (measurements_.empty())
            return 0.0;
        double m = measurements_[0];
        for (auto x : measurements_)
            m = (x < m) ? x : m;
        return m;
    }

    inline double Stopwatch::max_ns() const
    {
        if (measurements_.empty())
            return 0.0;
        double m = measurements_[0];
        for (auto x : measurements_)
            m = (x > m) ? x : m;
        return m;
    }

    inline double Stopwatch::mean_ns() const
    {
        if (measurements_.empty())
            return 0.0;
        double sum = 0.0;
        for (auto x : measurements_)
            sum += x;
        return sum / measurements_.size();
    }

    inline double Stopwatch::median_ns() const
    {
        if (measurements_.empty())
            return 0.0;
        auto sorted = measurements_;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        if (n % 2 == 0)
            return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        return sorted[n / 2];
    }

    inline double Stopwatch::stddev_ns() const
    {
        if (measurements_.size() < 2)
            return 0.0;
        double mean = mean_ns();
        double sum  = 0.0;
        for (auto x : measurements_)
            sum += (x - mean) * (x - mean);
        return std::sqrt(sum / (measurements_.size() - 1));
    }

    inline void Stopwatch::print_stats(const char* label) const
    {
        std::cout << "\n=== " << label << " ===\n";
        std::cout << "  Runs:    " << count() << "\n";
        std::cout << "  Min:     " << std::fixed << std::setprecision(3) << min_ns() / 1000.0 << " µs\n";
        std::cout << "  Max:     " << max_ns() / 1000.0 << " µs\n";
        std::cout << "  Mean:    " << mean_ns() / 1000.0 << " µs\n";
        std::cout << "  Median:  " << median_ns() / 1000.0 << " µs\n";
        std::cout << "  StdDev:  " << stddev_ns() / 1000.0 << " µs\n";
    }

} // namespace bench

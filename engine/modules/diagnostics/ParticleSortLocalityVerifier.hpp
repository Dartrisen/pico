#pragma once

#include <cmath>

namespace pico::diagnostics
{

struct ParticleSortLocalityVerificationResult
{
    bool   passed{false};
    double unsorted_time_ms{0.0};
    double sorted_time_ms{0.0};
    double speedup{0.0};
    double improvement_pct{0.0};
};

class ParticleSortLocalityVerifier
{
public:
    explicit ParticleSortLocalityVerifier(double minimum_speedup = 1.02) : minimum_speedup_(minimum_speedup) {}

    [[nodiscard]] ParticleSortLocalityVerificationResult verify(double unsorted_time_ms, double sorted_time_ms) const
    {
        ParticleSortLocalityVerificationResult result;
        result.unsorted_time_ms = unsorted_time_ms;
        result.sorted_time_ms   = sorted_time_ms;

        if (unsorted_time_ms <= 0.0 || sorted_time_ms <= 0.0)
        {
            return result;
        }

        result.speedup         = unsorted_time_ms / sorted_time_ms;
        result.improvement_pct = (1.0 - sorted_time_ms / unsorted_time_ms) * 100.0;
        result.passed          = std::isfinite(result.speedup) && result.speedup >= minimum_speedup_;

        return result;
    }

private:
    double minimum_speedup_;
};

} // namespace pico::diagnostics
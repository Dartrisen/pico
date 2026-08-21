#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace pico::diagnostics
{

struct CurrentVerificationResult
{
    bool   passed{false};
    double avg_measured_current{0.0};
    double expected_current{0.0};
    double current_error_pct{0.0};
    double max_ampere_residual{0.0};
};

class CurrentVerifier
{
public:
    // expected_current = q * n0 * v_drift
    explicit CurrentVerifier(double expected_current, double tolerance_pct = 2.0) : expected_current_(expected_current), tolerance_pct_(tolerance_pct) {}

    void record_step(double avg_jx_deposited, double max_ampere_residual)
    {
        avg_currents_.push_back(avg_jx_deposited);
        ampere_residuals_.push_back(max_ampere_residual);
    }

    CurrentVerificationResult verify() const
    {
        CurrentVerificationResult res;
        res.expected_current = expected_current_;

        if (avg_currents_.empty())
        {
            return res;
        }

        // Average measured current across all recorded simulation steps
        const double sum_j       = std::accumulate(avg_currents_.begin(), avg_currents_.end(), 0.0);
        res.avg_measured_current = sum_j / static_cast<double>(avg_currents_.size());

        if (std::abs(expected_current_) > 1e-12)
        {
            res.current_error_pct = (std::abs(res.avg_measured_current - expected_current_) / std::abs(expected_current_)) * 100.0;
        }
        else
        {
            res.current_error_pct = std::abs(res.avg_measured_current) * 100.0;
        }

        res.max_ampere_residual = *std::max_element(ampere_residuals_.begin(), ampere_residuals_.end());

        res.passed = (res.current_error_pct <= tolerance_pct_);
        return res;
    }

private:
    double              expected_current_;
    double              tolerance_pct_;
    std::vector<double> avg_currents_;
    std::vector<double> ampere_residuals_;
};

} // namespace pico::diagnostics
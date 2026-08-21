#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

namespace pico::diagnostics
{

struct VerificationResult
{
    bool   passed{false};
    double max_energy_drift_pct{0.0};
    double measured_freq{0.0};
    double expected_freq{0.0};
    double freq_error_pct{0.0};
};

class PlasmaWaveVerifier
{
public:
    PlasmaWaveVerifier(double dt, double dx, std::size_t ppc, double n0 = 1.0) : dt_(dt), dx_(dx), ppc_(ppc), n0_(n0)
    {
        // Physical density n_eff = n0 / dx
        // Plasma frequency w_p = sqrt(n_eff) = sqrt(n0 / dx)
        expected_wp_ = std::sqrt(n0_ / dx_);
    }

    void record_step(double e_ex_field, double total_energy)
    {
        ex_energies_.push_back(e_ex_field);
        total_energies_.push_back(total_energy);
    }

    VerificationResult verify(double energy_drift_tol_pct = 2.0, double freq_tol_pct = 5.0) const
    {
        VerificationResult res;
        res.expected_freq = expected_wp_;

        if (total_energies_.empty())
            return res;

        // 1. Total Energy Conservation Check
        const double e0        = total_energies_.front();
        double       max_drift = 0.0;
        for (double e : total_energies_)
        {
            const double drift = (std::abs(e - e0) / e0) * 100.0;
            max_drift          = std::max(max_drift, drift);
        }
        res.max_energy_drift_pct = max_drift;

        // 2. Measure Wave Frequency via Ex Field Energy Peaks (oscillates at 2 * w_p)
        double max_ex = 0.0;
        for (double e : ex_energies_)
        {
            max_ex = std::max(max_ex, e);
        }

        const double        threshold = 0.2 * max_ex;
        std::vector<double> peak_times;

        for (std::size_t i = 2; i < ex_energies_.size() - 2; ++i)
        {
            if (ex_energies_[i] > threshold && ex_energies_[i] >= ex_energies_[i - 1] && ex_energies_[i] >= ex_energies_[i - 2] && ex_energies_[i] >= ex_energies_[i + 1] &&
                ex_energies_[i] >= ex_energies_[i + 2])
            {
                const double t_curr = i * dt_;
                if (peak_times.empty() || (t_curr - peak_times.back()) > 10.0 * dt_)
                {
                    peak_times.push_back(t_curr);
                }
            }
        }

        if (peak_times.size() >= 2)
        {
            std::vector<double> periods;
            for (std::size_t i = 1; i < peak_times.size(); ++i)
            {
                periods.push_back(peak_times[i] - peak_times[i - 1]);
            }

            const double avg_period_field = std::accumulate(periods.begin(), periods.end(), 0.0) / periods.size();

            // T_field = pi / w_p => w_p = pi / T_field
            res.measured_freq  = std::numbers::pi / avg_period_field;
            res.freq_error_pct = (std::abs(res.measured_freq - expected_wp_) / expected_wp_) * 100.0;
        }

        res.passed = (res.max_energy_drift_pct <= energy_drift_tol_pct) && (res.freq_error_pct <= freq_tol_pct);
        return res;
    }

private:
    double              dt_;
    double              dx_;
    std::size_t         ppc_;
    double              n0_;
    double              expected_wp_;
    std::vector<double> ex_energies_;
    std::vector<double> total_energies_;
};

} // namespace pico::diagnostics

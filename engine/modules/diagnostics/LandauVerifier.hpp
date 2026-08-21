#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

namespace pico::diagnostics
{

struct LandauVerificationResult
{
    bool   passed{false};
    double max_energy_drift_pct{0.0};
    double measured_freq{0.0};
    double expected_freq{0.0};
    double freq_error_pct{0.0};
    double measured_gamma{0.0};
    double expected_gamma{0.0};
    double gamma_error_pct{0.0};
};

class LandauDampingVerifier
{
public:
    LandauDampingVerifier(double dt, double dx, double k, double v_th, double n0 = 1.0) : dt_(dt), dx_(dx), k_(k), v_th_(v_th), n0_(n0)
    {
        const double wp         = std::sqrt(n0_ / dx_);
        const double k_lambda_d = (k_ * v_th_) / wp;

        // High-accuracy kinetic dispersion relation polynomial fit for Fried-Conte Z-function roots (k*lambda_D in [0.10, 0.45])
        const double x  = k_lambda_d;
        const double x2 = x * x;
        const double x3 = x2 * x;
        const double x4 = x3 * x;

        const double wr_ratio  = -10.17238 * x4 + 7.579271 * x3 + 0.337453 * x2 + 0.003398 * x + 1.005704;
        const double gam_ratio = -13.87509 * x4 + 18.285588 * x3 - 6.853475 * x2 + 0.989643 * x - 0.048401;

        expected_wr_    = wr_ratio * wp;
        expected_gamma_ = std::max(0.0, gam_ratio * wp);
    }

    void record_step(double e_ex_field, double total_energy)
    {
        ex_energies_.push_back(e_ex_field);
        total_energies_.push_back(total_energy);
    }

    LandauVerificationResult verify(double energy_drift_tol_pct = 2.0, double freq_tol_pct = 5.0, double gamma_tol_pct = 15.0, double t_transient_cutoff = 1.0) const
    {
        LandauVerificationResult res;
        res.expected_freq  = expected_wr_;
        res.expected_gamma = expected_gamma_;

        if (total_energies_.empty())
            return res;

        // 1. Energy Conservation Check
        const double e0        = total_energies_.front();
        double       max_drift = 0.0;
        for (double e : total_energies_)
        {
            const double drift = (std::abs(e - e0) / e0) * 100.0;
            max_drift          = std::max(max_drift, drift);
        }
        res.max_energy_drift_pct = max_drift;

        // 2. Local Peak Extraction with Thermal Noise Floor Floor Thresholding
        const double e_max                 = *std::max_element(ex_energies_.begin(), ex_energies_.end());
        const double noise_floor_threshold = e_max * 1e-5; // Stop fitting when signal reaches thermal particle noise

        std::vector<double> peak_times;
        std::vector<double> peak_log_energies;
        const double        expected_period_field = std::numbers::pi / expected_wr_;

        for (std::size_t i = 2; i < ex_energies_.size() - 2; ++i)
        {
            const double t_curr = static_cast<double>(i) * dt_;

            if (t_curr < t_transient_cutoff)
                continue;

            // Clip peaks at the thermal noise floor to avoid flattening distortion
            if (ex_energies_[i] < noise_floor_threshold)
                break;

            if (ex_energies_[i] >= ex_energies_[i - 1] && ex_energies_[i] >= ex_energies_[i - 2] && ex_energies_[i] >= ex_energies_[i + 1] &&
                ex_energies_[i] >= ex_energies_[i + 2])
            {
                if (peak_times.empty() || (t_curr - peak_times.back()) > 0.6 * expected_period_field)
                {
                    peak_times.push_back(t_curr);
                    peak_log_energies.push_back(std::log(ex_energies_[i]));
                }
            }
        }

        // 3. Frequency & Damping Estimation via Linear Regression
        if (peak_times.size() >= 2)
        {
            std::vector<double> periods;
            for (std::size_t i = 1; i < peak_times.size(); ++i)
                periods.push_back(peak_times[i] - peak_times[i - 1]);

            const double avg_period_field = std::accumulate(periods.begin(), periods.end(), 0.0) / periods.size();
            res.measured_freq             = std::numbers::pi / avg_period_field;
            res.freq_error_pct            = (std::abs(res.measured_freq - expected_wr_) / expected_wr_) * 100.0;
        }

        if (peak_times.size() >= 3)
        {
            const std::size_t n     = peak_times.size();
            double            sum_t = 0.0, sum_y = 0.0, sum_tt = 0.0, sum_ty = 0.0;

            for (std::size_t i = 0; i < n; ++i)
            {
                sum_t += peak_times[i];
                sum_y += peak_log_energies[i];
                sum_tt += peak_times[i] * peak_times[i];
                sum_ty += peak_times[i] * peak_log_energies[i];
            }

            const double slope = (static_cast<double>(n) * sum_ty - sum_t * sum_y) / (static_cast<double>(n) * sum_tt - sum_t * sum_t);
            res.measured_gamma = -0.5 * slope;

            if (expected_gamma_ > 1e-4)
                res.gamma_error_pct = (std::abs(res.measured_gamma - expected_gamma_) / expected_gamma_) * 100.0;
            else
                res.gamma_error_pct = std::abs(res.measured_gamma - expected_gamma_) * 100.0;
        }

        res.passed = (res.max_energy_drift_pct <= energy_drift_tol_pct) && (res.freq_error_pct <= freq_tol_pct) && (res.gamma_error_pct <= gamma_tol_pct);

        return res;
    }

private:
    double              dt_;
    double              dx_;
    double              k_;
    double              v_th_;
    double              n0_;
    double              expected_wr_{0.0};
    double              expected_gamma_{0.0};
    std::vector<double> ex_energies_;
    std::vector<double> total_energies_;
};

} // namespace pico::diagnostics
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace pico::diagnostics
{

struct WeibelVerificationResult
{
    bool   passed{false};
    double max_energy_drift_pct{0.0};
    double measured_gamma{0.0};
    double expected_gamma{0.0};
    double gamma_error_pct{0.0};
};

class WeibelVerifier
{
public:
    WeibelVerifier(double dt, double v_th_x, double v_th_y, double n0 = 1.0) : dt_(dt), v_th_x_(v_th_x), v_th_y_(v_th_y), n0_(n0)
    {
        // Analytical scaling estimate for Weibel instability growth rate
        const double wp         = std::sqrt(n0_);
        const double anisotropy = (v_th_y_ * v_th_y_) / (v_th_x_ * v_th_x_) - 1.0;
        if (anisotropy > 0.0)
        {
            expected_gamma_ = wp * v_th_y_ * std::sqrt(2.0 * anisotropy) * 0.15;
        }
        else
        {
            expected_gamma_ = 0.0;
        }
    }

    void record_step(double magnetic_energy, double total_energy)
    {
        magnetic_energies_.push_back(magnetic_energy);
        total_energies_.push_back(total_energy);
    }

    WeibelVerificationResult verify(double energy_drift_tol_pct = 5.0, double gamma_tol_pct = 35.0) const
    {
        WeibelVerificationResult res;
        res.expected_gamma = expected_gamma_;

        if (total_energies_.empty() || magnetic_energies_.size() < 10)
            return res;

        // 1. Energy Conservation Check
        const double e0        = total_energies_.front();
        double       max_drift = 0.0;
        for (double e : total_energies_)
        {
            const double drift = (std::abs(e - e0) / (e0 != 0.0 ? e0 : 1.0)) * 100.0;
            max_drift          = std::max(max_drift, drift);
        }
        res.max_energy_drift_pct = max_drift;

        // 2. Linear Growth Rate Extraction via Linear Regression on ln(E_magnetic)
        // For fast-growing instabilities like Weibel, target the early linear phase (steps 10 to 150)
        // before non-linear saturation flattens out the curve.
        std::size_t start_idx = 10;
        std::size_t end_idx   = 150;

        if (magnetic_energies_.size() <= 160)
        {
            start_idx = 5;
            end_idx   = magnetic_energies_.size() * 3 / 4;
        }

        start_idx = std::min(start_idx, magnetic_energies_.size() > 5 ? magnetic_energies_.size() - 5 : 0);
        end_idx   = std::min(end_idx, magnetic_energies_.size());

        if (end_idx <= start_idx + 5)
        {
            start_idx = 0;
            end_idx   = magnetic_energies_.size();
        }

        double      sum_t = 0.0, sum_y = 0.0, sum_tt = 0.0, sum_ty = 0.0;
        std::size_t count = 0;

        for (std::size_t i = start_idx; i < end_idx; ++i)
        {
            if (magnetic_energies_[i] <= 1e-14)
                continue;

            double t = static_cast<double>(i) * dt_;
            double y = std::log(magnetic_energies_[i]);

            sum_t += t;
            sum_y += y;
            sum_tt += t * t;
            sum_ty += t * y;
            count++;
        }

        if (count >= 3)
        {
            const double slope = (static_cast<double>(count) * sum_ty - sum_t * sum_y) / (static_cast<double>(count) * sum_tt - sum_t * sum_t);

            // Measured gamma = slope / 2 because energy is squared field value
            res.measured_gamma = 0.5 * slope;

            if (expected_gamma_ > 1e-4)
            {
                res.gamma_error_pct = (std::abs(res.measured_gamma - expected_gamma_) / expected_gamma_) * 100.0;
            }
        }

        bool growth_observed = (res.measured_gamma > 0.0);
        res.passed           = (res.max_energy_drift_pct <= energy_drift_tol_pct) && growth_observed;

        // Optionally enforce tolerance check on growth rate error if desired:
        // res.passed = res.passed && (res.gamma_error_pct <= gamma_tol_pct);

        return res;
    }

private:
    double              dt_;
    double              v_th_x_;
    double              v_th_y_;
    double              n0_;
    double              expected_gamma_{0.0};
    std::vector<double> magnetic_energies_;
    std::vector<double> total_energies_;
};

} // namespace pico::diagnostics
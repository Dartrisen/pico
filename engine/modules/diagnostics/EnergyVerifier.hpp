#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace pico::diagnostics
{

struct EnergyVerificationResult
{
    bool   passed{false};
    double initial_energy{0.0};
    double final_energy{0.0};
    double final_injected_energy{0.0};
    double max_energy_drift_pct{0.0};
    double avg_energy_drift_pct{0.0};
};

class EnergyVerifier
{
public:
    explicit EnergyVerifier(double drift_tolerance_pct = 2.0) : drift_tolerance_pct_(drift_tolerance_pct) {}

    /// Backwards-compatible step recorder
    void record_step(double field_energy, double kinetic_energy, double injected_energy = 0.0)
    {
        field_energies_.push_back(field_energy);
        kinetic_energies_.push_back(kinetic_energy);
        injected_energies_.push_back(injected_energy);
        total_energies_.push_back(field_energy + kinetic_energy);
    }

    [[nodiscard]] EnergyVerificationResult verify() const
    {
        EnergyVerificationResult res;
        if (total_energies_.empty())
        {
            return res;
        }

        res.initial_energy        = total_energies_.front();
        res.final_energy          = total_energies_.back();
        res.final_injected_energy = injected_energies_.back();

        const double e0        = res.initial_energy;
        double       max_drift = 0.0;
        double       sum_drift = 0.0;

        for (std::size_t i = 0; i < total_energies_.size(); ++i)
        {
            const double e_current  = total_energies_[i];
            const double e_injected = injected_energies_[i];
            const double e_expected = e0 + e_injected;

            if (std::abs(e_expected) < 1e-12)
                continue;

            const double drift = (std::abs(e_current - e_expected) / e_expected) * 100.0;
            max_drift          = std::max(max_drift, drift);
            sum_drift += drift;
        }

        res.max_energy_drift_pct = max_drift;
        res.avg_energy_drift_pct = sum_drift / static_cast<double>(total_energies_.size());
        res.passed               = (max_drift <= drift_tolerance_pct_);

        return res;
    }

    [[nodiscard]] const std::vector<double>& field_energies() const noexcept
    {
        return field_energies_;
    }
    [[nodiscard]] const std::vector<double>& kinetic_energies() const noexcept
    {
        return kinetic_energies_;
    }
    [[nodiscard]] const std::vector<double>& injected_energies() const noexcept
    {
        return injected_energies_;
    }
    [[nodiscard]] const std::vector<double>& total_energies() const noexcept
    {
        return total_energies_;
    }

private:
    double              drift_tolerance_pct_;
    std::vector<double> field_energies_;
    std::vector<double> kinetic_energies_;
    std::vector<double> injected_energies_;
    std::vector<double> total_energies_;
};

} // namespace pico::diagnostics

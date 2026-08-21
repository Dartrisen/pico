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
    double max_energy_drift_pct{0.0};
    double avg_energy_drift_pct{0.0};
};

class EnergyVerifier
{
public:
    explicit EnergyVerifier(double drift_tolerance_pct = 2.0) : drift_tolerance_pct_(drift_tolerance_pct) {}

    void record_step(double field_energy, double kinetic_energy)
    {
        const double total = field_energy + kinetic_energy;
        field_energies_.push_back(field_energy);
        kinetic_energies_.push_back(kinetic_energy);
        total_energies_.push_back(total);
    }

    EnergyVerificationResult verify() const
    {
        EnergyVerificationResult res;
        if (total_energies_.empty())
        {
            return res;
        }

        res.initial_energy = total_energies_.front();
        res.final_energy   = total_energies_.back();

        const double e0 = res.initial_energy;
        if (std::abs(e0) < 1e-12)
        {
            return res;
        }

        double max_drift = 0.0;
        double sum_drift = 0.0;

        for (double e : total_energies_)
        {
            const double drift = (std::abs(e - e0) / e0) * 100.0;
            max_drift          = std::max(max_drift, drift);
            sum_drift += drift;
        }

        res.max_energy_drift_pct = max_drift;
        res.avg_energy_drift_pct = sum_drift / static_cast<double>(total_energies_.size());
        res.passed               = (max_drift <= drift_tolerance_pct_);

        return res;
    }

private:
    double              drift_tolerance_pct_;
    std::vector<double> field_energies_;
    std::vector<double> kinetic_energies_;
    std::vector<double> total_energies_;
};

} // namespace pico::diagnostics

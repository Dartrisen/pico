#pragma once

#include "data/field/include/field_em.hpp"

#include <cmath>
#include <cstddef>

namespace pico::modules::injector
{

template <std::size_t BLOCK_SIZE = 64>
struct NoInjector
{
    void inject(EMFields<BLOCK_SIZE>& /*fields*/, double /*t*/, double /*dt*/) const noexcept {}
};

/**
 * @brief Relativistic Plane Wave Laser Injector
 * Units: normalized (c = 1, omega_0 = 1, k_0 = 1, E_0 = a0)
 */
template <std::size_t BLOCK_SIZE = 64>
class PlaneWaveLaserInjector
{
public:
    PlaneWaveLaserInjector(std::size_t phys_cell_idx, float a0, float tau_duration, float t_peak) : phys_cell_idx_(phys_cell_idx), a0_(a0), tau_(tau_duration), t_peak_(t_peak) {}

    void inject(EMFields<BLOCK_SIZE>& fields, double t, double dt) const noexcept
    {
        if (a0_ <= 0.0f)
            return;

        const auto&       grid    = fields.E.grid();
        const std::size_t buf_idx = grid.physical_to_buffer(phys_cell_idx_);
        const float       dx      = static_cast<float>(grid.cell_size());

        // 1. E_y evaluated at node x_i, time t^n
        const float t_rel_e = static_cast<float>(t) - t_peak_;
        const float env_e   = (tau_ > 0.0f) ? std::exp(-(t_rel_e * t_rel_e) / (tau_ * tau_)) : 1.0f;
        const float ey_val  = a0_ * env_e * std::sin(t_rel_e);

        // 2. B_z evaluated at x_{i+1/2}, time t^{n+1/2} for +x propagation phase alignment
        const float t_rel_b = static_cast<float>(t + 0.5 * dt - 0.5 * dx) - t_peak_;
        const float env_b   = (tau_ > 0.0f) ? std::exp(-(t_rel_b * t_rel_b) / (tau_ * tau_)) : 1.0f;
        const float bz_val  = a0_ * env_b * std::sin(t_rel_b);

        fields.E.field_y(buf_idx) += ey_val;
        fields.B.field_z(buf_idx) += bz_val;
    }

private:
    std::size_t phys_cell_idx_{0};
    float       a0_{1.0f};
    float       tau_{10.0f};
    float       t_peak_{25.0f};
};

} // namespace pico::modules::injector
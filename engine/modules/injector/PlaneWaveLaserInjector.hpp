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
    PlaneWaveLaserInjector(std::size_t phys_cell_idx,
                           float       a0,           // Normalized vector potential / Peak field amplitude
                           float       tau_duration, // Pulse duration tau (in 1/omega_0 units)
                           float       t_peak)             // Peak temporal offset (in 1/omega_0 units)
            : phys_cell_idx_(phys_cell_idx), a0_(a0), tau_(tau_duration), t_peak_(t_peak)
    {
    }

    void inject(EMFields<BLOCK_SIZE>& fields, double t, double /*dt*/) const noexcept
    {
        const auto&       grid    = fields.E.grid();
        const std::size_t buf_idx = grid.physical_to_buffer(phys_cell_idx_);

        const float t_rel    = static_cast<float>(t) - t_peak_;
        const float envelope = (tau_ > 0.0f) ? std::exp(-(t_rel * t_rel) / (tau_ * tau_)) : 1.0f;
        const float carrier  = std::sin(t_rel);

        const float ey_val = a0_ * envelope * carrier;
        const float bz_val = ey_val; // Unidirectional propagation along +x (c = 1)

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
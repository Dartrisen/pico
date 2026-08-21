#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/boundary/SilverMuller.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/PlaneWaveLaserInjector.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

int main()
{
    constexpr std::size_t grid_cells = 256; // Domain length L = 12.8
    constexpr double      dx         = 0.05;
    constexpr double      dt         = 0.001;
    constexpr std::size_t BS         = 64;

    constexpr double t_check = 7.0;  // Expected envelope peak at x = 4.0
    constexpr double t_final = 20.0; // Pulse exits domain

    const std::size_t mid_check_step = static_cast<std::size_t>(std::round(t_check / dt));
    const std::size_t nsteps         = static_cast<std::size_t>(std::round(t_final / dt));

    constexpr float       a0          = 1.0f;
    constexpr float       tau         = 1.0f;
    constexpr float       t_peak      = 3.0f;
    constexpr std::size_t inject_cell = 0;

    assert(dx > 0.95 * dt && "CFL condition violated.");

    Grid grid(grid_cells, dx);

    using Shape     = kernels::shapes::Shape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::SilverMullerFieldBoundary<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::PlaneWaveLaserInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    Injector laser_injector(inject_cell, a0, tau, t_peak);
    EngineT  engine_instance{grid, /*ppc=*/0, BoundaryF{}, BoundaryP{}, std::move(laser_injector), /*n0=*/0.0f};

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    PICApp app(std::move(wrapper), dt);

    double max_field_energy   = 0.0;
    double final_field_energy = 0.0;
    double measured_peak_x    = 0.0;

    app.run(nsteps,
            [&](int step_idx)
            {
                const std::size_t step = static_cast<std::size_t>(step_idx);
                auto&             eng  = concrete_wrapper->engine();

                double              e_field = 0.0;
                std::vector<double> local_u(grid_cells, 0.0);

                for (std::size_t i = 0; i < grid_cells; ++i)
                {
                    const std::size_t buf_i = grid.physical_to_buffer(i);
                    const float       ey    = eng.fields().E.field_y(buf_i);
                    const float       bz    = eng.fields().B.field_z(buf_i);

                    const double energy_density = 0.5 * static_cast<double>(ey * ey + bz * bz);
                    local_u[i]                  = energy_density;
                    e_field += energy_density * dx;
                }

                max_field_energy = std::max(max_field_energy, e_field);

                if (step == mid_check_step)
                {
                    // Smooth u over optical wavelength (~2*pi / dx cells) to extract pulse envelope
                    constexpr int window_half_width = 30;
                    double        max_env_density   = 0.0;
                    std::size_t   max_env_idx       = 0;

                    for (std::size_t i = window_half_width; i < grid_cells - window_half_width; ++i)
                    {
                        double sum = 0.0;
                        for (int w = -window_half_width; w <= window_half_width; ++w)
                        {
                            sum += local_u[i + w];
                        }
                        if (sum > max_env_density)
                        {
                            max_env_density = sum;
                            max_env_idx     = i;
                        }
                    }
                    measured_peak_x = static_cast<double>(max_env_idx) * dx;
                }

                if (step == nsteps - 1)
                {
                    final_field_energy = e_field;
                }
            });

    const double expected_peak_x = (static_cast<double>(mid_check_step) * dt) - static_cast<double>(t_peak);
    const double pos_error       = std::abs(measured_peak_x - expected_peak_x);

    const double reflection_pct = (final_field_energy / max_field_energy) * 100.0;
    const bool   passed         = (reflection_pct < 2.0) && (pos_error <= 2.0 * dx) && (max_field_energy > 0.0);

    pico::ui::VerificationReport report("Laser Wave Injection & Boundary Absorption Verification", passed);

    report.add_sci_row("Peak Transverse Energy", max_field_energy);
    report.add_sci_row("Final Residual Energy", final_field_energy);
    report.add_pct_row("Boundary Reflection Ratio", reflection_pct, reflection_pct < 2.0);
    report.add_fixed_row("Measured Peak Energy Position", measured_peak_x, 4, "x");
    report.add_fixed_row("Expected Peak Energy Position", expected_peak_x, 4, "x");

    report.print();
    return report.passed() ? 0 : 1;
}

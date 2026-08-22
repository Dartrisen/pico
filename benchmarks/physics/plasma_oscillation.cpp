#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "app/VerificationReport.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/PeriodicParticleBoundary.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/diagnostics/PlasmaWaveVerifier.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/injector/PlaneWaveLaserInjector.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>

struct EnergyMetrics
{
    double e_ex{0.0};
    double e_total{0.0};
};

// Applies sinusoidal velocity perturbation while preserving initial charge and mass scaling
template <typename Engine>
void apply_wave_perturbation(Engine& engine, std::size_t grid_cells, double dx)
{
    auto&           particles = engine.particles();
    const double    L         = static_cast<double>(grid_cells) * dx;
    const double    k         = 2.0 * std::numbers::pi / L;
    constexpr float v0        = 0.05f;
    const double    dx_p      = L / static_cast<double>(particles.active_particles());

    std::size_t global_idx = 0;
    for (auto& block : particles)
    {
        for (std::size_t i = 0; i < block.activeCount; ++i, ++global_idx)
        {
            const double x0 = (static_cast<double>(global_idx) + 0.5) * dx_p;

            block.position_x[i] = static_cast<float>(x0);

            const float local_m = block.mass[i];
            block.momentum_x[i] = local_m * v0 * std::sin(static_cast<float>(k * x0));
            block.momentum_y[i] = 0.0f;
            block.momentum_z[i] = 0.0f;
        }
    }
}

// Single-pass computation for longitudinal Ex field energy and total system energy
template <typename Engine>
EnergyMetrics compute_energies(const Engine& eng, std::size_t grid_cells, double dx, std::size_t ppc)
{
    double      e_ex    = 0.0;
    double      e_field = 0.0;
    const auto& fields  = eng.fields();

    for (std::size_t i = 0; i < grid_cells; ++i)
    {
        const double ex = fields.E.field_x(i);
        const double ey = fields.E.field_y(i);
        const double ez = fields.E.field_z(i);
        const double bx = fields.B.field_x(i);
        const double by = fields.B.field_y(i);
        const double bz = fields.B.field_z(i);

        const double ex2 = ex * ex;
        e_ex += 0.5 * ex2 * dx;
        e_field += 0.5 * (ex2 + ey * ey + ez * ez + bx * bx + by * by + bz * bz) * dx;
    }

    double e_kin = 0.0;
    for (const auto& block : eng.particles())
    {
        for (std::size_t i = 0; i < block.activeCount; ++i)
        {
            const double px = block.momentum_x[i];
            const double py = block.momentum_y[i];
            const double pz = block.momentum_z[i];
            const double m  = block.mass[i];
            e_kin += 0.5 * (px * px + py * py + pz * pz) / m;
        }
    }
    e_kin /= static_cast<double>(ppc);

    return {e_ex, e_field + e_kin};
}

int main()
{
    constexpr std::size_t grid_cells = 256;
    constexpr double      dx         = 0.1;
    constexpr double      dt         = 0.002;
    constexpr std::size_t ppc        = 100;
    constexpr std::size_t nsteps     = 1000;
    constexpr std::size_t BS         = 64;
    constexpr float       target_n0  = 2.0f;

    assert(dx > 0.95 * dt && "CFL condition violated.");

    Grid grid(grid_cells, dx);

    using Shape     = kernels::shapes::Shape<3>;
    using Field     = pico::modules::field::YeeMaxwell<BS>;
    using Push      = pico::modules::pusher::BorisPusher<BS>;
    using Gather    = pico::modules::gather::Gather<Shape, BS>;
    using Dep       = pico::modules::deposit::SimpleDeposit<Shape, BS>;
    using BoundaryF = pico::modules::boundary::PeriodicBoundaryFieldHandler<BS>;
    using BoundaryP = pico::modules::boundary::PeriodicBoundaryParticleHandler<BS>;
    using Injector  = pico::modules::injector::NoInjector<BS>;

    using EngineT = PICEngine<Field, Gather, Push, Dep, BoundaryF, BoundaryP, Injector, BS>;

    // 1. Initialize Engine & Setup Initial Perturbation
    EngineT engine_instance{grid, ppc, target_n0};
    apply_wave_perturbation(engine_instance, grid_cells, dx);

    auto  wrapper          = std::make_unique<EngineWrapper<EngineT>>(std::move(engine_instance));
    auto* concrete_wrapper = wrapper.get();

    // 2. Setup Verification & Application Driver
    pico::diagnostics::PlasmaWaveVerifier verifier(dt, dx, ppc, target_n0);

    PICApp app(std::move(wrapper), dt);

    // 3. Execution Loop
    app.run(nsteps,
            [&](int /*step*/)
            {
                const auto [e_ex, e_total] = compute_energies(concrete_wrapper->engine(), grid_cells, dx, ppc);
                verifier.record_step(e_ex, e_total);
            });

    // 4. Verification Analysis & Reporting
    const auto res = verifier.verify(/*drift_tol=*/2.0, /*freq_tol=*/5.0);

    std::ostringstream title_ss;
    title_ss << "Physics Verification Summary (n0 = " << std::fixed << std::setprecision(1) << target_n0 << ")";

    pico::ui::VerificationReport report("Plasma Wave Physics Verification", res.passed, title_ss.str());

    report.add_pct_row("Max Energy Drift", res.max_energy_drift_pct, true);
    report.add_fixed_row("Measured Frequency (wp)", res.measured_freq, 4, "rad/s");
    report.add_fixed_row("Expected Frequency (wp)", res.expected_freq, 4, "rad/s");
    report.add_pct_row("Frequency Error", res.freq_error_pct, res.passed);

    report.print();
    return report.passed() ? 0 : 1;
}

#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace pico::diagnostics
{
class ScopedTimer
{
private:
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_time_{clock::now()};
    double&           accum_target_ns_;

public:
    explicit ScopedTimer(double& accum_target_ns) : accum_target_ns_(accum_target_ns) {}

    ~ScopedTimer()
    {
        auto end_time = clock::now();
        accum_target_ns_ += std::chrono::duration<double, std::nano>(end_time - start_time_).count();
    }
};

struct PerfStats
{
    double gather_ns{0.0};
    double push_ns{0.0};
    double particle_boundary_ns{0.0};
    double deposit_ns{0.0};
    double field_boundary_ns{0.0};
    double field_solver_ns{0.0};
    double sort_ns{0.0};
    double step_total_ns{0.0};

    void reset()
    {
        *this = PerfStats{};
    }

    void print_report(std::size_t total_particles, std::size_t steps) const
    {
        const double total_sec            = step_total_ns * 1e-9;
        const double total_particle_steps = static_cast<double>(total_particles * steps);
        const double mps                  = (total_particle_steps / total_sec) / 1e6;

        std::cout << "\n================= PIC Engine Performance Report =================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total Steps:        " << steps << "\n";
        std::cout << "Active Particles:   " << total_particles << "\n";
        std::cout << "Total Core Time:    " << total_sec * 1000.0 << " ms\n";
        std::cout << "Throughput:         " << mps << " MPSP (Million Particle Steps/sec)\n";
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "Phase Breakdown:\n";

        auto print_row = [this](std::string_view name, double ns)
        {
            const double ms  = ns * 1e-6;
            const double pct = (step_total_ns > 0.0) ? (ns / step_total_ns) * 100.0 : 0.0;
            std::cout << "  " << std::left << std::setw(20) << name << ": " << std::right << std::setw(8) << ms
                      << " ms (" << std::setw(5) << pct << "%)\n";
        };

        print_row("Gather", gather_ns);
        print_row("Pusher", push_ns);
        print_row("Deposit", deposit_ns);
        print_row("Particle Boundary", particle_boundary_ns);
        print_row("Field Boundary", field_boundary_ns);
        print_row("Field Solver", field_solver_ns);
        print_row("Particle Sorter", sort_ns);
        std::cout << "=================================================================\n\n";
    }
};
} // namespace pico::diagnostics
#pragma once

#include "IEngine.hpp"
#include "engine/perf/PipelineProfiler.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace pico::ui
{
constexpr const char* RESET   = "\033[0m";
constexpr const char* BOLD    = "\033[1m";
constexpr const char* CYAN    = "\033[1;36m";
constexpr const char* GREEN   = "\033[1;32m";
constexpr const char* YELLOW  = "\033[1;33m";
constexpr const char* MAGENTA = "\033[1;35m";
constexpr const char* GRAY    = "\033[1;30m";
constexpr const char* WHITE   = "\033[1;37m";
} // namespace pico::ui

class SimMonitor
{
public:
    using clock = std::chrono::high_resolution_clock;

    void start(const IEngine& engine, double dt, int nsteps, int log_interval)
    {
        dt_                 = dt;
        nsteps_             = nsteps;
        log_interval_       = log_interval;
        start_time_         = clock::now();
        last_interval_time_ = start_time_;

        print_banner(engine);
        print_header();
    }

    void log_step(int step, const IEngine& engine)
    {
        const auto   now          = clock::now();
        const double interval_sec = std::chrono::duration<double>(now - last_interval_time_).count();
        const double total_sec    = std::chrono::duration<double>(now - start_time_).count();
        last_interval_time_       = now;

        const double progress    = static_cast<double>(step) / nsteps_;
        const double avg_step_ms = (step > 0 && interval_sec > 0.0) ? (interval_sec / log_interval_) * 1000.0 : 0.0;
        const double mpsp =
                (interval_sec > 0.0) ? ((engine.total_particles() * log_interval_) / interval_sec) / 1e6 : 0.0;
        const double eta_sec = (progress > 0.0 && progress < 1.0) ? (total_sec / progress) * (1.0 - progress) : 0.0;

        constexpr int bar_width = 10;
        const int     filled    = static_cast<int>(std::round(bar_width * progress));
        std::string   bar       = "[";
        for (int i = 0; i < bar_width; ++i)
            bar += (i < filled ? "█" : "░");
        bar += "]";

        using namespace pico::ui;
        std::cout << CYAN << std::left << std::setw(8) << step << RESET << std::setw(12) << std::fixed
                  << std::setprecision(4) << (step * dt_) << YELLOW << std::setw(10) << std::fixed
                  << std::setprecision(2) << avg_step_ms << " ms " << RESET << GREEN << std::setw(10) << std::fixed
                  << std::setprecision(1) << mpsp << " MPSP" << RESET << MAGENTA << bar << " " << std::setw(4)
                  << static_cast<int>(progress * 100.0) << "%" << RESET << GRAY << std::setw(8) << std::fixed
                  << std::setprecision(1) << eta_sec << "s" << RESET << "\n";
    }

    void print_final_report(const IEngine& engine) const
    {
        using namespace pico::ui;
        const double total_wall_sec = std::chrono::duration<double>(clock::now() - start_time_).count();

        std::cout << GRAY
                  << "----------------------------------------------------------------------------------------\n"
                  << RESET;
        std::cout << GREEN << BOLD << " ✔ Simulation finished in " << std::fixed << std::setprecision(2)
                  << total_wall_sec << " seconds.\n\n"
                  << RESET;

        const auto&  profiler      = engine.profiler();
        const double profiled_time = profiler.total_seconds();

        std::cout << BOLD << WHITE << " Pipeline Kernel Execution Breakdown:\n" << RESET;
        std::cout << GRAY << " ┌──────────────────────────────────────┬─────────────┬───────────┐\n" << RESET;
        std::cout << " │ " << BOLD << std::left << std::setw(36) << "Stage / Kernel" << RESET << " │ " << BOLD
                  << std::setw(11) << "Time (s)" << RESET << " │ " << BOLD << std::setw(9) << "Share (%)" << RESET
                  << " │\n";
        std::cout << GRAY << " ├──────────────────────────────────────┼─────────────┼───────────┤\n" << RESET;

        for (std::size_t i = 0; i < static_cast<std::size_t>(pico::perf::Stage::Count); ++i)
        {
            const auto   stage = static_cast<pico::perf::Stage>(i);
            const double t     = profiler.seconds(stage);
            const double pct   = (profiled_time > 0.0) ? (t / profiled_time) * 100.0 : 0.0;

            std::cout << " │ " << CYAN << std::left << std::setw(36) << pico::perf::STAGE_NAMES[i] << RESET << " │ "
                      << YELLOW << std::right << std::setw(11) << pico::perf::PipelineProfiler::format_time(t) << RESET
                      << " │ " << GREEN << std::right << std::setw(8) << std::fixed << std::setprecision(1) << pct
                      << "%" << RESET << " │\n";
        }
        std::cout << GRAY << " └──────────────────────────────────────┴─────────────┴───────────┘\n\n" << RESET;
    }

private:
    void print_banner(const IEngine& engine) const
    {
        using namespace pico::ui;
        std::cout << CYAN << BOLD << R"(
  ____  ___ ____ ___  
 |  _ \|_ _/ ___/ _ \ 
 | |_) || | |  | | | |
 |  __/ | | |__| |_| |
 |_|   |___\____\___/ 
)" << RESET << GRAY
                  << " High-Performance Particle-in-Cell Engine\n"
                  << RESET;

        std::cout << GRAY
                  << "┌── Simulation Setup ───────────────────────────────────────────────────────────────────┐\n"
                  << RESET;
        std::cout << "│ " << BOLD << "Grid Cells:" << RESET << " " << std::setw(8) << engine.grid_cells() << " │ "
                  << BOLD << "Time Step (dt):" << RESET << " " << std::setw(8) << dt_ << " │ " << BOLD
                  << "Target Time:" << RESET << " " << std::setw(8) << (nsteps_ * dt_) << " │\n";
        std::cout << "│ " << BOLD << "Active Parts:" << RESET << " " << std::setw(8) << engine.total_particles()
                  << " │ " << BOLD << "Total Steps:" << RESET << " " << std::setw(8) << nsteps_ << " │ " << BOLD
                  << "Log Every:" << RESET << " " << std::setw(8) << log_interval_ << " │\n";
        std::cout << GRAY
                  << "└───────────────────────────────────────────────────────────────────────────────────────┘\n\n"
                  << RESET;
    }

    void print_header() const
    {
        using namespace pico::ui;
        std::cout << BOLD << WHITE << std::left << std::setw(8) << "Step" << std::setw(12) << "Sim Time"
                  << std::setw(14) << "Step Lat" << std::setw(16) << "Throughput" << std::setw(20) << "Progress"
                  << std::setw(10) << "ETA" << RESET << "\n";
        std::cout << GRAY
                  << "----------------------------------------------------------------------------------------\n"
                  << RESET;
    }

    double            dt_{0.0};
    int               nsteps_{0};
    int               log_interval_{1};
    clock::time_point start_time_;
    clock::time_point last_interval_time_;
};

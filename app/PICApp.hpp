#pragma once

#include "IEngine.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

// ANSI Terminal Colors
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

class PICApp
{
public:
    explicit PICApp(std::unique_ptr<IEngine> engine, double dt, int log_interval = 50)
            : engine_(std::move(engine)), dt_(dt), log_interval_(log_interval)
    {
    }

    void run(int nsteps)
    {
        using clock = std::chrono::high_resolution_clock;

        print_banner(nsteps);
        print_table_header();

        const auto start_wall_time    = clock::now();
        auto       last_interval_time = start_wall_time;

        for (int step = 0; step < nsteps; ++step)
        {
            if (step % log_interval_ == 0 && step > 0)
            {
                const auto   now           = clock::now();
                const double interval_sec  = std::chrono::duration<double>(now - last_interval_time).count();
                const double total_elapsed = std::chrono::duration<double>(now - start_wall_time).count();

                log_step_status(step, nsteps, interval_sec, total_elapsed);
                last_interval_time = now;
            }

            engine_->advance(dt_);
        }

        const auto   end_wall_time = clock::now();
        const double total_sec     = std::chrono::duration<double>(end_wall_time - start_wall_time).count();

        // Print final step status
        log_step_status(nsteps, nsteps, 0.0, total_sec);

        std::cout << pico::ui::GRAY
                  << "----------------------------------------------------------------------------------------\n"
                  << pico::ui::RESET;
        std::cout << pico::ui::GREEN << pico::ui::BOLD << " ✔ Simulation finished in " << std::fixed
                  << std::setprecision(2) << total_sec << " seconds.\n\n"
                  << pico::ui::RESET;

        engine_->print_perf_report();
    }

private:
    void print_banner(int nsteps) const
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
        std::cout << "│ " << BOLD << "Grid Cells:" << RESET << " " << std::setw(8) << engine_->grid_cells() << " │ "
                  << BOLD << "Time Step (dt):" << RESET << " " << std::setw(8) << dt_ << " │ " << BOLD
                  << "Target Time:" << RESET << " " << std::setw(8) << (nsteps * dt_) << " │\n";
        std::cout << "│ " << BOLD << "Active Parts:" << RESET << " " << std::setw(8) << engine_->total_particles()
                  << " │ " << BOLD << "Total Steps:" << RESET << " " << std::setw(8) << nsteps << " │ " << BOLD
                  << "Log Every:" << RESET << " " << std::setw(8) << log_interval_ << " │\n";
        std::cout << GRAY
                  << "└───────────────────────────────────────────────────────────────────────────────────────┘\n\n"
                  << RESET;
    }

    void print_table_header() const
    {
        using namespace pico::ui;
        std::cout << BOLD << WHITE << std::left << std::setw(10) << "Step" << std::setw(12) << "Sim Time"
                  << std::setw(14) << "Avg Step Lat" << std::setw(16) << "Throughput" << std::setw(22) << "Progress"
                  << std::setw(10) << "ETA" << RESET << "\n";
        std::cout << GRAY
                  << "----------------------------------------------------------------------------------------\n"
                  << RESET;
    }

    void log_step_status(int step, int total_steps, double interval_sec, double total_elapsed_sec) const
    {
        using namespace pico::ui;

        const double progress = static_cast<double>(step) / total_steps;
        const double sim_time = step * dt_;

        // Performance metrics
        const double avg_step_ms = (step > 0 && interval_sec > 0.0) ? (interval_sec / log_interval_) * 1000.0 : 0.0;

        const double total_part_steps = static_cast<double>(engine_->total_particles()) * log_interval_;
        const double mpsp             = (interval_sec > 0.0) ? (total_part_steps / interval_sec) / 1e6 : 0.0;

        // Remaining time estimate
        const double eta_sec =
                (progress > 0.0 && progress < 1.0) ? (total_elapsed_sec / progress) * (1.0 - progress) : 0.0;

        // Progress bar rendering
        constexpr int bar_width  = 12;
        const int     filled_len = static_cast<int>(std::round(bar_width * progress));
        std::string   bar        = "[";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled_len)
                bar += "█";
            else
                bar += "░";
        }
        bar += "]";

        std::cout << std::left << CYAN << std::setw(10) << step << RESET << std::setw(12) << std::fixed
                  << std::setprecision(4) << sim_time << YELLOW << std::setw(10) << std::fixed << std::setprecision(2)
                  << avg_step_ms << " ms  " << RESET << GREEN << std::setw(10) << std::fixed << std::setprecision(1)
                  << mpsp << " MPSP " << RESET << MAGENTA << bar << " " << std::setw(4)
                  << static_cast<int>(progress * 100.0) << "% " << RESET << GRAY << std::setw(8) << std::fixed
                  << std::setprecision(1) << eta_sec << "s" << RESET << "\n";
    }

    std::unique_ptr<IEngine> engine_;
    double                   dt_;
    int                      log_interval_;
};

#pragma once

#include "Colors.hpp"
#include "IEngine.hpp"
#include "engine/perf/PipelineProfiler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

class SimMonitor
{
public:
    // Uses steady_clock to prevent time jumps from NTP updates
    using clock = std::chrono::steady_clock;

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
        const double mpsp        = (interval_sec > 0.0) ? ((engine.total_particles() * log_interval_) / interval_sec) / 1e6 : 0.0;
        const double eta_sec     = (progress > 0.0 && progress < 1.0) ? (total_sec / progress) * (1.0 - progress) : 0.0;
        const double stride      = engine.mean_cell_stride();

        // Zero-allocation stack buffers for string alignment
        char lat_buf[32], mpsp_buf[32], stride_buf[32];
        std::snprintf(lat_buf, sizeof(lat_buf), "%.2f ms", avg_step_ms);
        std::snprintf(mpsp_buf, sizeof(mpsp_buf), "%.1f MPSP", mpsp);

        if (stride > 0.0)
            std::snprintf(stride_buf, sizeof(stride_buf), "%.2f ΔC", stride);
        else
            std::snprintf(stride_buf, sizeof(stride_buf), "N/A");

        constexpr int bar_width = 10;
        const int     filled    = std::clamp(static_cast<int>(std::round(bar_width * progress)), 0, bar_width);

        using namespace pico::ui;
        std::cout << CYAN << std::left << std::setw(8) << step << RESET << std::left << std::setw(12) << std::fixed << std::setprecision(4) << (step * dt_) << YELLOW << std::left
                  << std::setw(14) << lat_buf << RESET << GREEN << std::left << std::setw(16) << mpsp_buf << RESET << BLUE << std::left << std::setw(12) << stride_buf << RESET
                  << MAGENTA << '[';

        // Zero-allocation progress bar output
        for (int i = 0; i < bar_width; ++i)
            std::cout << (i < filled ? "█" : "░");

        std::cout << "] " << std::right << std::setw(3) << static_cast<int>(progress * 100.0) << "%   " << RESET << GRAY << std::left << std::setw(10) << format_duration(eta_sec)
                  << RESET << "\n";
    }

    void print_final_report(const IEngine& engine) const
    {
        using namespace pico::ui;
        const double   total_wall_sec   = std::chrono::duration<double>(clock::now() - start_time_).count();
        const uint64_t total_part_steps = static_cast<uint64_t>(engine.total_particles()) * nsteps_;

        std::cout << GRAY << "-----------------------------------------------------------------------------------------------------\n" << RESET;
        std::cout << GREEN << BOLD << " ✔ Simulation finished in " << std::fixed << std::setprecision(2) << total_wall_sec << " seconds.\n\n" << RESET;

        const auto&  profiler      = engine.profiler();
        const double profiled_time = profiler.total_seconds();

        if (profiled_time > 0.0)
        {
            std::cout << BOLD << WHITE << " Pipeline Kernel Execution Breakdown:\n" << RESET;
            std::cout << GRAY << " ┌──────────────────────────────────────┬──────────────┬───────────┬────────────────┐\n" << RESET;
            std::cout << " │ " << BOLD << std::left << std::setw(36) << "Stage / Kernel" << RESET << " │ " << BOLD << std::setw(12) << "Wall Time" << RESET << " │ " << BOLD
                      << std::setw(9) << "Share" << RESET << " │ " << BOLD << std::setw(14) << "Cost / Part" << RESET << " │\n";
            std::cout << GRAY << " ├──────────────────────────────────────┼──────────────┼───────────┼────────────────┤\n" << RESET;

            for (std::size_t i = 0; i < static_cast<std::size_t>(pico::perf::Stage::Count); ++i)
            {
                const auto   stage = static_cast<pico::perf::Stage>(i);
                const double t     = profiler.seconds(stage);

                if (t <= 0.0)
                    continue;

                const double pct         = (t / profiled_time) * 100.0;
                const double ns_per_part = (total_part_steps > 0) ? (t * 1e9) / total_part_steps : 0.0;

                char pct_buf[16];
                std::snprintf(pct_buf, sizeof(pct_buf), "%.1f %%", pct);

                const std::string time_str = pico::perf::PipelineProfiler::format_time(t);
                const std::string ns_str   = pico::perf::PipelineProfiler::format_ns(ns_per_part);

                std::cout << " │ " << CYAN << std::left << std::setw(36) << pico::perf::STAGE_NAMES[i] << RESET << " │ " << YELLOW << std::right << std::setw(12) << time_str
                          << RESET << " │ " << GREEN << std::right << std::setw(9) << pct_buf << RESET << " │ " << MAGENTA << std::right << std::setw(14) << ns_str << RESET
                          << " │\n";
            }
            std::cout << GRAY << " └──────────────────────────────────────┴──────────────┴───────────┴────────────────┘\n\n" << RESET;
        }
        else
        {
            std::cout << GRAY << " Pipeline Kernel Execution Breakdown : Disabled\n\n" << RESET;
        }

        if (total_part_steps > 0)
        {
            const double avg_mpsp        = (total_part_steps / total_wall_sec) / 1e6;
            const double overall_ns_part = (total_wall_sec * 1e9) / total_part_steps;
            const double final_stride    = engine.mean_cell_stride();

            std::cout << BOLD << WHITE << " Particle Performance Summary:\n" << RESET;
            std::cout << "   • Total Particle Updates : " << total_part_steps << "\n";
            std::cout << "   • Average Throughput     : " << GREEN << std::fixed << std::setprecision(2) << avg_mpsp << " MPSP" << RESET << "\n";
            std::cout << "   • Total Cost / Particle  : " << MAGENTA << std::fixed << std::setprecision(2) << overall_ns_part << " ns/particle-step" << RESET << "\n";

            std::cout << "   • Spatial Stride (ΔC)    : ";
            if (final_stride > 0.0)
                std::cout << BLUE << std::fixed << std::setprecision(3) << final_stride << " cells/pair" << RESET << "\n\n";
            else
                std::cout << GRAY << "N/A (Diagnostics Disabled)" << RESET << "\n\n";
        }
    }

private:
    static std::string format_duration(double seconds)
    {
        if (seconds <= 0.0)
            return "0.0s";
        if (seconds < 60.0)
        {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.1fs", seconds);
            return std::string(buf);
        }
        const int total_sec = static_cast<int>(seconds);
        const int mins      = total_sec / 60;
        const int secs      = total_sec % 60;

        char buf[16];
        std::snprintf(buf, sizeof(buf), "%dm %02ds", mins, secs);
        return std::string(buf);
    }

    void print_banner(const IEngine& engine) const
    {
        using namespace pico::ui;
        std::cout << BOLD << CYAN << "  ██████╗   ██╗  ██████╗   ██████╗ \n"
                  << "  ██╔══██╗  ██║ ██╔════╝  ██╔═══██╗\n"
                  << "  ██████╔╝  ██║ ██║       ██║   ██║\n"
                  << "  ██╔═══╝   ██║ ██║       ██║   ██║\n"
                  << "  ██║       ██║ ╚██████╗  ╚██████╔╝\n"
                  << "  ╚═╝       ╚═╝  ╚═════╝   ╚═════╝ \n"
                  << RESET << GRAY << "  High-Performance Particle-in-Cell Engine\n\n"
                  << RESET;

        // Top border (84 characters wide)
        std::cout << GRAY << "┌── Simulation Setup ─────────────────────────────────────────────────────────┐\n" << RESET;

        // Row 1
        std::cout << "│ " << BOLD << std::left << std::setw(14) << "Grid Cells:" << RESET << std::right << std::setw(10) << engine.grid_cells() << " │ " << BOLD << std::left
                  << std::setw(16) << "Time Step (dt):" << RESET << std::right << std::setw(10) << dt_ << " │ " << BOLD << std::left << std::setw(14) << "Target Time:" << RESET
                  << std::right << std::setw(5) << (nsteps_ * dt_) << " │\n";

        // Row 2
        std::cout << "│ " << BOLD << std::left << std::setw(14) << "Active Parts:" << RESET << std::right << std::setw(10) << engine.total_particles() << " │ " << BOLD << std::left
                  << std::setw(16) << "Total Steps:" << RESET << std::right << std::setw(10) << nsteps_ << " │ " << BOLD << std::left << std::setw(14) << "Log Every:" << RESET
                  << std::right << std::setw(5) << log_interval_ << " │\n";

        // Bottom border (84 characters wide)
        std::cout << GRAY << "└─────────────────────────────────────────────────────────────────────────────┘\n\n" << RESET;
    }

    void print_header() const
    {
        using namespace pico::ui;
        std::cout << BOLD << WHITE << std::left << std::setw(8) << "Step" << std::setw(12) << "Sim Time" << std::setw(14) << "Step Lat" << std::setw(16) << "Throughput"
                  << std::setw(12) << "Locality" << std::setw(20) << "Progress" << std::setw(10) << "ETA" << RESET << "\n";
        std::cout << GRAY << "-----------------------------------------------------------------------------------------------------\n" << RESET;
    }

    double            dt_{0.0};
    int               nsteps_{0};
    int               log_interval_{1};
    clock::time_point start_time_;
    clock::time_point last_interval_time_;
};
#pragma once

#include "Colors.hpp"
#include "IEngine.hpp"
#include "engine/perf/PipelineProfiler.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

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
        const double mpsp        = (interval_sec > 0.0) ? ((engine.total_particles() * log_interval_) / interval_sec) / 1e6 : 0.0;
        const double eta_sec     = (progress > 0.0 && progress < 1.0) ? (total_sec / progress) * (1.0 - progress) : 0.0;
        const double stride      = engine.mean_cell_stride();

        constexpr int bar_width = 10;
        const int     filled    = static_cast<int>(std::round(bar_width * progress));
        std::string   bar       = "[";
        for (int i = 0; i < bar_width; ++i)
            bar += (i < filled ? "█" : "░");
        bar += "]";

        std::ostringstream lat_ss, mpsp_ss, stride_ss, pct_ss;
        lat_ss << std::fixed << std::setprecision(2) << avg_step_ms << " ms";
        mpsp_ss << std::fixed << std::setprecision(1) << mpsp << " MPSP";
        stride_ss << std::fixed << std::setprecision(2) << stride << " ΔC";
        pct_ss << std::right << std::setw(3) << static_cast<int>(progress * 100.0) << "%";

        using namespace pico::ui;
        std::cout << CYAN << std::left << std::setw(8) << step << RESET << std::left << std::setw(12) << std::fixed << std::setprecision(4) << (step * dt_) << YELLOW << std::left
                  << std::setw(14) << lat_ss.str() << RESET << GREEN << std::left << std::setw(16) << mpsp_ss.str() << RESET << BLUE << std::left << std::setw(12)
                  << stride_ss.str() << RESET << MAGENTA << bar << " " << pct_ss.str() << "   " << RESET << GRAY << std::left << std::setw(10) << format_duration(eta_sec) << RESET
                  << "\n";
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

        std::cout << BOLD << WHITE << " Pipeline Kernel Execution Breakdown:\n" << RESET;
        std::cout << GRAY << " ┌──────────────────────────────────────┬──────────────┬───────────┬────────────────┐\n" << RESET;
        std::cout << " │ " << BOLD << std::left << std::setw(36) << "Stage / Kernel" << RESET << " │ " << BOLD << std::setw(12) << "Wall Time" << RESET << " │ " << BOLD
                  << std::setw(9) << "Share" << RESET << " │ " << BOLD << std::setw(14) << "Cost / Part" << RESET << " │\n";
        std::cout << GRAY << " ├──────────────────────────────────────┼──────────────┼───────────┼────────────────┤\n" << RESET;

        for (std::size_t i = 0; i < static_cast<std::size_t>(pico::perf::Stage::Count); ++i)
        {
            const auto   stage       = static_cast<pico::perf::Stage>(i);
            const double t           = profiler.seconds(stage);
            const double pct         = (profiled_time > 0.0) ? (t / profiled_time) * 100.0 : 0.0;
            const double ns_per_part = (total_part_steps > 0) ? (t * 1e9) / total_part_steps : 0.0;

            std::ostringstream time_ss, pct_ss, ns_ss;
            time_ss << pico::perf::PipelineProfiler::format_time(t);
            pct_ss << std::fixed << std::setprecision(1) << pct << " %";
            ns_ss << std::fixed << std::setprecision(2) << ns_per_part << " ns";

            std::cout << " │ " << CYAN << std::left << std::setw(36) << pico::perf::STAGE_NAMES[i] << RESET << " │ " << YELLOW << std::right << std::setw(12) << time_ss.str()
                      << RESET << " │ " << GREEN << std::right << std::setw(9) << pct_ss.str() << RESET << " │ " << MAGENTA << std::right << std::setw(14) << ns_ss.str() << RESET
                      << " │\n";
        }
        std::cout << GRAY << " └──────────────────────────────────────┴──────────────┴───────────┴────────────────┘\n\n" << RESET;

        if (total_part_steps > 0)
        {
            const double avg_mpsp        = (total_part_steps / total_wall_sec) / 1e6;
            const double overall_ns_part = (total_wall_sec * 1e9) / total_part_steps;
            const double final_stride    = engine.mean_cell_stride();

            std::cout << BOLD << WHITE << " Particle Performance Summary:\n" << RESET;
            std::cout << "   • Total Particle Updates : " << total_part_steps << "\n";
            std::cout << "   • Average Throughput     : " << GREEN << std::fixed << std::setprecision(2) << avg_mpsp << " MPSP" << RESET << "\n";
            std::cout << "   • Total Cost / Particle  : " << MAGENTA << std::fixed << std::setprecision(2) << overall_ns_part << " ns/particle-step" << RESET << "\n";
            std::cout << "   • Spatial Stride (ΔC)    : " << BLUE << std::fixed << std::setprecision(3) << final_stride << " cells/pair" << RESET << "\n\n";
        }
    }

private:
    static std::string format_duration(double seconds)
    {
        if (seconds <= 0.0)
            return "0.0s";
        if (seconds < 60.0)
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << seconds << "s";
            return ss.str();
        }
        const int total_sec = static_cast<int>(seconds);
        const int mins      = total_sec / 60;
        const int secs      = total_sec % 60;

        std::ostringstream ss;
        ss << mins << "m " << std::setw(2) << std::setfill('0') << secs << "s";
        return ss.str();
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

        std::cout << GRAY << "┌── Simulation Setup ───────────────────────────────────────────────────────────────────┐\n" << RESET;
        std::cout << "│ " << BOLD << "Grid Cells:" << RESET << " " << std::setw(8) << engine.grid_cells() << " │ " << BOLD << "Time Step (dt):" << RESET << " " << std::setw(8)
                  << dt_ << " │ " << BOLD << "Target Time:" << RESET << " " << std::setw(8) << (nsteps_ * dt_) << " │\n";
        std::cout << "│ " << BOLD << "Active Parts:" << RESET << " " << std::setw(8) << engine.total_particles() << " │ " << BOLD << "Total Steps:" << RESET << " " << std::setw(8)
                  << nsteps_ << " │ " << BOLD << "Log Every:" << RESET << " " << std::setw(8) << log_interval_ << " │\n";
        std::cout << GRAY << "└───────────────────────────────────────────────────────────────────────────────────────┘\n\n" << RESET;
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
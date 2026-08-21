#pragma once

#include "IEngine.hpp"
#include "app/SimMonitor.hpp"

#include <memory>
#include <utility>

class PICApp
{
public:
    explicit PICApp(std::unique_ptr<IEngine> engine, double dt, int log_interval = 50)
            : engine_(std::move(engine)), dt_(dt), log_interval_(log_interval)
    {
    }

    void run(int nsteps)
    {
        monitor_.start(*engine_, dt_, nsteps, log_interval_);

        for (int step = 0; step < nsteps; ++step)
        {
            if (step % log_interval_ == 0 && step > 0)
            {
                monitor_.log_step(step, *engine_);
            }

            engine_->advance(dt_);
        }

        monitor_.log_step(nsteps, *engine_);
        monitor_.print_final_report(*engine_);
    }

private:
    std::unique_ptr<IEngine> engine_;
    double                   dt_;
    int                      log_interval_;
    SimMonitor               monitor_;
};

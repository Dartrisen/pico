#pragma once
#include <memory>
#include "IEngine.hpp"

class PICApp
{
public:
    explicit PICApp(std::unique_ptr<IEngine> engine, double dt)
        : engine_(std::move(engine)), dt_(dt) {}

    void run(int nsteps)
    {
        for (int i = 0; i < nsteps; ++i)
        {
            engine_->advance(dt_);
            // diagnostics, IO, checkpoints
        }
    }

private:
    std::unique_ptr<IEngine> engine_;
    double dt_;
};

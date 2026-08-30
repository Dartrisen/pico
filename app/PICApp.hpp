#pragma once

#include "IEngine.hpp"
#include "app/SimMonitor.hpp"

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

class PICApp
{
public:
    explicit PICApp(std::unique_ptr<IEngine> engine, double dt, int log_interval = 50) : engine_(std::move(engine)), dt_(dt), log_interval_(log_interval) {}

    // Compile-time dispatched callback:
    //   fn(step)
    //   fn(step, IEngine&)
    template <typename Callback = std::nullptr_t>
    void run(int nsteps, Callback&& on_step = nullptr)
    {
        monitor_.start(*engine_, dt_, nsteps, log_interval_);

        for (int step = 0; step < nsteps; ++step)
        {
            invoke_step_callback(on_step, step);

            if (step % log_interval_ == 0 && step > 0)
            {
                monitor_.log_step(step, *engine_);
            }

            engine_->advance(dt_);
        }

        // Final callback at exactly nsteps.
        invoke_step_callback(on_step, nsteps);

        monitor_.log_step(nsteps, *engine_);
        monitor_.print_final_report(*engine_);
    }

    [[nodiscard]] IEngine& engine() noexcept
    {
        return *engine_;
    }

    [[nodiscard]] const IEngine& engine() const noexcept
    {
        return *engine_;
    }

private:
    template <typename Callback>
    void invoke_step_callback(Callback&& callback, int step)
    {
        using CallbackType = std::remove_cvref_t<Callback>;

        if constexpr (std::same_as<CallbackType, std::nullptr_t>)
        {
            return;
        }
        else if constexpr (std::invocable<Callback&, int, IEngine&>)
        {
            std::forward<Callback>(callback)(step, *engine_);
        }
        else if constexpr (std::invocable<Callback&, int>)
        {
            std::forward<Callback>(callback)(step);
        }
        else
        {
            static_assert(std::invocable<Callback&, int, IEngine&> || std::invocable<Callback&, int>, "Step callback must be callable with (int) or (int, IEngine&)");
        }
    }

    std::unique_ptr<IEngine> engine_;
    double                   dt_;
    int                      log_interval_;
    SimMonitor               monitor_;
};

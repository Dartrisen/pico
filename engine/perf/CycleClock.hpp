#pragma once

#include <chrono>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

namespace pico::perf
{

/**
 * @brief Reads the hardware cycle/system counter with minimal overhead.
 */
[[nodiscard]] inline std::uint64_t read_cpu_ticks() noexcept
{
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM64 (Apple Silicon M1/M2/M3/M4 & Linux AArch64)
    std::uint64_t ticks;
    asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));
    return ticks;
#elif defined(__x86_64__) || defined(_M_X64)
    // 64-bit x86 Linux / macOS
    return __rdtsc();
#else
    // Portable fallback for non-64-bit targets
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

/**
 * @brief Returns the frequency multiplier to convert ARM64/x86_64 hardware ticks to nanoseconds.
 */
[[nodiscard]] inline double ticks_to_nanoseconds_scale() noexcept
{
#if defined(__aarch64__) || defined(_M_ARM64)
    std::uint64_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return 1e9 / static_cast<double>(freq);
#else
    // Standard 1:1 fallback for calibrated ticks or relative measurement
    return 1.0;
#endif
}

} // namespace pico::perf
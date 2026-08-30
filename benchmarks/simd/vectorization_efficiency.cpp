#include "app/VerificationReport.hpp"
#include "engine/perf/PipelineProfiler.hpp"

#include <cmath>
#include <vector>

namespace benchmarks
{

/**
 * @brief Measure instruction-level parallelism (ILP)
 */
void bench_instruction_parallelism()
{
    pico::ui::VerificationReport report("Instruction-Level Parallelism (ILP)", true, "Dependent vs. Independent Operations");
    constexpr size_t             num_elements = 100000;
    std::vector<float>           a(num_elements), b(num_elements), c(num_elements);

    for (size_t i = 0; i < num_elements; ++i)
    {
        a[i] = 1.0f + static_cast<float>(i % 128) / 128.0f;
        b[i] = 2.0f + static_cast<float>(i % 64) / 64.0f;
        c[i] = 3.0f + static_cast<float>(i % 32) / 32.0f;
    }

    pico::perf::PipelineProfiler profiler;
    constexpr size_t             iterations = 100;

    // Test 1: Dependent chain
    profiler.reset();
    float result1 = 0.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::FieldSolver);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            result1 = a[0];
            for (size_t i = 1; i < num_elements; ++i)
            {
                result1 = result1 * a[i] + b[i];
            }
        }
        asm volatile("" : "+m"(result1));
    }
    double time1_sec = profiler.seconds(pico::perf::Stage::FieldSolver);
    double gflop1    = (static_cast<double>(num_elements) * iterations * 2) / 1e9;
    double gflops1   = gflop1 / time1_sec;

    // Test 2: Independent accumulators (vectorizable)
    profiler.reset();
    float result2a = 0.0f, result2b = 0.0f, result2c = 0.0f, result2d = 0.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::FieldSolver);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; i += 4)
            {
                if (i + 0 < num_elements)
                    result2a = result2a * a[i + 0] + b[i + 0];
                if (i + 1 < num_elements)
                    result2b = result2b * a[i + 1] + b[i + 1];
                if (i + 2 < num_elements)
                    result2c = result2c * a[i + 2] + b[i + 2];
                if (i + 3 < num_elements)
                    result2d = result2d * a[i + 3] + b[i + 3];
            }
        }
        asm volatile("" : "+m"(result2a), "+m"(result2b), "+m"(result2c), "+m"(result2d));
    }
    double time2_sec = profiler.seconds(pico::perf::Stage::FieldSolver);
    double gflop2    = (static_cast<double>(num_elements) * iterations * 2) / 1e9;
    double gflops2   = gflop2 / time2_sec;
    double speedup   = time1_sec / time2_sec;

    report.add_fixed_row("Dependent Operations", gflops1, 2, "GFLOP/s");
    report.add_fixed_row("Independent Operations", gflops2, 2, "GFLOP/s");
    report.add_fixed_row("ILP Speedup", speedup, 2, "x");
    report.print();
}

/**
 * @brief SIMD-friendly vs unfriendly access patterns
 */
void bench_simd_access_patterns()
{
    pico::ui::VerificationReport report("SIMD Access Patterns", true, "Unit Stride vs. Strided Access Efficiency");
    constexpr size_t             num_elements = 100000;
    std::vector<float>           data(num_elements);
    for (size_t i = 0; i < num_elements; ++i)
        data[i] = 1.0f + static_cast<float>(i) / num_elements;

    pico::perf::PipelineProfiler profiler;
    constexpr size_t             iterations = 1000;

    // Pattern 1: Unit stride (SIMD-friendly)
    profiler.reset();
    float sum1 = 0.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::Gather);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
            {
                sum1 += data[i];
            }
        }
        asm volatile("" : "+m"(sum1));
    }
    double time1_sec = profiler.seconds(pico::perf::Stage::Gather);
    double elems1_m  = (static_cast<double>(num_elements) * iterations) / time1_sec / 1e6;

    // Pattern 2: Strided access
    profiler.reset();
    float sum2 = 0.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::Gather);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; i += 8)
            {
                sum2 += data[i];
            }
        }
        asm volatile("" : "+m"(sum2));
    }
    double time2_sec = profiler.seconds(pico::perf::Stage::Gather);
    double ops2      = static_cast<double>(num_elements / 8) * iterations;
    double elems2_m  = ops2 / time2_sec / 1e6;

    report.add_fixed_row("Unit Stride (Vectorizable)", elems1_m, 2, "M elem/s");
    report.add_fixed_row("Strided Access (Stride=8)", elems2_m, 2, "M elem/s");
    report.print();
}

/**
 * @brief Branch impact on vectorization
 */
void bench_branch_impact()
{
    pico::ui::VerificationReport report("Branch Impact on Performance", true, "Predictable vs. Unpredictable Control Flow");
    constexpr size_t             num_elements = 100000;
    std::vector<float>           data(num_elements);
    for (size_t i = 0; i < num_elements; ++i)
        data[i] = static_cast<float>(i % 256);

    pico::perf::PipelineProfiler profiler;
    constexpr size_t             iterations = 1000;

    // Scenario 1: Predictable branch
    profiler.reset();
    float sum1 = 0.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::Deposit);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
            {
                if (data[i] >= 0.0f)
                    sum1 += data[i];
            }
        }
        asm volatile("" : "+m"(sum1));
    }
    double time1_sec = profiler.seconds(pico::perf::Stage::Deposit);

    // Scenario 2: Unpredictable branch
    profiler.reset();
    float sum2 = 0.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::Deposit);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
            {
                if (static_cast<int>(data[i]) & 1)
                    sum2 += data[i];
            }
        }
        asm volatile("" : "+m"(sum2));
    }
    double time2_sec = profiler.seconds(pico::perf::Stage::Deposit);
    double penalty   = time2_sec / time1_sec;

    report.add_fixed_row("Predictable Branch Time", time1_sec * 1000.0, 3, "ms");
    report.add_fixed_row("Unpredictable Branch Time", time2_sec * 1000.0, 3, "ms");
    report.add_fixed_row("Branch Penalty Ratio", penalty, 2, "x");
    report.print();
}

/**
 * @brief Floating-point operation efficiency mix
 */
void bench_flop_mix()
{
    pico::ui::VerificationReport report("Floating-Point Operation Mix", true, "Throughput by Instruction Type");
    constexpr size_t             num_elements = 100000;
    constexpr size_t             iterations   = 100;

    std::vector<float> a(num_elements), b(num_elements);
    for (size_t i = 0; i < num_elements; ++i)
    {
        a[i] = 1.0f + static_cast<float>(i % 128) / 128.0f;
        b[i] = 2.0f + static_cast<float>(i % 64) / 64.0f;
    }

    pico::perf::PipelineProfiler profiler;

    // Addition
    profiler.reset();
    float sum = 0.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::Pusher);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
                sum = a[i] + b[i];
        }
        asm volatile("" : "+m"(sum));
    }
    double time_add   = profiler.seconds(pico::perf::Stage::Pusher);
    double gflops_add = (static_cast<double>(num_elements) * iterations / 1e9) / time_add;

    // Multiplication
    profiler.reset();
    float prod = 1.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::Pusher);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
                prod = a[i] * b[i];
        }
        asm volatile("" : "+m"(prod));
    }
    double time_mul   = profiler.seconds(pico::perf::Stage::Pusher);
    double gflops_mul = (static_cast<double>(num_elements) * iterations / 1e9) / time_mul;

    // FMA
    profiler.reset();
    float result = 0.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::Pusher);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
                result = a[i] * b[i] + a[i];
        }
        asm volatile("" : "+m"(result));
    }
    double time_fma   = profiler.seconds(pico::perf::Stage::Pusher);
    double gflops_fma = (static_cast<double>(num_elements) * iterations * 2 / 1e9) / time_fma;

    // Division
    profiler.reset();
    float quot = 1.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::Pusher);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
                quot = a[i] / (b[i] + 1e-6f);
        }
        asm volatile("" : "+m"(quot));
    }
    double time_div   = profiler.seconds(pico::perf::Stage::Pusher);
    double gflops_div = (static_cast<double>(num_elements) * iterations / 1e9) / time_div;

    // Square root
    profiler.reset();
    float root = 1.0f;
    {
        auto scope = profiler.time_stage(pico::perf::Stage::Pusher);
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
                root = std::sqrt(a[i] * b[i] + 1e-6f);
        }
        asm volatile("" : "+m"(root));
    }
    double time_sqrt   = profiler.seconds(pico::perf::Stage::Pusher);
    double gflops_sqrt = (static_cast<double>(num_elements) * iterations / 1e9) / time_sqrt;

    report.add_fixed_row("Addition", gflops_add, 2, "GFLOP/s");
    report.add_fixed_row("Multiplication", gflops_mul, 2, "GFLOP/s");
    report.add_fixed_row("FMA (Mul + Add)", gflops_fma, 2, "GFLOP/s");
    report.add_fixed_row("Division", gflops_div, 2, "GFLOP/s");
    report.add_fixed_row("Square Root", gflops_sqrt, 2, "GFLOP/s");
    report.print();
}

} // namespace benchmarks

int main()
{
    benchmarks::bench_instruction_parallelism();
    benchmarks::bench_simd_access_patterns();
    benchmarks::bench_branch_impact();
    benchmarks::bench_flop_mix();

    return 0;
}
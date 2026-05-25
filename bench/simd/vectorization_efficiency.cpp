#include "bench/core/timer.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

    /**
     * @brief Measure instruction-level parallelism
     *
     * Test 1: Dependent operations (true data dependency)
     *   result = a * b + c (each operation depends on previous)
     *   Cannot parallelize, exposes latency
     *
     * Test 2: Independent operations (no dependency)
     *   Multiple accumulations in parallel
     *   Can be auto-vectorized
     */
    void bench_instruction_parallelism()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "INSTRUCTION-LEVEL PARALLELISM (ILP) ANALYSIS\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t   num_elements = 100000;
        std::vector<float> a(num_elements), b(num_elements), c(num_elements);

        for (size_t i = 0; i < num_elements; ++i)
        {
            a[i] = 1.0f + static_cast<float>(i % 128) / 128.0f;
            b[i] = 2.0f + static_cast<float>(i % 64) / 64.0f;
            c[i] = 3.0f + static_cast<float>(i % 32) / 32.0f;
        }

        // Test 1: Dependent chain
        std::cout << "\n1. Dependent Operations (sequential):\n";
        bench::Timer     timer1;
        volatile float   result1    = 0.0f;
        constexpr size_t iterations = 100;
        for (size_t iter = 0; iter < iterations; ++iter)
        {
            result1 = a[0];
            for (size_t i = 1; i < num_elements; ++i)
            {
                result1 = result1 * a[i] + b[i]; // Each iteration depends on previous
            }
        }
        double time1_ns = timer1.elapsed_ns();
        double gflop1   = (num_elements * iterations * 2 * 1e-9);
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << (time1_ns / 1e9) << " sec\n";
        std::cout << "  GFLOP/s: " << gflop1 / (time1_ns / 1e9) << "\n";

        // Test 2: Independent accumulators (vectorizable)
        std::cout << "\n2. Independent Operations (parallel):\n";
        bench::Timer   timer2;
        volatile float result2a = 0.0f, result2b = 0.0f, result2c = 0.0f, result2d = 0.0f;
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
        double time2_ns = timer2.elapsed_ns();
        double gflop2   = (num_elements * iterations * 2 * 1e-9);
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << (time2_ns / 1e9) << " sec\n";
        std::cout << "  GFLOP/s: " << gflop2 / (time2_ns / 1e9) << "\n";
        std::cout << "  Speedup: " << time1_ns / time2_ns << "x\n";

        (void) result1;
        (void) result2a;
        (void) result2b;
        (void) result2c;
        (void) result2d;
    }

    /**
     * @brief SIMD-friendly vs unfriendly access patterns
     */
    void bench_simd_access_patterns()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "SIMD ACCESS PATTERNS\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t   num_elements = 100000;
        std::vector<float> data(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
            data[i] = 1.0f + static_cast<float>(i) / num_elements;

        // Pattern 1: Unit stride (SIMD-friendly)
        std::cout << "\n1. Unit Stride (vectorizable):\n";
        bench::Timer   timer1;
        volatile float sum1 = 0.0f;
        for (size_t iter = 0; iter < 1000; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
            {
                sum1 += data[i];
            }
        }
        double time1_ns = timer1.elapsed_ns();
        std::cout << "  Throughput: " << std::scientific << std::setprecision(2)
                  << (num_elements * 1000 * 1e9 / time1_ns) << " elem/s\n";

        // Pattern 2: Strided access (poor cache, hard to vectorize)
        std::cout << "\n2. Strided Access (stride=8, problematic):\n";
        bench::Timer   timer2;
        volatile float sum2 = 0.0f;
        for (size_t iter = 0; iter < 1000; ++iter)
        {
            for (size_t i = 0; i < num_elements; i += 8)
            {
                sum2 += data[i];
            }
        }
        double time2_ns = timer2.elapsed_ns();
        double ops2     = (num_elements / 8) * 1000;
        std::cout << "  Throughput: " << std::scientific << std::setprecision(2) << (ops2 * 1e9 / time2_ns)
                  << " elem/s\n";

        (void) sum1;
        (void) sum2;
    }

    /**
     * @brief Branch impact on vectorization
     *
     * Branches prevent SIMD (divergence problem)
     */
    void bench_branch_impact()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BRANCH IMPACT ON PERFORMANCE\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t   num_elements = 100000;
        std::vector<float> data(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
            data[i] = static_cast<float>(i % 256);

        // Scenario 1: Predictable branch (always true or always false)
        std::cout << "\n1. Predictable Branch:\n";
        bench::Timer   timer1;
        volatile float sum1 = 0.0f;
        for (size_t iter = 0; iter < 1000; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
            {
                if (data[i] >= 0.0f) // Always true
                    sum1 += data[i];
            }
        }
        double time1_ns = timer1.elapsed_ns();
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << (time1_ns / 1e9) << " sec\n";

        // Scenario 2: Random branch (unpredictable)
        std::cout << "\n2. Unpredictable Branch:\n";
        bench::Timer   timer2;
        volatile float sum2 = 0.0f;
        for (size_t iter = 0; iter < 1000; ++iter)
        {
            for (size_t i = 0; i < num_elements; ++i)
            {
                if (static_cast<int>(data[i]) & 1) // Pseudo-random based on data
                    sum2 += data[i];
            }
        }
        double time2_ns = timer2.elapsed_ns();
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << (time2_ns / 1e9) << " sec\n";

        std::cout << "\n  Penalty: " << std::fixed << std::setprecision(2) << (time2_ns / time1_ns) << "x\n";

        (void) sum1;
        (void) sum2;
    }

    /**
     * @brief Floating-point operation efficiency
     *
     * Different operations have different costs on modern CPUs
     */
    void bench_flop_mix()
    {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "FLOATING-POINT OPERATION MIX\n";
        std::cout << std::string(80, '=') << "\n";

        constexpr size_t num_elements = 100000;
        constexpr size_t iterations   = 100;

        std::vector<float> a(num_elements), b(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
        {
            a[i] = 1.0f + static_cast<float>(i % 128) / 128.0f;
            b[i] = 2.0f + static_cast<float>(i % 64) / 64.0f;
        }

        std::cout << std::left << std::setw(20) << "Operation"
                  << " | " << std::setw(12) << "GFLOP/s"
                  << " | " << std::setw(12) << "Time (ms)\n";
        std::cout << std::string(60, '=') << "\n";

        // Addition
        {
            bench::Timer   timer;
            volatile float sum = 0.0f;
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < num_elements; ++i)
                    sum = a[i] + b[i];
            }
            double time_ns = timer.elapsed_ns();
            double gflop   = (num_elements * iterations * 1e-9) / (time_ns / 1e9);
            std::cout << std::setw(20) << "Addition" << " | " << std::scientific << std::setprecision(2)
                      << std::setw(12) << gflop << " | " << std::fixed << std::setprecision(3) << std::setw(12)
                      << (time_ns / 1e6) << "\n";
            (void) sum;
        }

        // Multiplication
        {
            bench::Timer   timer;
            volatile float prod = 1.0f;
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < num_elements; ++i)
                    prod = a[i] * b[i];
            }
            double time_ns = timer.elapsed_ns();
            double gflop   = (num_elements * iterations * 1e-9) / (time_ns / 1e9);
            std::cout << std::setw(20) << "Multiplication" << " | " << std::scientific << std::setprecision(2)
                      << std::setw(12) << gflop << " | " << std::fixed << std::setprecision(3) << std::setw(12)
                      << (time_ns / 1e6) << "\n";
            (void) prod;
        }

        // FMA (Fused Multiply-Add)
        {
            bench::Timer   timer;
            volatile float result = 0.0f;
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < num_elements; ++i)
                    result = a[i] * b[i] + a[i];
            }
            double time_ns = timer.elapsed_ns();
            double gflop   = (num_elements * iterations * 2 * 1e-9) / (time_ns / 1e9); // 2 ops
            std::cout << std::setw(20) << "FMA (mul+add)" << " | " << std::scientific << std::setprecision(2)
                      << std::setw(12) << gflop << " | " << std::fixed << std::setprecision(3) << std::setw(12)
                      << (time_ns / 1e6) << "\n";
            (void) result;
        }

        // Division
        {
            bench::Timer   timer;
            volatile float quot = 1.0f;
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < num_elements; ++i)
                    quot = a[i] / (b[i] + 1e-6f);
            }
            double time_ns = timer.elapsed_ns();
            double gflop   = (num_elements * iterations * 1e-9) / (time_ns / 1e9);
            std::cout << std::setw(20) << "Division" << " | " << std::scientific << std::setprecision(2)
                      << std::setw(12) << gflop << " | " << std::fixed << std::setprecision(3) << std::setw(12)
                      << (time_ns / 1e6) << "\n";
            (void) quot;
        }

        // Square root
        {
            bench::Timer   timer;
            volatile float root = 1.0f;
            for (size_t iter = 0; iter < iterations; ++iter)
            {
                for (size_t i = 0; i < num_elements; ++i)
                    root = std::sqrt(a[i] * b[i] + 1e-6f);
            }
            double time_ns = timer.elapsed_ns();
            double gflop   = (num_elements * iterations * 1e-9) / (time_ns / 1e9);
            std::cout << std::setw(20) << "Square Root" << " | " << std::scientific << std::setprecision(2)
                      << std::setw(12) << gflop << " | " << std::fixed << std::setprecision(3) << std::setw(12)
                      << (time_ns / 1e6) << "\n";
            (void) root;
        }
    }

} // namespace

int main()
{
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║      SIMD VECTORIZATION EFFICIENCY ANALYSIS            ║\n";
    std::cout << "║  ILP, access patterns, branches, FLOP mix              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    bench_instruction_parallelism();
    bench_simd_access_patterns();
    bench_branch_impact();
    bench_flop_mix();

    return 0;
}

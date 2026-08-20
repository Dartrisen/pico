#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace kernels::shapes
{
    template <int Order>
    struct SplineTraits;

    template <>
    struct SplineTraits<1>
    {
        static constexpr int support = 2;
    };
    template <>
    struct SplineTraits<2>
    {
        static constexpr int support = 3;
    };
    template <>
    struct SplineTraits<3>
    {
        static constexpr int support = 4;
    };

    template <int Order>
    struct SplineShape
    {
        static constexpr int S = SplineTraits<Order>::support;

        static inline void weights(double x, double dx, int& i0, double (&w)[S])
        {
            const double normalized = x / dx;
            const int    i          = static_cast<int>(std::floor(normalized));
            const double frac       = normalized - static_cast<double>(i);

            i0 = i - (S - 1) / 2;

            if constexpr (Order == 1)
            {
                w[0] = 1.0 - frac;
                w[1] = frac;
            }
            else if constexpr (Order == 2)
            {
                const double xm   = 1.0 - frac;
                const double f_05 = frac - 0.5;
                w[0]              = 0.5 * xm * xm;
                w[1]              = 0.75 - (f_05 * f_05);
                w[2]              = 0.5 * frac * frac;
            }
            else if constexpr (Order == 3)
            {
                constexpr double inv6 = 1.0 / 6.0;
                const double     x0   = frac;
                const double     x1   = 1.0 - frac;

                const double x0_2 = x0 * x0;
                const double x0_3 = x0_2 * x0;
                const double x1_2 = x1 * x1;
                const double x1_3 = x1_2 * x1;

                w[0] = inv6 * x1_3;
                w[1] = inv6 * (4.0 - 6.0 * x0_2 + 3.0 * x0_3);
                w[2] = inv6 * (4.0 - 6.0 * x1_2 + 3.0 * x1_3);
                w[3] = inv6 * x0_3;
            }
        }
    };
} // namespace kernels::shapes
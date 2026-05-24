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
        static constexpr int support = 2; // linear / CIC
    };

    template <>
    struct SplineTraits<2>
    {
        static constexpr int support = 3; // quadratic / TSC
    };

    template <>
    struct SplineTraits<3>
    {
        static constexpr int support = 4; // cubic B-spline
    };

    template <int Order>
    struct SplineShape
    {
        static constexpr int S = SplineTraits<Order>::support;

        static inline void weights(double x, double dx, int& i0, double (&w)[S])
        {
            if (dx <= 0.0)
            {
                throw std::invalid_argument("dx must be positive");
            }

            const double normalized = x / dx;
            const int    i          = static_cast<int>(std::floor(normalized));
            double       frac       = normalized - i;

            i0 = i - (S - 1) / 2;

            if constexpr (Order == 1)
            {
                // linear
                w[0] = 1.0 - frac;
                w[1] = frac;
            }
            else if constexpr (Order == 2)
            {
                // quadratic, 3-point support (TSC-like)
                const double xm = 1.0 - frac;
                w[0]            = 0.5 * xm * xm;
                w[1]            = 0.75 - std::pow(frac - 0.5, 2);
                w[2]            = 0.5 * frac * frac;
            }
            else if constexpr (Order == 3)
            {
                // cubic B-spline
                const double x0 = frac;
                const double x1 = 1.0 - frac;
                w[0]            = (1.0 / 6.0) * std::pow(x1, 3);
                w[1]            = (1.0 / 6.0) * (4.0 - 6.0 * frac * frac + 3.0 * std::pow(frac, 3));
                w[2]            = (1.0 / 6.0) * (4.0 - 6.0 * x1 * x1 + 3.0 * std::pow(x1, 3));
                w[3]            = (1.0 / 6.0) * std::pow(x0, 3);
            }
            else
            {
                static_assert(Order >= 1 && Order <= 3, "SplineShape supports Order 1..3");
            }
        }
    };

} // namespace kernels::shapes

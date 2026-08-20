#pragma once

#include "spline.hpp"

namespace kernels::shapes
{
    template <int Order>
    using Shape = SplineShape<Order>;

} // namespace kernels::shapes

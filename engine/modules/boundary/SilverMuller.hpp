#pragma once

#include "data/field/include/field_em.hpp"
#include "data/grid/include/grid.hpp"

namespace pico::modules::boundary
{
    template <std::size_t BLOCK_SIZE>
    struct SilverMullerFieldBoundary
    {
        static void fill_field_guards(EMFields<BLOCK_SIZE>& fields, const Grid& grid)
        {
            const std::size_t G = grid.guard_cells();
            const std::size_t N = grid.physical_size();

            // Left Open Boundary: Ey - c*Bz = 0
            for (std::size_t g = 0; g < G; ++g)
            {
                fields.E.field_y(g) = fields.B.field_z(G); // Radiation condition matching
            }

            // Right Open Boundary: Ey + c*Bz = 0
            for (std::size_t g = 0; g < G; ++g)
            {
                fields.E.field_y(N + G + g) = -fields.B.field_z(N + G - 1);
            }
        }

        static void fold_currents(FieldSystem<BLOCK_SIZE>& J, const Grid& grid)
        {
            // Non-periodic absorbing boundaries do not fold guard currents back;
            // guard currents are zeroed out or absorbed at the wall absorber.
            const std::size_t G = grid.guard_cells();
            const std::size_t N = grid.physical_size();

            for (std::size_t g = 0; g < G; ++g)
            {
                J.field_x(g)         = 0.0f;
                J.field_y(g)         = 0.0f;
                J.field_z(g)         = 0.0f;
                J.field_x(N + G + g) = 0.0f;
                J.field_y(N + G + g) = 0.0f;
                J.field_z(N + G + g) = 0.0f;
            }
        }
    };
} // namespace pico::modules::boundary

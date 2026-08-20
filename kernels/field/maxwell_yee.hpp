#pragma once

#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"

namespace kernels::field
{
    template <std::size_t BLOCK_SIZE>
    struct YeeMaxwell
    {
        static void advance_electric_field(FieldSystem<BLOCK_SIZE>& E, const FieldSystem<BLOCK_SIZE>& B,
                                           const FieldSystem<BLOCK_SIZE>& J, const Grid& grid, float dt)
        {
            const std::size_t G   = grid.guard_cells();
            const std::size_t N   = grid.physical_size();
            const double      dxd = grid.cell_size();

            // Loop over physical domain starting at guard cell offset G
            for (std::size_t i = G; i < G + N; ++i)
            {
                // dBz/dx derivative using staggered grid offset
                float dBz_dx = (B.field_z(i) - B.field_z(i - 1)) / dxd;

                E.field_y(i) += dt * (-dBz_dx - J.field_y(i));
                E.field_x(i) += dt * (-J.field_x(i));
                E.field_z(i) += dt * (-J.field_z(i));
            }
        }

        static void advance_magnetic_field(FieldSystem<BLOCK_SIZE>& B, const FieldSystem<BLOCK_SIZE>& E,
                                           const Grid& grid, float dt)
        {
            const std::size_t G   = grid.guard_cells();
            const std::size_t N   = grid.physical_size();
            const double      dxd = grid.cell_size();

            for (std::size_t i = G; i < G + N; ++i)
            {
                // dEy/dx derivative
                float dEy_dx = (E.field_y(i + 1) - E.field_y(i)) / dxd;

                B.field_z(i) -= dt * dEy_dx;
            }
        }
    };
} // namespace kernels::field
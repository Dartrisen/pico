#pragma once

#include "data/field/include/field_em.hpp"
#include "data/grid/include/grid.hpp"

#include <cstddef>

namespace pico::modules::boundary
{

template <std::size_t BLOCK_SIZE>
struct SilverMullerFieldBoundary
{
    static void fill_field_guards(EMFields<BLOCK_SIZE>& fields, const Grid& grid)
    {
        const std::size_t G = grid.guard_cells();
        const std::size_t N = grid.physical_size();

        // Left Open Boundary (-x wave: Ey = -Bz, Ez = +By)
        for (std::size_t g = 0; g < G; ++g)
        {
            // Transverse Mode 1
            fields.E.field_y(g) = -fields.B.field_z(G);
            fields.B.field_z(g) = -fields.E.field_y(G);

            // Transverse Mode 2
            fields.E.field_z(g) = fields.B.field_y(G);
            fields.B.field_y(g) = fields.E.field_z(G);

            // Electrostatic / Longitudinal (Neumann zero-gradient)
            fields.E.field_x(g) = fields.E.field_x(G);
            fields.B.field_x(g) = fields.B.field_x(G);
        }

        // Right Open Boundary (+x wave: Ey = +Bz, Ez = -By)
        for (std::size_t g = 0; g < G; ++g)
        {
            const std::size_t last_phys = N + G - 1;
            const std::size_t guard_idx = N + G + g;

            // Transverse Mode 1
            fields.E.field_y(guard_idx) = fields.B.field_z(last_phys);
            fields.B.field_z(guard_idx) = fields.E.field_y(last_phys);

            // Transverse Mode 2
            fields.E.field_z(guard_idx) = -fields.B.field_y(last_phys);
            fields.B.field_y(guard_idx) = -fields.E.field_z(last_phys);

            // Electrostatic / Longitudinal (Neumann zero-gradient)
            fields.E.field_x(guard_idx) = fields.E.field_x(last_phys);
            fields.B.field_x(guard_idx) = fields.B.field_x(last_phys);
        }
    }

    static void fold_currents(FieldSystem<BLOCK_SIZE>& J, const Grid& grid)
    {
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
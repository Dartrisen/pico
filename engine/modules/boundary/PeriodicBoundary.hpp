#pragma once

#include "data/field/include/field_em.hpp"
#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"

namespace pico::modules::boundary
{

template <std::size_t BLOCK_SIZE>
struct PeriodicBoundaryHandler
{
    // 1. Fold guard cell currents into physical domain, then reset guard cells
    static void fold_currents(FieldSystem<BLOCK_SIZE>& J, const Grid& grid)
    {
        const std::size_t N = grid.physical_size();
        const std::size_t G = grid.guard_cells();

        for (std::size_t g = 0; g < G; ++g)
        {
            // Left guard cell -> Right physical cell
            J.field_x(N + g) += J.field_x(g);
            J.field_x(g) = 0.0f;
            J.field_y(N + g) += J.field_y(g);
            J.field_y(g) = 0.0f;
            J.field_z(N + g) += J.field_z(g);
            J.field_z(g) = 0.0f;

            // Right guard cell -> Left physical cell
            J.field_x(G + g) += J.field_x(N + G + g);
            J.field_x(N + G + g) = 0.0f;
            J.field_y(G + g) += J.field_y(N + G + g);
            J.field_y(N + G + g) = 0.0f;
            J.field_z(G + g) += J.field_z(N + G + g);
            J.field_z(N + G + g) = 0.0f;
        }
    }

    // 2. Synchronize physical boundary fields to guard cells before gathering
    static void fill_field_guards(EMFields<BLOCK_SIZE>& fields, const Grid& grid)
    {
        const std::size_t N = grid.physical_size();
        const std::size_t G = grid.guard_cells();

        for (std::size_t g = 0; g < G; ++g)
        {
            // Right physical cells -> Left guard cells
            fields.E.field_x(g) = fields.E.field_x(N + g);
            fields.E.field_y(g) = fields.E.field_y(N + g);
            fields.E.field_z(g) = fields.E.field_z(N + g);

            fields.B.field_x(g) = fields.B.field_x(N + g);
            fields.B.field_y(g) = fields.B.field_y(N + g);
            fields.B.field_z(g) = fields.B.field_z(N + g);

            // Left physical cells -> Right guard cells
            fields.E.field_x(N + G + g) = fields.E.field_x(G + g);
            fields.E.field_y(N + G + g) = fields.E.field_y(G + g);
            fields.E.field_z(N + G + g) = fields.E.field_z(G + g);

            fields.B.field_x(N + G + g) = fields.B.field_x(G + g);
            fields.B.field_y(N + G + g) = fields.B.field_y(G + g);
            fields.B.field_z(N + G + g) = fields.B.field_z(G + g);
        }
    }
};
} // namespace pico::modules::boundary

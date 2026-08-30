#pragma once
#include "data/field/include/field_em.hpp"
#include "kernels/field/maxwell_yee.hpp"

namespace pico::modules::field
{

template <size_t BLOCK_SIZE>
struct YeeMaxwell
{
    template <typename FieldBoundaryT>
    inline void solve(EMFields<BLOCK_SIZE>& fields, const FieldSystem<BLOCK_SIZE>& J, FieldBoundaryT& boundary, float dt)
    {
        const Grid& grid = fields.E.grid();

        // 1. Advance B to t^(n+1/2)
        kernels::field::YeeMaxwell<BLOCK_SIZE>::advance_magnetic_field(fields.B, fields.E, grid, dt);

        // 2. REFRESH B GUARD CELLS (B[G-1] now gets fresh B[G+N-1] at t^(n+1/2))
        boundary.fill_field_guards(fields, grid);

        // 3. Advance E to t^(n+1) using consistent t^(n+1/2) B fields
        kernels::field::YeeMaxwell<BLOCK_SIZE>::advance_electric_field(fields.E, fields.B, J, grid, dt);
    }
};

} // namespace pico::modules::field
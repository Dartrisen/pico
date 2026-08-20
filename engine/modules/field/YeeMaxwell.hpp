#pragma once
#include "data/field/include/field_em.hpp"
#include "kernels/field/maxwell_yee.hpp"

namespace pico::modules::field
{

    template <size_t BLOCK_SIZE>
    struct YeeMaxwell
    {
        inline void solve(EMFields<BLOCK_SIZE>& fields, const FieldSystem<BLOCK_SIZE>& J, float dt)
        {
            const Grid& grid = fields.E.grid();

            kernels::field::MaxwellYeeKernel<BLOCK_SIZE>::update_B(fields, grid, dt);
            kernels::field::MaxwellYeeKernel<BLOCK_SIZE>::update_E(fields, J, grid, dt);
        }
    };

} // namespace pico::modules::field

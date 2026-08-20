#pragma once

namespace kernels::field
{

    template <size_t BLOCK_SIZE>
    struct MaxwellYeeKernel
    {
        static inline void update_B(EMFields<BLOCK_SIZE>& fields, const Grid& grid, float dt)
        {
            const float dx_inv = 1.f / grid.cell_size();

            // scalar loop version (keep for now)
            for (size_t i = 0; i < grid.size() - 1; ++i)
            {
                // Bx is constant in 1D propagation along x (d/dy = d/dz = 0)
                // dBy/dt =  dEz/dx
                // dBz/dt = -dEy/dx
                fields.B.field_y(i) += dt * dx_inv * (fields.E.field_z(i + 1) - fields.E.field_z(i));
                fields.B.field_z(i) -= dt * dx_inv * (fields.E.field_y(i + 1) - fields.E.field_y(i));
            }
        }

        static inline void update_E(EMFields<BLOCK_SIZE>& fields, const FieldSystem<BLOCK_SIZE>& J, const Grid& grid,
                                    float dt)
        {
            const float dx_inv = 1.f / grid.cell_size();

            for (size_t i = 1; i < grid.size() - 1; ++i)
            {
                // dEx/dt = -Jx  (Longitudinal electrostatics)
                // dEy/dt = -dBz/dx - Jy
                // dEz/dt =  dBy/dx - Jz
                fields.E.field_x(i) -= dt * J.field_x(i);
                fields.E.field_y(i) -= dt * (dx_inv * (fields.B.field_z(i) - fields.B.field_z(i - 1)) + J.field_y(i));
                fields.E.field_z(i) += dt * (dx_inv * (fields.B.field_y(i) - fields.B.field_y(i - 1)) - J.field_z(i));
            }
        }
    };

} // namespace kernels::field

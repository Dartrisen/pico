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
                fields.B[i] -= dt * dx_inv * (fields.E[i + 1] - fields.E[i]);
            }
        }

        static inline void update_E(EMFields<BLOCK_SIZE>& fields, const FieldSystem<BLOCK_SIZE>& J, const Grid& grid,
                                    float dt)
        {
            const float dx_inv = 1.f / grid.cell_size();

            for (size_t i = 1; i < grid.size() - 1; ++i)
            {
                fields.E[i] += dt * ((fields.B[i] - fields.B[i - 1]) * dx_inv - J[i]);
            }
        }
    };

} // namespace kernels::field

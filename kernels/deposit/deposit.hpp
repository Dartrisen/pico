#pragma once

#include "data/field/include/field_system.hpp"
#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"

namespace kernels::deposit
{

template <class Shape, std::size_t BLOCK_SIZE>
struct CurrentDeposit
{
private:
    template <::field::FieldComp Component>
    static void deposit_component(FieldSystem<BLOCK_SIZE>& J, double particle_x, float flux, float inv_ppc,
                                  float inv_dx, double dx, std::size_t guard_cells, double grid_offset)
    {
        constexpr int Support = Shape::S;

        int    first_index = 0;
        double weights[Support]{};

        const double shifted_x = particle_x - grid_offset * dx;
        Shape::weights(shifted_x, dx, first_index, weights);

        const int buffer_start = first_index + static_cast<int>(guard_cells);

        for (int stencil = 0; stencil < Support; ++stencil)
        {
            // Included inv_dx (1 / dx) for 1D density scaling
            const float ws = static_cast<float>(weights[stencil]) * inv_ppc * inv_dx;
            J.template field<Component>(static_cast<std::size_t>(buffer_start + stencil)) += ws * flux;
        }
    }

public:
    static void deposit(const particle::ParticleBlock<BLOCK_SIZE>& pb, FieldSystem<BLOCK_SIZE>& J, const Grid& grid,
                        float dt, float ppc)
    {
        constexpr double NodeOffset     = 0.0;
        constexpr double HalfCellOffset = 0.5;

        const double      dx          = grid.cell_size();
        const std::size_t guard_cells = grid.guard_cells();
        const float       inv_ppc     = 1.0f / ppc;
        const float       inv_dx      = static_cast<float>(1.0 / dx);

        for (std::size_t p = 0; p < pb.activeCount; ++p)
        {
            const double x     = static_cast<double>(pb.position_x[p]);
            const float  inv_m = 1.0f / pb.mass[p];
            const float  q     = pb.charge[p];

            const float qvx = q * (pb.momentum_x[p] * inv_m);
            const float qvy = q * (pb.momentum_y[p] * inv_m);
            const float qvz = q * (pb.momentum_z[p] * inv_m);

            deposit_component<::field::FieldComp::X>(J, x, qvx, inv_ppc, inv_dx, dx, guard_cells, HalfCellOffset);
            deposit_component<::field::FieldComp::Y>(J, x, qvy, inv_ppc, inv_dx, dx, guard_cells, NodeOffset);
            deposit_component<::field::FieldComp::Z>(J, x, qvz, inv_ppc, inv_dx, dx, guard_cells, NodeOffset);
        }
    }
};

} // namespace kernels::deposit
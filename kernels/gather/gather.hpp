#pragma once

#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_block.hpp"

#include <cstddef>

namespace kernels::gather
{

    template <class Shape, std::size_t BLOCK_SIZE>
    struct FieldGather
    {
    private:
        static int wrap_index(int index, int grid_size) noexcept
        {
            index %= grid_size;

            if (index < 0)
            {
                index += grid_size;
            }

            return index;
        }

        template <::field::FieldComp Component>
        static float interpolate_component(const FieldSystem<BLOCK_SIZE>& field_system, double particle_x, double dx,
                                           int grid_size, double grid_offset)
        {
            constexpr int Support = Shape::S;

            int    first_index = 0;
            double weights[Support]{};

            /*
             * Field samples are located at:
             *
             *     x_i = (i + grid_offset) * dx
             *
             * Shape::weights() expects samples at i * dx, so shift the
             * particle coordinate by the component's Yee-grid offset.
             */
            const double shifted_x = particle_x - grid_offset * dx;

            Shape::weights(shifted_x, dx, first_index, weights);

            float result = 0.0f;

            for (int stencil = 0; stencil < Support; ++stencil)
            {
                const int index = wrap_index(first_index + stencil, grid_size);

                result += static_cast<float>(weights[stencil]) *
                          field_system.template field<Component>(static_cast<std::size_t>(index));
            }

            return result;
        }

    public:
        static void gather(const particle::ParticleBlock<BLOCK_SIZE>& particle_block,
                           const EMFields<BLOCK_SIZE>& fields, const Grid& grid, FieldScratch<BLOCK_SIZE>& scratch)
        {
            constexpr int Support = Shape::S;

            static_assert(Support == 2, "The initial PIC implementation currently supports "
                                        "linear CIC gathering only.");

            constexpr double NodeOffset     = 0.0;
            constexpr double HalfCellOffset = 0.5;

            const double dx        = grid.cell_size();
            const int    grid_size = static_cast<int>(grid.size());

            for (std::size_t particle = 0; particle < particle_block.activeCount; ++particle)
            {
                const double x = static_cast<double>(particle_block.position_x[particle]);

                /*
                 * 1D3V Yee-grid component locations:
                 *
                 * Ex:     i + 1/2
                 * Ey, Ez: i
                 * Bx:     i
                 * By, Bz: i + 1/2
                 */

                scratch.Ex[particle] =
                        interpolate_component<::field::FieldComp::X>(fields.E, x, dx, grid_size, HalfCellOffset);

                scratch.Ey[particle] =
                        interpolate_component<::field::FieldComp::Y>(fields.E, x, dx, grid_size, NodeOffset);

                scratch.Ez[particle] =
                        interpolate_component<::field::FieldComp::Z>(fields.E, x, dx, grid_size, NodeOffset);

                scratch.Bx[particle] =
                        interpolate_component<::field::FieldComp::X>(fields.B, x, dx, grid_size, NodeOffset);

                scratch.By[particle] =
                        interpolate_component<::field::FieldComp::Y>(fields.B, x, dx, grid_size, HalfCellOffset);

                scratch.Bz[particle] =
                        interpolate_component<::field::FieldComp::Z>(fields.B, x, dx, grid_size, HalfCellOffset);
            }
        }
    };

} // namespace kernels::gather

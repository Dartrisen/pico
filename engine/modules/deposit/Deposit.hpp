#pragma once
#include "kernels/deposit/deposit.hpp"

namespace pico::modules::deposit
{

    template <class Shape, size_t BLOCK_SIZE>
    struct SimpleDeposit
    {
        template <typename Block, typename FieldSystem, typename GridT>
        void deposit_block(const Block& block, FieldSystem& current, const GridT& grid, double dt, uint32_t n_ppc) const
        {
            kernels::deposit::CurrentDeposit<Shape, BLOCK_SIZE>::deposit(block, current, grid, dt, n_ppc);
        }
    };

} // namespace pico::modules::deposit
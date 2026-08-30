#pragma once
#include "kernels/deposit/deposit.hpp"
#include "kernels/deposit/esirkepov.hpp"

namespace pico::modules::deposit
{

template <class Shape, size_t BLOCK_SIZE>
struct SimpleDeposit
{
    template <typename Block, typename FieldSystem, typename GridT>
    void deposit_block(const Block& block, FieldSystem& current, const GridT& grid, double dt, uint32_t n_ppc, float base_charge) const
    {
        kernels::deposit::CurrentDeposit<Shape, BLOCK_SIZE>::deposit(block, current, grid, dt, n_ppc, base_charge);
    }
};

template <class Shape, std::size_t BLOCK_SIZE>
struct EsirkepovDeposit
{
    template <typename Block, typename FieldSystem, typename GridT>
    void deposit_block(const Block& block, FieldSystem& current, const GridT& grid, double dt, uint32_t n_ppc, float base_charge) const
    {
        kernels::deposit::EsirkepovDeposit<Shape, BLOCK_SIZE>::deposit(block, current, grid, dt, n_ppc, base_charge);
    }
};

} // namespace pico::modules::deposit
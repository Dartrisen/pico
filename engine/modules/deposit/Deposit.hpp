#pragma once
#include "kernels/deposit/deposit.hpp"
#include "kernels/deposit/esirkepov.hpp"

namespace pico::modules::deposit
{

template <class Shape, size_t BLOCK_SIZE>
struct SimpleDeposit
{
    template <typename Block, typename FieldSystem, typename GridT>
    void deposit_block(const Block& block, FieldSystem& current, const GridT& grid, double dt, uint32_t n_ppc, float base_charge, float base_mass) const
    {
        kernels::deposit::CurrentDeposit<Shape, BLOCK_SIZE>::deposit(block, current, grid, dt, n_ppc, base_charge, base_mass);
    }
};

template <class Shape, std::size_t BLOCK_SIZE>
struct EsirkepovDeposit
{
    template <typename Block, typename FieldSystem, typename GridT>
    void deposit_block(const Block& block, FieldSystem& current, const GridT& grid, double dt, uint32_t n_ppc, float base_charge, float base_mass) const
    {
        kernels::deposit::EsirkepovDeposit<Shape, BLOCK_SIZE>::deposit(block, current, grid, dt, n_ppc, base_charge, base_mass);
    }
};

template <class Shape, std::size_t BLOCK_SIZE>
struct FastEsirkepovDeposit
{
    template <typename Block, typename FieldSystem, typename GridT>
    void deposit_block(const Block& block, FieldSystem& current, const GridT& grid, double dt, uint32_t n_ppc, float base_charge, float base_mass) const
    {
        kernels::deposit::FastEsirkepovDeposit<Shape, BLOCK_SIZE>::deposit(block, current, grid, dt, n_ppc, base_charge, base_mass);
    }
};

template <class Shape, std::size_t BLOCK_SIZE>
struct OptEsirkepovDeposit
{
    template <typename Block, typename FieldSystem, typename GridT>
    void deposit_block(const Block& block, FieldSystem& current, const GridT& grid, double dt, uint32_t n_ppc, float base_charge, float base_mass) const
    {
        kernels::deposit::OptEsirkepovDeposit<Shape, BLOCK_SIZE>::deposit(block, current, grid, dt, n_ppc, base_charge, base_mass);
    }
};

} // namespace pico::modules::deposit
#pragma once

#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

namespace kernels::sorter
{

template <std::size_t BLOCK_SIZE>
struct ParticleSorterScratch
{
    std::vector<double>      pos_x_buf;
    std::vector<float>       mom_x_buf, mom_y_buf, mom_z_buf, mass_buf, charge_buf;
    std::vector<std::size_t> indices;

    void resize(std::size_t count)
    {
        pos_x_buf.resize(count);
        mom_x_buf.resize(count);
        mom_y_buf.resize(count);
        mom_z_buf.resize(count);
        mass_buf.resize(count);
        charge_buf.resize(count);
        indices.resize(count);
    }
};

template <std::size_t BLOCK_SIZE>
struct ParticleSorter
{
    static void sort_block(particle::ParticleBlock<BLOCK_SIZE>& block, const Grid& grid, ParticleSorterScratch<BLOCK_SIZE>& scratch)
    {
        const std::size_t count = block.activeCount;
        if (count <= 1)
        {
            return;
        }

        scratch.resize(count);

        std::iota(scratch.indices.begin(), scratch.indices.end(), 0);

        const double inv_dx = 1.0 / grid.cell_size();

        std::sort(scratch.indices.begin(), scratch.indices.end(),
                  [&](std::size_t a, std::size_t b)
                  {
                      const auto cell_a = static_cast<std::size_t>(block.position_x[a] * inv_dx);
                      const auto cell_b = static_cast<std::size_t>(block.position_x[b] * inv_dx);
                      return cell_a < cell_b;
                  });

        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t src = scratch.indices[i];
            scratch.pos_x_buf[i]  = block.position_x[src];
            scratch.mom_x_buf[i]  = block.momentum_x[src];
            scratch.mom_y_buf[i]  = block.momentum_y[src];
            scratch.mom_z_buf[i]  = block.momentum_z[src];
            scratch.mass_buf[i]   = block.mass[src];
            scratch.charge_buf[i] = block.charge[src];
        }

        std::copy_n(scratch.pos_x_buf.begin(), count, std::data(block.position_x));
        std::copy_n(scratch.mom_x_buf.begin(), count, std::data(block.momentum_x));
        std::copy_n(scratch.mom_y_buf.begin(), count, std::data(block.momentum_y));
        std::copy_n(scratch.mom_z_buf.begin(), count, std::data(block.momentum_z));
        std::copy_n(scratch.mass_buf.begin(), count, std::data(block.mass));
        std::copy_n(scratch.charge_buf.begin(), count, std::data(block.charge));
    }
};

} // namespace kernels::sorter
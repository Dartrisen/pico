#pragma once

#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"
#include "data/particle/include/particle_system.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kernels::sorter
{

template <std::size_t BLOCK_SIZE>
struct ParticleSorterScratch
{
    // Number of particles in each spatial cell.
    std::vector<std::size_t> counts;

    // First output particle for each cell.
    std::vector<std::size_t> offsets;

    // Current write position for each cell.
    std::vector<std::size_t> write_pos;

    // Spatially reordered blocks.
    std::vector<particle::ParticleBlock<BLOCK_SIZE>> sorted_blocks;

    void resize_cells(std::size_t n)
    {
        counts.resize(n);
        offsets.resize(n);
        write_pos.resize(n);
    }

    void resize_blocks(std::size_t n)
    {
        sorted_blocks.resize(n);
    }
};

template <std::size_t BLOCK_SIZE>
struct ParticleSorter
{
    using Block = particle::ParticleBlock<BLOCK_SIZE>;

    static void sort(particle::ParticleSystem<BLOCK_SIZE>& particles, const Grid& grid, ParticleSorterScratch<BLOCK_SIZE>& scratch)
    {
        const std::size_t particle_count = particles.active_particles();
        if (particle_count <= 1)
            return;

        const double inv_dx = 1.0 / grid.cell_size();

        // ------------------------------------------------------------
        // Number of spatial bins.
        //
        // This sorter is currently 1D, so one bin = one x-cell.
        // ------------------------------------------------------------

        const std::size_t num_cells = static_cast<std::size_t>(grid.physical_size());
        if (num_cells == 0)
            return;

        scratch.resize_cells(num_cells);

        // ------------------------------------------------------------
        // Number of blocks required.
        // ------------------------------------------------------------

        const std::size_t num_blocks = (particle_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scratch.resize_blocks(num_blocks);

        // ------------------------------------------------------------
        // 1. Histogram directly from ParticleBlocks.
        // ------------------------------------------------------------
        std::fill(scratch.counts.begin(), scratch.counts.end(), std::size_t{0});
        for (const auto& block : particles)
        {
            for (std::size_t i = 0; i < block.activeCount; ++i)
            {
                const double x    = block.position_x[i];
                std::size_t  cell = x > 0.0 ? static_cast<std::size_t>(x * inv_dx) : 0;

                if (cell >= num_cells)
                    cell = num_cells - 1;

                ++scratch.counts[cell];
            }
        }

        // ------------------------------------------------------------
        // 2. Prefix sum.
        // ------------------------------------------------------------
        scratch.offsets[0] = 0;
        for (std::size_t cell = 1; cell < num_cells; ++cell)
        {
            scratch.offsets[cell] = scratch.offsets[cell - 1] + scratch.counts[cell - 1];
        }

        std::copy(scratch.offsets.begin(), scratch.offsets.end(), scratch.write_pos.begin());

        // ------------------------------------------------------------
        // 3. Scatter directly into sorted ParticleBlocks.
        // ------------------------------------------------------------
        for (auto& block : scratch.sorted_blocks)
        {
            block.activeCount = 0;
        }

        for (const auto& block : particles)
        {
            for (std::size_t i = 0; i < block.activeCount; ++i)
            {
                const double x    = block.position_x[i];
                std::size_t  cell = x > 0.0 ? static_cast<std::size_t>(x * inv_dx) : 0;

                if (cell >= num_cells)
                    cell = num_cells - 1;

                const std::size_t dst      = scratch.write_pos[cell]++;
                const std::size_t block_id = dst / BLOCK_SIZE;
                const std::size_t local_id = dst % BLOCK_SIZE;

                auto& dst_block = scratch.sorted_blocks[block_id];

                dst_block.position_x[local_id] = block.position_x[i];

                dst_block.momentum_x[local_id] = block.momentum_x[i];
                dst_block.momentum_y[local_id] = block.momentum_y[i];
                dst_block.momentum_z[local_id] = block.momentum_z[i];

                dst_block.weight[local_id] = block.weight[i];
                dst_block.mass[local_id]   = block.mass[i];
                dst_block.charge[local_id] = block.charge[i];

                ++dst_block.activeCount;
            }
        }

        // ------------------------------------------------------------
        // 4. Copy reordered blocks back.
        // ------------------------------------------------------------
        std::size_t block_id = 0;
        for (auto& block : particles)
        {
            auto& sorted_block = scratch.sorted_blocks[block_id++];

            block.activeCount = sorted_block.activeCount;
            for (std::size_t i = 0; i < sorted_block.activeCount; ++i)
            {
                block.position_x[i] = sorted_block.position_x[i];

                block.momentum_x[i] = sorted_block.momentum_x[i];
                block.momentum_y[i] = sorted_block.momentum_y[i];
                block.momentum_z[i] = sorted_block.momentum_z[i];

                block.weight[i] = sorted_block.weight[i];
                block.mass[i]   = sorted_block.mass[i];
                block.charge[i] = sorted_block.charge[i];
            }
        }
    }
};

} // namespace kernels::sorter
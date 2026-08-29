#pragma once

#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"
#include "data/particle/include/particle_system.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace kernels::sorter
{

template <std::size_t BLOCK_SIZE>
struct ParticleSorterScratch
{
    std::vector<std::size_t>                         counts;
    std::vector<std::size_t>                         offsets;
    std::vector<std::size_t>                         write_pos;
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

        const std::size_t num_cells = static_cast<std::size_t>(grid.physical_size());
        if (num_cells == 0)
            return;

        const double      inv_dx        = 1.0 / grid.cell_size();
        const std::size_t system_blocks = particles.num_blocks();

        scratch.resize_cells(num_cells);
        scratch.resize_blocks(system_blocks);

        // Helper: Map position to spatial cell with boundary clamping
        auto get_cell = [num_cells, inv_dx](double x) -> std::size_t
        {
            if (x <= 0.0)
                return 0;
            const std::size_t c = static_cast<std::size_t>(x * inv_dx);
            return std::min(c, num_cells - 1);
        };

        // 1. Histogram directly from ParticleBlocks
        std::fill(scratch.counts.begin(), scratch.counts.end(), std::size_t{0});
        for (const auto& block : particles)
        {
            for (std::size_t i = 0; i < block.activeCount; ++i)
            {
                ++scratch.counts[get_cell(block.position_x[i])];
            }
        }

        // 2. Prefix sum
        scratch.offsets[0] = 0;
        for (std::size_t cell = 1; cell < num_cells; ++cell)
        {
            scratch.offsets[cell] = scratch.offsets[cell - 1] + scratch.counts[cell - 1];
        }
        std::copy(scratch.offsets.begin(), scratch.offsets.end(), scratch.write_pos.begin());

        // 3. Pre-calculate activeCount for output blocks
        const std::size_t active_blocks = (particle_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (std::size_t b = 0; b < system_blocks; ++b)
        {
            if (b < active_blocks - 1)
            {
                scratch.sorted_blocks[b].activeCount = BLOCK_SIZE;
            }
            else if (b == active_blocks - 1)
            {
                const std::size_t rem                = particle_count % BLOCK_SIZE;
                scratch.sorted_blocks[b].activeCount = (rem == 0) ? BLOCK_SIZE : rem;
            }
            else
            {
                scratch.sorted_blocks[b].activeCount = 0;
            }
        }

        // 4. Scatter particles into sorted ParticleBlocks
        for (const auto& block : particles)
        {
            for (std::size_t i = 0; i < block.activeCount; ++i)
            {
                const std::size_t cell     = get_cell(block.position_x[i]);
                const std::size_t dst      = scratch.write_pos[cell]++;
                const std::size_t block_id = dst / BLOCK_SIZE;
                const std::size_t local_id = dst % BLOCK_SIZE;

                auto& dst_block = scratch.sorted_blocks[block_id];

                dst_block.position_x[local_id] = block.position_x[i];

                dst_block.momentum_x[local_id] = block.momentum_x[i];
                dst_block.momentum_y[local_id] = block.momentum_y[i];
                dst_block.momentum_z[local_id] = block.momentum_z[i];

                dst_block.inv_gamma[local_id] = block.inv_gamma[i];
                dst_block.weight[local_id]    = block.weight[i];
            }
        }

        // 5. Copy reordered blocks back
        std::size_t block_idx = 0;
        for (auto& block : particles)
        {
            block = scratch.sorted_blocks[block_idx++];
        }
    }
};

} // namespace kernels::sorter
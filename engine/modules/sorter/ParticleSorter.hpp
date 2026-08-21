#pragma once

#include "data/grid/include/grid.hpp"
#include "data/particle/include/particle_block.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

namespace pico::modules::sorter
{
template <std::size_t BLOCK_SIZE>
class ParticleSorter
{
private:
    std::vector<double>      pos_x_buf_;
    std::vector<float>       mom_x_buf_, mom_y_buf_, mom_z_buf_, mass_buf_, charge_buf_;
    std::vector<std::size_t> indices_;

public:
    void sort_block(particle::ParticleBlock<BLOCK_SIZE>& block, const Grid& grid)
    {
        const std::size_t count = block.activeCount;
        if (count <= 1)
            return;

        pos_x_buf_.resize(count);
        mom_x_buf_.resize(count);
        mom_y_buf_.resize(count);
        mom_z_buf_.resize(count);
        mass_buf_.resize(count);
        charge_buf_.resize(count);
        indices_.resize(count);

        std::iota(indices_.begin(), indices_.end(), 0);

        const double inv_dx = 1.0 / grid.cell_size();

        std::sort(indices_.begin(), indices_.end(),
                  [&](std::size_t a, std::size_t b)
                  {
                      const auto cell_a = static_cast<std::size_t>(block.position_x[a] * inv_dx);
                      const auto cell_b = static_cast<std::size_t>(block.position_x[b] * inv_dx);
                      return cell_a < cell_b;
                  });

        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t src = indices_[i];
            pos_x_buf_[i]         = block.position_x[src];
            mom_x_buf_[i]         = block.momentum_x[src];
            mom_y_buf_[i]         = block.momentum_y[src];
            mom_z_buf_[i]         = block.momentum_z[src];
            mass_buf_[i]          = block.mass[src];
            charge_buf_[i]        = block.charge[src];
        }

        // Copy back data via std::data() iterators
        std::copy_n(pos_x_buf_.begin(), count, std::data(block.position_x));
        std::copy_n(mom_x_buf_.begin(), count, std::data(block.momentum_x));
        std::copy_n(mom_y_buf_.begin(), count, std::data(block.momentum_y));
        std::copy_n(mom_z_buf_.begin(), count, std::data(block.momentum_z));
        std::copy_n(mass_buf_.begin(), count, std::data(block.mass));
        std::copy_n(charge_buf_.begin(), count, std::data(block.charge));
    }
};
} // namespace pico::modules::sorter
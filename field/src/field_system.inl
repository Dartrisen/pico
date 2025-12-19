#include "field/include/field_system.hpp"
#include "grid/include/grid.hpp"


/**
 * @brief Construct ParticleSystem using default mass and charge values
 */
template <size_t BLOCK_SIZE>
FieldSystem<BLOCK_SIZE>::FieldSystem(const Grid& grid)
  : grid_(grid)
{
    if (num_blocks_ == 0) throw std::bad_alloc();
    const size_t total_cells = grid_.num_cells();
    num_blocks_ = (total_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks_.resize(num_blocks_);
}

template <size_t BLOCK_SIZE>
field::FieldBlock<BLOCK_SIZE>& FieldSystem<BLOCK_SIZE>::block(size_t b) noexcept { return blocks_[b]; }

template <size_t BLOCK_SIZE>
const field::FieldBlock<BLOCK_SIZE>& FieldSystem<BLOCK_SIZE>::block(size_t b) const noexcept { return blocks_[b]; }

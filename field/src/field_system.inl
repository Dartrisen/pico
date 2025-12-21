#include "field/include/field_system.hpp"
#include "grid/include/grid.hpp"


/**
 * @brief Construct ParticleSystem using default mass and charge values
 */
template <size_t BLOCK_SIZE>
FieldSystem<BLOCK_SIZE>::FieldSystem(const Grid& grid)
  : grid_(grid),
    num_blocks_((grid.size() + BLOCK_SIZE - 1)/BLOCK_SIZE),
    blocks_(num_blocks_)
{
    if (num_blocks_ == 0)
        throw std::bad_alloc();

    for (auto& block : blocks_) {
        std::fill(block.field_x.begin(), block.field_x.end(), 0.0f);
        //std::fill(block.field_y.begin(), block.field_y.end(), 0.0f);
        //std::fill(block.field_z.begin(), block.field_z.end(), 0.0f);
    }
}

template <size_t BLOCK_SIZE>
field::FieldBlock<BLOCK_SIZE>& FieldSystem<BLOCK_SIZE>::block(size_t b) noexcept { return blocks_[b]; }

template <size_t BLOCK_SIZE>
const field::FieldBlock<BLOCK_SIZE>& FieldSystem<BLOCK_SIZE>::block(size_t b) const noexcept { return blocks_[b]; }

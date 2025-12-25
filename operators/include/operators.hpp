#pragma once
#include "particle_block.hpp"
#include "field_em.hpp"


template <class Shape, size_t BLOCK_SIZE>
struct FieldGather {
    void operator()(const particle::ParticleBlock<BLOCK_SIZE>& pb,
                    const EMFields<BLOCK_SIZE>& fields,
                    const Grid& grid,
                    FieldScratch<BLOCK_SIZE>& scratch) const;
};

template <size_t BLOCK_SIZE>
struct BorisPusher {
    void operator()(particle::ParticleBlock<BLOCK_SIZE>& pb,
                    const FieldScratch<BLOCK_SIZE>& scratch,
                    float dt) const;
};

template <size_t BLOCK_SIZE>
struct MaxwellSolver {
    void operator()(EMFields<BLOCK_SIZE>& fields,
                    const FieldSystem<BLOCK_SIZE>& J,
                    const Grid& grid,
                    float dt) const;
};

template <class Shape, size_t BLOCK_SIZE>
struct CurrentDeposit {
    void operator()(const particle::ParticleBlock<BLOCK_SIZE>& pb,
                    FieldSystem<BLOCK_SIZE>& J,
                    const Grid& grid,
                    float dt) const;
};

#include "operators/src/operators.inl"

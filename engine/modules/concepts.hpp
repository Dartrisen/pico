#pragma once
#include "data/field/include/field_em.hpp"
#include "data/particle/include/particle_system.hpp"

#include <concepts>

namespace pico::modules
{

    // Field solver concept
    template <class T, class EMFields, class FieldSystem>
    concept FieldSolver = requires(T s, EMFields& f, FieldSystem& J, double dt) {
        { s.solve(f, J, dt) } -> std::same_as<void>;
    };

    // Particle pusher concept
    template <class T, class ParticleBlock, class FieldsScratch>
    concept Pusher = requires(T p, ParticleBlock& block, const FieldsScratch& f, double dt) {
        { p.push_block(block, f, dt) } -> std::same_as<void>;
    };

    // Collision model concept
    template <class T, class Particles>
    concept Collision = requires(T c, Particles& p, double dt) {
        { c.apply(p, dt) } -> std::same_as<void>;
    };

    // Gather concept: map grid fields onto particles
    template <class T, class ParticleBlock, class EMFields, class FieldScratch>
    concept Gather =
            requires(T g, const ParticleBlock& block, const EMFields& fields, const Grid& grid, FieldScratch& scratch) {
                { g.gather_block(block, fields, grid, scratch) } -> std::same_as<void>;
            };

    // Current / charge deposition concept
    template <class T, class Block, class FieldSystem>
    concept Deposit =
            requires(T d, const Block& block, FieldSystem& current, const Grid& grid, double dt, uint32_t n_ppc) {
                { d.deposit_block(block, current, grid, dt, n_ppc) } -> std::same_as<void>;
            };

    template <class B, class FieldsT, class SystemT>
    concept FieldBoundary = requires(B b, FieldsT& fields, SystemT& system, const Grid& grid) {
        { b.fill_field_guards(fields, grid) } -> std::same_as<void>;
        { b.fold_currents(system, grid) } -> std::same_as<void>;
    };

    template <class B, class BlockT>
    concept ParticleBoundary = requires(B b, BlockT& block, const Grid& grid) {
        { b.apply(block, grid) } -> std::same_as<void>;
    };

} // namespace pico::modules

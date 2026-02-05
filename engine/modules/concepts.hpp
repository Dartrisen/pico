#pragma once
#include <concepts>
#include "field/include/field_em.hpp"


namespace pico::modules {

// Field solver concept
template<class T, class EMFields, class FieldSystem>
concept FieldSolver = requires(T s, EMFields& f, FieldSystem& J, double dt) {
    { s.solve(f, J, dt) } -> std::same_as<void>;
};

// Particle pusher concept
template<class T, class Particles, class Fields>
concept Pusher = requires(T p, Particles& part, const Fields& f, double dt) {
    { p.push(part, f, dt) } -> std::same_as<void>;
};

// Collision model concept
template<class T, class Particles>
concept Collision = requires(T c, Particles& p, double dt) {
    { c.apply(p, dt) } -> std::same_as<void>;
};

// Current / charge deposition concept
template<class T, class Particles, class Fields>
concept Deposit = requires(T d, const Particles& p, Fields& f) {
    { d.deposit(p, f) } -> std::same_as<void>;
};

} // namespace pico::modules

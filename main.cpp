#include "block/particle_system.hpp"
#include <iostream>

int main() {
    constexpr size_t BS = 64;
    ParticleSystem<BS> ps(1024);
    ps.set_active(1024);
    ps.update_positions(0.016f);
    std::cout << "Max: " << ps.max_particles()
              << " Active: " << ps.active_particles()
              << " Blocks: " << ps.num_blocks() << "\n";
    return 0;
}

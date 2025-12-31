#include <cassert>

#include "particle_block.hpp"
#include "field_system.hpp"
#include "operators.hpp"


void test_boris_energy_conservation()
{
    constexpr size_t BS = 8;

    particle::ParticleBlock<BS> pb;
    pb.activeCount = 1;
    auto& mom_x = pb.component(particle::MomentumComp::X);
    auto& mom_y = pb.component(particle::MomentumComp::Y);
    auto& mom_z = pb.component(particle::MomentumComp::Z);
    mom_x[0] = 1.0f;

    FieldScratch<BS> fs{};
    fs.Bz[0] = 1.0f;

    BorisPusher<BS> boris;
    boris(pb, fs, 0.1f);

    float u2 = mom_x[0]*mom_x[0] + mom_y[0]*mom_y[0] + mom_z[0]*mom_z[0];
    assert(std::abs(u2 - 1.0f) < 1e-5);
}

int main(){
    test_boris_energy_conservation();
    return 0;
}
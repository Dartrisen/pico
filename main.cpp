#include "particle/include/particle_system.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstdint>

template <typename Func>
double median_time(Func f, int warmups = 3, int iterations = 11) {
    for (int i = 0; i < warmups; ++i) f();

    std::vector<double> times;
    times.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

template <size_t B>
double run_block_benchmark(size_t numParticles, float dt, int warmups, int iterations) {
    // create system, set active and measure UpdatePositions
    ParticleSystem<B> sys(numParticles);
    sys.set_active(numParticles);

    auto run = [&]() { sys.update_positions(dt); };
    return median_time(run, warmups, iterations);
}

int main() {
    const std::vector<size_t> block_sizes = {8, 16, 32, 64, 128, 256};
    const size_t NUM_PARTICLES = 100'000'000ULL;
    constexpr float deltaTime = 0.016f;
    constexpr int warmups = 3;
    constexpr int iterations = 11;

    // print table header
    std::cout << std::left
              << std::setw(12) << "BlockSize"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Throughput (Mpps)"
              << std::setw(18) << "Mem/particle (B)"
              << '\n';

    std::cout << std::string(12 + 15 + 18 + 18, '-') << '\n';

    for (size_t B : block_sizes) {
        double soa_time = 0.0;

        // dispatch to templated benchmark depending on B
        switch (B) {
            case 8:   soa_time = run_block_benchmark<8>(NUM_PARTICLES, deltaTime, warmups, iterations); break;
            case 16:  soa_time = run_block_benchmark<16>(NUM_PARTICLES, deltaTime, warmups, iterations); break;
            case 32:  soa_time = run_block_benchmark<32>(NUM_PARTICLES, deltaTime, warmups, iterations); break;
            case 64:  soa_time = run_block_benchmark<64>(NUM_PARTICLES, deltaTime, warmups, iterations); break;
            case 128: soa_time = run_block_benchmark<128>(NUM_PARTICLES, deltaTime, warmups, iterations); break;
            case 256: soa_time = run_block_benchmark<256>(NUM_PARTICLES, deltaTime, warmups, iterations); break;
            default: break;
        }

        // convert to ms and throughput (M particles per second)
        double time_ms = soa_time * 1e3;
        double mpps = (static_cast<double>(NUM_PARTICLES) / soa_time) / 1e6;

        // approximate memory per particle: size of a block divided by its B
        double mem_per_particle = 0.0;
        switch (B) {
            case 8:   mem_per_particle = double(sizeof(particle::ParticleBlock<8>)) / 8.0; break;
            case 16:  mem_per_particle = double(sizeof(particle::ParticleBlock<16>)) / 16.0; break;
            case 32:  mem_per_particle = double(sizeof(particle::ParticleBlock<32>)) / 32.0; break;
            case 64:  mem_per_particle = double(sizeof(particle::ParticleBlock<64>)) / 64.0; break;
            case 128: mem_per_particle = double(sizeof(particle::ParticleBlock<128>)) / 128.0; break;
            case 256: mem_per_particle = double(sizeof(particle::ParticleBlock<256>)) / 256.0; break;
            default: break;
        }

        std::cout << std::left
                  << std::setw(12) << B
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(18) << std::fixed << std::setprecision(3) << mpps
                  << std::setw(18) << std::fixed << std::setprecision(1) << mem_per_particle
                  << '\n';
    }

    return 0;
}

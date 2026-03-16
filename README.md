# Pico: High-Performance Particle-in-Cell Simulation Framework

Pico is a modern C++ framework for particle-in-cell (PIC) simulations, designed with a focus on performance, modularity, and extensibility. It implements a layered architecture that separates high-level orchestration from low-level computational kernels, enabling efficient simulations on modern hardware.

## Architecture Overview

Pico's architecture is built around three distinct layers, each with a specific responsibility and design philosophy. This separation allows for runtime configurability at the top level while maintaining compile-time optimizations in the computational core.

### Layer 1: Application Layer (Runtime Configuration)

The application layer handles input parsing, simulation orchestration, and output management. It uses virtual interfaces to enable runtime selection of physics models and algorithms.

**Key Components:**
- `PICApp`: Main application class that drives the simulation loop
- `IEngine`: Abstract interface for physics engines
- `EngineWrapper`: Type-erased wrapper that bridges virtual and templated code

**Design Principles:**
- Virtual functions are allowed here for flexibility
- No direct particle or field manipulation
- Handles I/O, diagnostics, and checkpointing

### Layer 2: Physics Engine (Algorithm Composition)

The physics engine implements PIC algorithms through compile-time composition using templates and CRTP (Curiously Recurring Template Pattern). This layer wires together different physics modules without runtime overhead.

**Key Components:**
- `EngineBase`: CRTP base class for engines
- `PICEngine`: Main PIC engine template that composes field solvers, particle pushers, and other modules
- Module concepts: Type-safe interfaces for physics components

**Design Principles:**
- Zero virtual calls in hot loops
- Template-based composition for optimal inlining
- Compile-time algorithm selection

### Layer 3: Inner Kernels (Performance-Critical Code)

The kernel layer contains the tight computational loops that operate on raw data structures. These are designed for maximum performance with SIMD-friendly memory layouts and minimal abstraction overhead.

**Key Components:**
- Maxwell solver kernels (Yee scheme)
- Particle pusher kernels (Boris algorithm)
- Deposition and interpolation kernels

**Design Principles:**
- Structure-of-arrays (SoA) data layouts
- Explicit loops with compile-time constants
- No heap allocation or virtual calls

### Why This Architecture?

This layered design was chosen to balance several competing requirements in scientific computing:

1. **Performance**: PIC simulations are computationally intensive, requiring millions of particle operations per timestep. The architecture ensures that hot loops have zero abstraction overhead through template instantiation and inlining.

2. **Modularity**: Different physics models (relativistic vs. non-relativistic, various field solvers) can be mixed and matched at compile time without runtime performance penalties.

3. **Extensibility**: New algorithms can be added by implementing the appropriate concepts, with automatic integration into the engine composition.

4. **Maintainability**: Clear separation of concerns makes the codebase easier to understand and modify, while preventing performance-critical code from being cluttered with high-level logic.

5. **Hardware Efficiency**: The design enables compiler optimizations like LTO (Link-Time Optimization) and takes advantage of modern CPU features through SIMD-friendly data structures.

## Adding a New Module

To add a new physics module (such as a collision model or different field solver), follow these steps:

### 1. Implement the Kernel

Create a new kernel in the `kernels/` directory. Kernels should be stateless, use static methods, and operate on raw data structures.

```cpp
// kernels/collision/simple_collision.hpp
namespace kernels::collision {
    template<size_t BLOCK_SIZE>
    struct SimpleCollisionKernel {
        static inline void apply(
            particle::ParticleBlock<BLOCK_SIZE>& block,
            float dt) {
            // Collision logic here
            for(size_t i = 0; i < block.activeCount; ++i) {
                // Modify particle velocities
            }
        }
    };
}
```

### 2. Create the Module Wrapper

Implement a module class in `engine/modules/` that uses the kernel and satisfies the appropriate concept.

```cpp
// engine/modules/collision/SimpleCollision.hpp
namespace pico::modules::collision {
    template<size_t BLOCK_SIZE>
    struct SimpleCollision {
        void apply(
            particle::ParticleSystem<BLOCK_SIZE>& particles,
            double dt) const {
            for(auto& block : particles) {
                kernels::collision::SimpleCollisionKernel<BLOCK_SIZE>::apply(
                    block, dt);
            }
        }
    };
}
```

### 3. Update PICEngine

Modify `PICEngine.hpp` to include the new module in the template parameters and advance method.

```cpp
template <
    class FieldSolverT,
    class PusherT,
    class CollisionT,  // New parameter
    size_t BLOCK_SIZE>
class PICEngine final : public EngineBase<PICEngine<...>> {
    // Add member variable
    CollisionT collision_;

    void advance_impl(double dt) {
        field_solver_.solve(fields_, current_, dt);
        pusher_.push(particles_, scratch_, dt);
        collision_.apply(particles_, dt);  // New call
    }
};
```

### 4. Define Concepts

Add the concept definition in `engine/modules/concepts.hpp` if it doesn't exist.

```cpp
template<class T, class ParticleSystem>
concept Collision = requires(T c, ParticleSystem& p, double dt) {
    { c.apply(p, dt) } -> std::same_as<void>;
};
```

### 5. Instantiate and Use

Create an instance with the new module:

```cpp
using Engine = PICEngine<
    YeeMaxwell<BS>,
    BorisPusher<BS>,
    SimpleCollision<BS>,  // New module
    BS>;

auto engine = std::make_unique<EngineWrapper<Engine>>(grid);
PICApp app(std::move(engine), dt);
```

## Codebase Structure

```
pico/
├── app/                   # Application layer
│   ├── PICApp.hpp         # Main application class
│   ├── IEngine.hpp        # Virtual engine interface
│   └── EngineWrapper.hpp  # Type erasure bridge
├── engine/                # Physics engine layer
│   ├── EngineBase.hpp     # CRTP base class
│   ├── PICEngine.hpp      # Main PIC engine
│   └── modules/           # Physics modules
│       ├── concepts.hpp   # Module concepts
│       ├── field/         # Field solvers
│       └── pusher/        # Particle pushers
├── kernels/               # Computational kernels
│   ├── field/             # Field update kernels
│   └── pusher/            # Particle push kernels
├── data/                  # Data structures
│   ├── field/             # Electromagnetic field systems
│   ├── grid/              # Spatial grid
│   └── particle/          # Particle systems
└── tests/                 # Unit tests
```

## Building and Running

Pico uses Bazel for build management. To build the project:

```bash
bazel build //app:app_bin
```

Run tests:

```bash
bazel test //tests/...
```

## Example Usage

```cpp
#include "app/PICApp.hpp"
#include "app/EngineWrapper.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "data/grid/include/grid.hpp"

int main() {
    constexpr size_t BS = 8;
    Grid grid(64, 1.0f);

    using Engine = PICEngine<
        pico::modules::field::YeeMaxwell<BS>,
        pico::modules::pusher::BorisPusher<BS>,
        BS>;

    auto engine = std::make_unique<EngineWrapper<Engine>>(grid);
    PICApp app(std::move(engine), 0.1);

    app.run(1000);
    return 0;
}
```

This framework demonstrates modern C++ techniques for high-performance scientific computing, balancing abstraction with computational efficiency.

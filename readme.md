<div align="center">

<pre style="background: transparent; border: none;"><font color="#3b82f6"><b>
  ██████╗   ██╗  ██████╗   ██████╗ 
  ██╔══██╗  ██║ ██╔════╝  ██╔═══██╗
  ██████╔╝  ██║ ██║       ██║   ██║
  ██╔═══╝   ██║ ██║       ██║   ██║
  ██║       ██║ ╚██████╗  ╚██████╔╝
  ╚═╝       ╚═╝  ╚═════╝   ╚═════╝ 
  High-Performance Particle-in-Cell Engine
</b></font></pre>

### A small, composable C++ particle-in-cell engine

Compile-time physics composition, cache-friendly particle blocks, and a clear path from prototype kernel to measured simulation.

<br>

`C++20`  |  `Bazel`  |  `OpenMP`  |  `GoogleTest`

</div>

Pico is a research-oriented particle-in-cell (PIC) framework. It provides a compact implementation of the main PIC data path: gather electromagnetic fields, push particles, apply particle boundaries, deposit current, and advance the fields. The project is deliberately explicit about where runtime flexibility ends and compile-time performance begins.

> **Project status**
> Pico currently contains a 1D fully-relativistic PIC path with a Yee-Maxwell field solver, Boris pusher, field gather, current deposition, boundary handlers, a laser injector, sorting, profiling, verification programs, and microbenchmarks. YAML/JSON configuration, checkpointing, and collision execution are not currently part of `PICEngine`.

## Contents

- [Architecture Overview](#architecture-overview)
- [The Timestep](#the-timestep)
- [Data and Memory Layout](#data-and-memory-layout)
- [Available Building Blocks](#available-building-blocks)
- [Build and Run](#build-and-run)
- [Construct an Engine](#construct-an-engine)
- [Add a New Kernel and Module](#add-a-new-kernel-and-module)
- [Concepts and Compile-Time Contracts](#concepts-and-compile-time-contracts)
- [Tests, Verification, and Benchmarks](#tests-verification-and-benchmarks)
- [Repository Layout](#repository-layout)
- [Current Boundaries](#current-boundaries)

## Architecture Overview

Pico uses a four-layer architecture that completely replaces slow runtime physics registries with compile-time template composition and static polymorphism (CRTP). Runtime virtual dispatch is isolated strictly to the top-level application wrapper, ensuring hot particle loops execute with zero virtual function overhead, full inline optimizations, and SIMD vectorization.

```text
Application
  PICApp -> IEngine -> EngineWrapper<EngineT>
                        |
Physics engine          v
  EngineBase<Derived> -> PICEngine<Field, BLOCK_SIZE Deposit, FieldBoundary, Gather, Injector, ParticleBoundary, Pusher, ...>
                        |
Data and kernels        v
  ParticleSystem, FieldSystem, Grid
  Yee, gather, Boris, deposit, shapes, sorting
```

### Application Layer (Runtime Boundary)

`PICApp` owns the high-level run loop, timestep tracking, and logging. `IEngine` is a minimal runtime interface, adapted by `EngineWrapper<EngineT>` via type erasure.

Virtual dispatch occurs **only once per timestep** at this outer boundary. Particle iteration and field computations stay entirely inside the concrete `PICEngine` instantiation. Simulation configuration (grid size, `dt`, steps, block size) is currently specified in [app/app.cpp](https://www.google.com/search?q=app/app.cpp).

### Physics Engine Layer (Static Composition)

`PICEngine` inherits from `EngineBase<PICEngine>` via the Curiously Recurring Template Pattern (CRTP) and composes algorithms directly via C++20 concepts:

* Field solver (`FieldSolver`)
* Field gather (`Gather`)
* Particle pusher (`Pusher`)
* Current/charge deposition (`Deposit`)
* Field boundary handler (`FieldBoundary`)
* Particle boundary handler (`ParticleBoundary`)
* Optional field injector (`Injector`)

By avoiding dynamic plugin registries or virtual dispatch in this layer, the compiler inline-expands module calls directly into the execution loop. Concept constraints live in [engine/modules/concepts.hpp](https://www.google.com/search?q=engine/modules/concepts.hpp).

### Data Layer (Memory Storage)

`Grid` defines physical domain indexing and guard cell offsets. `EMFields` manages electric and magnetic component blocks. `ParticleSystem` manages cache-aligned `ParticleBlock` instances, active counts, and thread-safe allocations.

### Kernel Layer (Inner Loops)

The `kernels/` directory contains pure, stateless computational routines operating on raw data pointers. Modules in `engine/modules/` adapt these kernels to the engine's policy signatures.

## The Timestep

For each call to `PICEngine::advance(dt)`, the engine executes:

1. **Particle Sorting:** Sort active particle blocks every 50 steps.
2. **Field Preparation:** Fill field guard cells and clear global current buffers.
3. **Thread-Local Execution (OpenMP):**
* Create private current and field scratch storage per thread.
* For each particle block:
* Gather `E` and `B` fields onto particle positions.
* Advance momentum and position with the configured pusher.
* Enforce particle domain boundaries.
* Deposit current into thread-local scratch space.




4. **Current Reduction:** Aggregate thread-local currents and fold guard cells.
5. **Field Advance:** Resolve Maxwell equations using the configured field solver.
6. **Injections & Profiling:** Apply field injectors and log stage timings with `PipelineProfiler`.

## Data and Memory Layout

Particle data uses Structure-of-Arrays (SoA) layout inside cache-aligned `ParticleBlock<BLOCK_SIZE>` structures (where `BLOCK_SIZE` must be a positive multiple of 16).

Field components ($E_x, E_y, E_z, B_x, B_y, B_z$) are stored in contiguous block structures. Guard cell padding is included in buffer backing storage, with physical cell mapping managed via `Grid::physical_to_buffer()`.

## Available Building Blocks

### Engine Modules

| Area | Implementation | Role | Notes |
| --- | --- | --- | --- |
| Field | `pico::modules::field::YeeMaxwell<BLOCK_SIZE>` | Yee-style Maxwell field update | 1D spatial discretization |
| Gather | `pico::modules::gather::Gather<Shape, BLOCK_SIZE>` | Field interpolation | Non-relativistic |
| Pusher | `pico::modules::pusher::BorisPusher<BLOCK_SIZE>` | Particle momentum advance | Non-relativistic |
| Deposit | `pico::modules::deposit::SimpleDeposit<Shape, BLOCK_SIZE>` | Current deposition | Non-relativistic |
| Field boundary | `PeriodicBoundaryFieldHandler`, `SilverMullerFieldBoundary` | Guard cell updates & current folding | Domain boundary policies |
| Particle boundary | `PeriodicBoundaryParticleHandler`, `ThermalizingParticleBoundary` | Particle boundary conditions | Domain boundary policies |
| Injector | `NoInjector`, `PlaneWaveLaserInjector` | External EM source excitation | Optional engine policy |
| Sorting | `ParticleSorter<BLOCK_SIZE>` | Spatial cell re-sorting | Memory locality optimization |

### Diagnostics and Profiling

Verification tools for plasma oscillations, energy/current conservation, and Landau damping reside in `engine/modules/diagnostics/`. Executable verifiers are located under `benchmarks/physics/`.

## Build and Run

Pico uses Bazel with C++20 support:

```bash
# Build main executable
bazel build //app:app_bin

# Run default 1D simulation
bazel run //app:app_bin

# Run test suite
bazel test //tests/...
```

Targeted execution:

```bash
bazel test //tests:yee_maxwell_test --test_output=all
bazel build --config=linux //engine //kernels/... //data/...
```

## Construct an Engine

Engine composition via template wiring:

```cpp
#include "app/EngineWrapper.hpp"
#include "app/PICApp.hpp"
#include "data/grid/include/grid.hpp"
#include "engine/PICEngine.hpp"
#include "engine/modules/boundary/PeriodicFieldBoundary.hpp"
#include "engine/modules/boundary/Thermalizing.hpp"
#include "engine/modules/deposit/Deposit.hpp"
#include "engine/modules/field/YeeMaxwell.hpp"
#include "engine/modules/gather/Gather.hpp"
#include "engine/modules/pusher/BorisPusher.hpp"
#include "kernels/shapes/shape.hpp"

constexpr std::size_t block_size = 64;
using Shape = kernels::shapes::Shape<1>;
using Field = pico::modules::field::YeeMaxwell<block_size>;
using Gather = pico::modules::gather::Gather<Shape, block_size>;
using Pusher = pico::modules::pusher::BorisPusher<block_size>;
using Deposit = pico::modules::deposit::SimpleDeposit<Shape, block_size>;
using FieldBoundary = pico::modules::boundary::PeriodicBoundaryFieldHandler<block_size>;
using ParticleBoundary = pico::modules::boundary::ThermalizingParticleBoundary<block_size>;

using Engine = PICEngine<Field, Deposit, FieldBoundary, Gather, ParticleBoundary, Pusher, block_size>;

Grid grid(256, 0.1);
auto wrapped = std::make_unique<EngineWrapper<Engine>>(Engine{grid, 10});
PICApp app(std::move(wrapped), 1e-3);
app.run(1000);
```

## Add a New Kernel and Module

1. **Implement Kernel:** Write a stateless static class in `kernels/`:
```cpp
namespace kernels::field {

template <std::size_t BLOCK_SIZE>
struct MyFieldKernel {
    static inline void update(FieldSystem<BLOCK_SIZE>& fields,
                              const FieldSystem<BLOCK_SIZE>& current,
                              double dt) {
        // Direct numerical loop over blocks
    }
};

} // namespace kernels::field
```


2. **Implement Engine Module Wrapper:** Adapt to standard concept interfaces:
```cpp
template <std::size_t BLOCK_SIZE>
struct MyFieldSolver {
    void solve(EMFields<BLOCK_SIZE>& fields,
               FieldSystem<BLOCK_SIZE>& current,
               double dt) const {
        kernels::field::MyFieldKernel<BLOCK_SIZE>::update(fields.E, current, dt);
    }
};
```


3. **Configure Build:** Expose headers in `BUILD.bazel` and substitute type in `PICEngine<...>`.

## Concepts and Compile-Time Contracts

| Concept | Contract Requirement |
| --- | --- |
| `FieldSolver` | `solve(fields, current, dt)` |
| `Gather` | `gather_block(block, fields, grid, scratch)` |
| `Pusher` | `push_block(block, scratch, dt)` |
| `Deposit` | `deposit_block(block, current, grid, dt, particles_per_cell)` |
| `FieldBoundary` | `fill_field_guards(fields, grid)` and `fold_currents(current, grid)` |
| `ParticleBoundary` | `apply(block, grid)` |

## Tests, Verification, and Benchmarks

```bash
# Unit tests
bazel test //tests:yee_maxwell_test

# Physics scenario verifiers
bazel run //benchmarks/physics:plasma_oscillation
bazel run //benchmarks/physics:energy_conservation
bazel run //benchmarks/physics:current_conservation
bazel run //benchmarks/physics:laser_propagation
bazel run //benchmarks/physics:landau_damping

# Performance microbenchmarks (those are actually outdated)
bazel run //benchmarks:bench_boris_push
bazel run //benchmarks:bench_full_pic
```

## Repository Layout

```text
app/                 Runtime application, type erasure wrappers, performance logs
benchmarks/          Physics scenario binaries and microbenchmarks
data/                Grid indexing, field blocks, particle memory layout
engine/              PICEngine composition, CRTP base classes, profiler
engine/modules/      Engine policy wrappers and C++20 concepts
kernels/             Stateless numerical loops (Yee, Boris push, deposition, splines)
tests/               GoogleTest suite
```

## Current Boundaries

* [x] Relativistic dynamics
* [ ] YAML, JSON, or command-line input file parsing
* [ ] Checkpoint and restart state serialization
* [ ] Binary collision execution in `PICEngine`
* [ ] Multi-dimensional spatial grid implementations (currently 1D)
* [ ] High-throughput file export and visualization layer

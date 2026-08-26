The `PICEngine` supports multi-species plasma, where every species (e.g., background electrons vs. high-energy ions) can have its own independent `ppc` resolution.

**PICEngine Architecture & Lifecycle**

`PICEngine` manages grid fields, thread-local current accumulators, and an arbitrary number of particle species across a parallelized SIMD pipeline.

**Constructor Workflow**

| Constructor | Purpose | Internal Action |
| --- | --- | --- |
| `explicit PICEngine(grid, boundaries...)` | Base Shell | Allocates `EMFields` and `FieldSystem` on `grid`. Initializes 0 species. |
| `PICEngine(grid, ppc, n0, q, m)` | Single Cold Species | Calls base constructor, then delegates to `add_species_uniform(...)`. |
| `PICEngine(grid, ppc, v_th, n0, q, m)` | Single Thermal Species | Calls base constructor, then delegates to `add_species_thermal(...)`. |

**Species Initialization & PPC Tracking**

When invoking any `add_species_*` method:

* **Allocation:** Computes $N_{\text{total}} = \text{grid.physical\_size()} \times \text{ppc}$ and resizes internal particle blocks.
* **Density Initialization:** Writes physical plasma density $n(x)$ directly into `pb.weight[i]`.
* **Registry:** Pushes `particles_per_cell` into the engine's `species_ppc_` array. This keeps `ppc` decoupled from the generic `ParticleSystem` container.

**Pipeline Execution Loop (`advance_impl`)**

In every time step `dt`, the engine executes five pipeline stages across OpenMP thread buffers:

* **1. Sorting:** Reorders macro-particles into cell-contiguous SIMD blocks if spatial memory stride degrades.
* **2. Boundary Fields:** Fills guard cells on `fields_`.
* **3. Parallel Particle Loop:**
    * **Gather:** Interpolates $E$ and $B$ fields from grid nodes to particle positions.
    * **Push:** Advances particle momentum using $q/m$.
    * **Particle Boundary:** Applies thermalizing or periodic boundary conditions.
    * **Deposit:** Invokes `deposit_block`, explicitly passing `ppc` from `species_ppc_[s]`. The deposit kernel multiplies `pb.weight[i]` by `base_scale = (q / (m * ppc)) * inv_dx` to convert macro-particle motion into physical grid current $J$.


* **4. Reduction:** Merges thread-local current buffers and folds boundaries.
* **5. Field Solver:** Solves Maxwell's equations using the updated total grid current $J$.
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

In every time step $\Delta t$, the engine executes five pipeline stages across OpenMP thread buffers:

* **1. Sorting:** Reorders macro-particles into cell-contiguous SIMD blocks if spatial memory stride degrades.
* **2. Boundary Fields:** Fills guard cells on `fields_`.
* **3. Parallel Particle Loop:**
    * **Gather:** Interpolates $E$ and $B$ fields from grid nodes to particle positions $x^n$.
    * **Momentum Push:** Advances particle momentum $\mathbf{u}^{n-1/2} \to \mathbf{u}^{n+1/2}$ using $q/m$.
    * **Relativistic Factor Update:** Recomputes $\mathtt{inv\_gamma}_i = 1/\sqrt{1 + \vert{}\mathbf{u}_i^{n+1/2}\vert{}^2}$ and updates velocity $\mathbf{v} = \mathbf{u} \cdot \mathtt{inv\_gamma}_i$.
    * **Position Push:** Advances spatial position $x^{n+1} = x^n + v_x^{n+1/2}\Delta t$.
    * **Particle Boundary:** Applies thermalizing or periodic boundary conditions on $x^{n+1}$ (recomputing $\mathtt{inv\_gamma}_i$ if particles reflect or thermalize).
    * **Deposit:** Invokes `deposit_block` with `ppc` from `species_ppc_[s]` and species charge $q$. `pb.weight[i]` is local physical density $n(x)$.
    * **Direct assignment (`SimpleDeposit` / `CurrentDeposit`).** Deposits $q\mathbf{v}$ at current position $x^{n+1/2}$. Half-node shape weights receive $J_x$; primal nodes receive $J_y, J_z$:

    $$J \mathrel{+}= \frac{q\,n}{\mathrm{ppc}}\,W(x)\,\mathbf{v}$$

    * **Charge-conserving (`EsirkepovDeposit`).** Evaluates shape weights at $x^n$ and $x^{n+1}$. Longitudinal current $J_x$ is derived from 1D charge continuity:

    $$J_{x,\,i+1/2} \mathrel{+}= \frac{q\,n\,\Delta x}{\mathrm{ppc}\,\Delta t} \sum_{k\le i}\bigl(W_k(x^n) - W_k(x^{n+1})\bigr)$$

    * Transverse current is time-centered and matches direct scaling:

    $$J_{y,z} \mathrel{+}= \frac{q\,n}{\mathrm{ppc}}\,\tfrac12\bigl(W(x^n) + W(x^{n+1})\bigr)\,v_{y,z}$$

* **4. Reduction:** Merges thread-local current buffers and folds boundaries.
* **5. Field Solver:** Solves Maxwell's equations using the updated total grid current $J$.
#include "operators/include/operators.hpp"
#include "grid/include/grid.hpp"
#include "field/include/field_em.hpp"


struct CIC {
    static inline void weights(float xp, float dx, int& i0, float& w0, float& w1) noexcept
    {
        float s = xp * (1.0f / dx);
        i0 = static_cast<int>(s);
        float f = s - i0;
        w0 = 1.0f - f;
        w1 = f;
    }
};

template <class Shape, size_t BLOCK_SIZE>
void FieldGather<Shape, BLOCK_SIZE>::operator()(
    const particle::ParticleBlock<BLOCK_SIZE>& pb,
    const EMFields<BLOCK_SIZE>& fields,
    const Grid& grid,
    FieldScratch<BLOCK_SIZE>& scratch) const
{
    scratch.clear();

    for (size_t p = 0; p < pb.size_x; ++p) {
        int i0;
        float w0, w1;

        // Compute shape weights
        Shape::weights(pb.position_x[p], grid.cell_size(), i0, w0, w1);

        // Linear grid indices
        const size_t idx0 = grid.idx(i0);
        const size_t idx1 = idx0 + 1;

        // Block indices
        const size_t b0 = idx0 / BLOCK_SIZE;
        const size_t o0 = idx0 % BLOCK_SIZE;

        const size_t b1 = idx1 / BLOCK_SIZE;
        const size_t o1 = idx1 % BLOCK_SIZE;

        // Access blocks directly (FAST)
        const auto& Eb0 = fields.E.block(b0);
        const auto& Bb0 = fields.B.block(b0);

        scratch.Ex[p] =
              w0 * Eb0.field_x[o0]
            + w1 * (b0 == b1
                ? Eb0.field_x[o1]
                : fields.E.block(b1).field_x[o1]);

        scratch.Bx[p] =
              w0 * Bb0.field_x[o0]
            + w1 * (b0 == b1
                ? Bb0.field_x[o1]
                : fields.B.block(b1).field_x[o1]);
    }
}

template <size_t BLOCK_SIZE>
void BorisPusher<BLOCK_SIZE>::operator()(
    particle::ParticleBlock<BLOCK_SIZE>& pb,
    const FieldScratch<BLOCK_SIZE>& fs,
    float dt) const
{
    const float half_dt = 0.5f * dt;

    for (size_t p = 0; p < pb.activeCount; ++p) {

        // --- half electric kick ---
        float ux = pb.momentum_x[p] + half_dt * fs.Ex[p];
        float uy = pb.momentum_y[p] + half_dt * fs.Ey[p];
        float uz = pb.momentum_z[p] + half_dt * fs.Ez[p];

        // --- magnetic rotation ---
        float tx = half_dt * fs.Bx[p];
        float ty = half_dt * fs.By[p];
        float tz = half_dt * fs.Bz[p];

        float t2 = tx*tx + ty*ty + tz*tz;
        float sx = 2.f * tx / (1.f + t2);
        float sy = 2.f * ty / (1.f + t2);
        float sz = 2.f * tz / (1.f + t2);

        // u' = u- + u- x t
        float uxp = ux + (uy*tz - uz*ty);
        float uyp = uy + (uz*tx - ux*tz);
        float uzp = uz + (ux*ty - uy*tx);

        // u+ = u- + u' x s
        ux += (uyp*sz - uzp*sy);
        uy += (uzp*sx - uxp*sz);
        uz += (uxp*sy - uyp*sx);

        // --- half electric kick ---
        pb.momentum_x[p] = ux + half_dt * fs.Ex[p];
        pb.momentum_y[p] = uy + half_dt * fs.Ey[p];
        pb.momentum_z[p] = uz + half_dt * fs.Ez[p];

        // --- position update ---
        pb.position_x[p] += dt * pb.momentum_x[p];
    }
}

template <class Shape, size_t BLOCK_SIZE>
void CurrentDeposit<Shape, BLOCK_SIZE>::operator()(
    const particle::ParticleBlock<BLOCK_SIZE>& pb,
    FieldSystem<BLOCK_SIZE>& J,
    const Grid& grid,
    float dt) const
{
    for (size_t p = 0; p < pb.size_x; ++p) {

        int i0;
        float w0, w1;

        Shape::weights(pb.momentum_x[p], grid.cell_size(), i0, w0, w1);

        float qvx = pb.charge[p] * pb.momentum_x[p];
        float qvy = pb.charge[p] * pb.momentum_y[p];
        float qvz = pb.charge[p] * pb.momentum_z[p];

        size_t idx0 = grid.idx(i0);
        size_t idx1 = grid.idx(i0 + 1);

        float J0 = J.field_x(idx0);
        float J1 = J.field_x(idx1);

        J0 += w0 * qvx;
        J0 += w0 * qvy;
        J0 += w0 * qvz;

        J1 += w1 * qvx;
        J1 += w1 * qvy;
        J1 += w1 * qvz;
    }
}

template <size_t BLOCK_SIZE>
void MaxwellSolver<BLOCK_SIZE>::operator()(
    EMFields<BLOCK_SIZE>& fields,
    const FieldSystem<BLOCK_SIZE>& J,
    const Grid& grid,
    float dt) const
{
    // Update B (half step)
    for (size_t i = 0; i < grid.size() - 1; ++i) {
        fields.B[i] -= dt / grid.cell_size() * (fields.E[i+1] - fields.E[i]);
    }

    // Update E (full step)
    for (size_t i = 1; i < grid.size() - 1; ++i) {
        fields.E[i] += dt * (
            (fields.B[i] - fields.B[i-1]) / grid.cell_size() - J[i]
        );
    }
}

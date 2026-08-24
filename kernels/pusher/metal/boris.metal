#include <metal_stdlib>
using namespace metal;

kernel void relativistic_boris_push(
    device float*       pos_x          [[buffer(0)]],
    device float*       mom_x          [[buffer(1)]],
    device float*       mom_y          [[buffer(2)]],
    device float*       mom_z          [[buffer(3)]],
    device const float* charge         [[buffer(4)]],
    device const float* mass           [[buffer(5)]],
    device const float* Ex             [[buffer(6)]],
    device const float* Ey             [[buffer(7)]],
    device const float* Ez             [[buffer(8)]],
    device const float* Bx             [[buffer(9)]],
    device const float* By             [[buffer(10)]],
    device const float* Bz             [[buffer(11)]],
    constant float&     dt             [[buffer(12)]],
    uint                p              [[thread_position_in_grid]])
{
    const float half_dt   = 0.5f * dt;
    const float q         = charge[p];
    const float m         = mass[p];
    const float half_q_dt = half_dt * q;

    // --- Half electric kick ---
    float px = mom_x[p] + half_q_dt * Ex[p];
    float py = mom_y[p] + half_q_dt * Ey[p];
    float pz = mom_z[p] + half_q_dt * Ez[p];

    // --- Compute Lorentz factor gamma ---
    const float p_sq  = px * px + py * py + pz * pz;
    const float gamma = sqrt(1.0f + p_sq / (m * m));

    // --- Magnetic rotation factor ---
    const float half_qm_dt_gamma = half_q_dt / (gamma * m);
    float tx = half_qm_dt_gamma * Bx[p];
    float ty = half_qm_dt_gamma * By[p];
    float tz = half_qm_dt_gamma * Bz[p];

    float t2 = tx * tx + ty * ty + tz * tz;
    float sx = 2.0f * tx / (1.0f + t2);
    float sy = 2.0f * ty / (1.0f + t2);
    float sz = 2.0f * tz / (1.0f + t2);

    // --- Rotation ---
    float pxp = px + (py * tz - pz * ty);
    float pyp = py + (pz * tx - px * tz);
    float pzp = pz + (px * ty - py * tx);

    px += (pyp * sz - pzp * sy);
    py += (pzp * sx - pxp * sz);
    pz += (pxp * sy - pyp * sx);

    // --- Second half electric kick ---
    px += half_q_dt * Ex[p];
    py += half_q_dt * Ey[p];
    pz += half_q_dt * Ez[p];

    mom_x[p] = px;
    mom_y[p] = py;
    mom_z[p] = pz;

    // --- Position Update ---
    const float p_final_sq  = px * px + py * py + pz * pz;
    const float gamma_final = sqrt(1.0f + p_final_sq / (m * m));

    pos_x[p] += dt * px / (gamma_final * m);
}
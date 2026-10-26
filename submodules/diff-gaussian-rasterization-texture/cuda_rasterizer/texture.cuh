// RGB-only canonical 3x3 texture sampling. Alpha/visibility are unchanged.
#pragma once
#include <cuda_runtime.h>

struct TextureBasis {
    float q0[3], d[3], denominator, dot, t;
    float u, v, fu, fv, weights[4];
    int ids[4];
    bool valid;
};

__device__ inline TextureBasis texture_basis(const float* packed, int id, float x, float y) {
    TextureBasis b;
    const float* p = packed + id * 12;
    b.denominator = 0.f; b.dot = 0.f;
    for (int k=0;k<3;k++) {
        b.q0[k] = p[k];
        b.d[k] = p[3+k*3]*x + p[4+k*3]*y + p[5+k*3];
        b.denominator += b.d[k]*b.d[k];
        b.dot += b.d[k]*b.q0[k];
    }
    const bool ray_valid = isfinite(b.denominator) && b.denominator > 1e-12f && isfinite(b.dot);
    b.denominator = fmaxf(b.denominator, 1e-12f);
    b.t = -b.dot / b.denominator;
    b.u = (b.q0[0] + b.t*b.d[0])/3.f + 1.f;
    b.v = (b.q0[1] + b.t*b.d[1])/3.f + 1.f;
    b.valid = ray_valid && isfinite(b.u) && isfinite(b.v) && isfinite(b.t);
    const float u = b.valid ? fminf(2.f,fmaxf(0.f,b.u)) : 1.f;
    const float v = b.valid ? fminf(2.f,fmaxf(0.f,b.v)) : 1.f;
    const int ix = min(1,(int)floorf(u)), iy = min(1,(int)floorf(v));
    b.fu = u-ix; b.fv = v-iy;
    b.ids[0]=iy*3+ix; b.ids[1]=iy*3+ix+1;
    b.ids[2]=(iy+1)*3+ix; b.ids[3]=(iy+1)*3+ix+1;
    b.weights[0]=(1.f-b.fu)*(1.f-b.fv); b.weights[1]=b.fu*(1.f-b.fv);
    b.weights[2]=(1.f-b.fu)*b.fv; b.weights[3]=b.fu*b.fv;
    return b;
}

__device__ inline float texture_value(const float* textures, int id, int ch, const TextureBasis& b) {
    if (!b.valid) return 0.f;
    float value=0.f;
    for (int k=0;k<4;k++) value += b.weights[k]*textures[id*27+b.ids[k]*3+ch];
    return value;
}

__device__ inline void texture_backward(const float* textures, const float* gains,
    int id, const TextureBasis& b, float x, float y, const float* grad_rgb,
    float* grad_texture, float* grad_uv, float* grad_gain) {
    if (!b.valid) return;
    float gu=0.f, gv=0.f;
    for (int ch=0;ch<3;ch++) {
        const float* q = textures+id*27;
        const float g = grad_rgb[ch]*gains[id*3+ch];
        for (int k=0;k<4;k++) atomicAdd(grad_texture+id*27+b.ids[k]*3+ch, g*b.weights[k]);
        atomicAdd(grad_gain+id*3+ch, grad_rgb[ch]*texture_value(textures,id,ch,b));
        gu += g*((1.f-b.fv)*(q[b.ids[1]*3+ch]-q[b.ids[0]*3+ch])
                 +b.fv*(q[b.ids[3]*3+ch]-q[b.ids[2]*3+ch]));
        gv += g*((1.f-b.fu)*(q[b.ids[2]*3+ch]-q[b.ids[0]*3+ch])
                 +b.fu*(q[b.ids[3]*3+ch]-q[b.ids[1]*3+ch]));
    }
    float gq[3] = {b.u>0.f && b.u<2.f ? gu/3.f : 0.f,
                   b.v>0.f && b.v<2.f ? gv/3.f : 0.f, 0.f};
    float gd = gq[0]*b.d[0]+gq[1]*b.d[1];
    const float inverse=1.f/b.denominator;
    for (int k=0;k<3;k++) {
        const float g0 = gq[k]-gd*b.d[k]*inverse;
        const float gr = b.t*gq[k]-gd*b.q0[k]*inverse
                         +2.f*b.dot*gd*b.d[k]*inverse*inverse;
        atomicAdd(grad_uv+id*12+k,g0);
        atomicAdd(grad_uv+id*12+3+k*3,gr*x);
        atomicAdd(grad_uv+id*12+4+k*3,gr*y);
        atomicAdd(grad_uv+id*12+5+k*3,gr);
    }
}

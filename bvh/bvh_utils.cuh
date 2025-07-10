#pragma once
#include <cuda_runtime.h>

// TODO: Morton code utilities, AABB helpers

struct AABB {
    float3 min;
    float3 max;
};

__host__ __device__ inline AABB merge(const AABB& a, const AABB& b) {
    return {
        make_float3(fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z)),
        make_float3(fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z))
    };
}
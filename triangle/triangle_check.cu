#include "triangle_utils.cuh"
#include "../include/kernel_launcher.h"
#include "../include/config.h"

__device__ float atomicMinFloat(float* addr, float value) {
    int* address_as_int = (int*)addr;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);

    return __int_as_float(old);
}

extern "C" __global__ void triangle_collision_kernel(
    const float* __restrict__ triangles1,
    const float* __restrict__ triangles2,
    bool* __restrict__ collisions,
    int B, int T1, int T2,
    float threshold,
    int64_t start, int num_pairs
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= B || tid >= num_pairs) return;

    int64_t global_idx = start + tid;
    int tri1_idx = global_idx / T2;
    int tri2_idx = global_idx % T2;

    // if (tri1_idx >= T1 || tri2_idx >= T2 || tri1_idx < 0 || tri2_idx < 0) {
    //     printf("❌ Out-of-bounds access: tri1_idx=%d, tri2_idx=%d (T1=%d, T2=%d)\n", tri1_idx, tri2_idx, T1, T2);
    //     return;
    // }

    const float* tri1_base = triangles1 + (batch_idx * T1 * 9) + (tri1_idx * 9);
    const float* tri2_base = triangles2 + (batch_idx * T2 * 9) + (tri2_idx * 9);

    float3 tri1[3];
    float3 tri2[3];
    for (int i = 0; i < 3; ++i) {
        tri1[i] = make_float3(tri1_base[i * 3 + 0], tri1_base[i * 3 + 1], tri1_base[i * 3 + 2]);
        tri2[i] = make_float3(tri2_base[i * 3 + 0], tri2_base[i * 3 + 1], tri2_base[i * 3 + 2]);
    }

    // Degenerate triangle check
    bool tri1_zero = true, tri2_zero = true;
    for (int k = 0; k < 3; ++k) {
        tri1_zero &= (tri1[k].x == 0.0f && tri1[k].y == 0.0f && tri1[k].z == 0.0f);
        tri2_zero &= (tri2[k].x == 0.0f && tri2[k].y == 0.0f && tri2[k].z == 0.0f);
    }
    if (tri1_zero || tri2_zero) return;

    float dist = sqrtf(triangle_distance(tri1, tri2));

    // printf(
    //     "distance: %.6f [batch %d, tri1 %d, tri2 %d] dist=%.6f | "
    //     "tri1: [(%.3f %.3f %.3f), (%.3f %.3f %.3f), (%.3f %.3f %.3f)] | "
    //     "tri2: [(%.3f %.3f %.3f), (%.3f %.3f %.3f), (%.3f %.3f %.3f)]\n",
    //     dist-threshold, batch_idx, tri1_idx, tri2_idx, dist,
    //     tri1[0].x, tri1[0].y, tri1[0].z,
    //     tri1[1].x, tri1[1].y, tri1[1].z,
    //     tri1[2].x, tri1[2].y, tri1[2].z,
    //     tri2[0].x, tri2[0].y, tri2[0].z,
    //     tri2[1].x, tri2[1].y, tri2[1].z,
    //     tri2[2].x, tri2[2].y, tri2[2].z
    // );

    // atomicMinFloat(&collisions[batch_idx], dist);

    if (dist <= threshold) {
        collisions[batch_idx] = true;
    }
}

void launch_tridist_kernel(
    const float* triangles1,
    const float* triangles2,
    bool* collisions,
    int B, int T1, int T2,
    float threshold
) {
    int max_blocks_y = 65535;  // Maximum allowed blocks in the y-dimension
    int chunk_size = max_blocks_y * THREADS_PER_BLOCK;

    int64_t total_pairs = static_cast<int64_t>(T1) * T2;

    for (int64_t start = 0; start < total_pairs; start += chunk_size) {
        int num_pairs = static_cast<int>(min(static_cast<int64_t>(chunk_size), total_pairs - start));
    
        dim3 blocks(B, (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 threads(THREADS_PER_BLOCK);
    
        triangle_collision_kernel<<<blocks, threads>>>(
            triangles1, triangles2, collisions,
            B, T1, T2, threshold,
            static_cast<int64_t>(start), num_pairs
        );
    }
}
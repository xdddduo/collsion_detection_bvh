#pragma once

void launch_tridist_kernel(
    const float* triangles1,
    const float* triangles2,
    bool* collisions,
    int B, int T1, int T2,
    float threshold
);
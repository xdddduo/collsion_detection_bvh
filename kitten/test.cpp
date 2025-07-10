#include <vector>
#include <cstdio>
#include "BVHwrapper.h"

int main() {
    using Kitten::Bound;
    using glm::vec3;

    // Create manually defined AABBs
    std::vector<Bound<3, float>> boxes;

    // AABB 0: from (0, 0, 0) to (1, 1, 1)
    boxes.emplace_back(vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f));

    // AABB 1: overlaps with AABB 0
    boxes.emplace_back(vec3(1.001f, 1.001f, 1.001f), vec3(2.0f, 2.0f, 2.0f));

    // AABB 2: separate
    boxes.emplace_back(vec3(2.0f, 2.0f, 2.0f), vec3(3.0f, 3.0f, 3.0f));

    // AABB 3: overlaps with AABB 2
    boxes.emplace_back(vec3(2.5f, 2.5f, 2.5f), vec3(3.5f, 3.5f, 3.5f));

    // AABB 4: overlaps with AABB 3
    boxes.emplace_back(vec3(3.0f, 3.0f, 3.0f), vec3(4.0f, 4.0f, 4.0f));

    // AABB 5: fully inside AABB 0
    boxes.emplace_back(vec3(0.2f, 0.2f, 0.2f), vec3(0.8f, 0.8f, 0.8f));

    // AABB 6: overlaps with AABB 1 only
    boxes.emplace_back(vec3(1.2f, 1.2f, 1.2f), vec3(1.4f, 1.4f, 1.4f));

    // AABB 7: separate from all others
    boxes.emplace_back(vec3(10.0f, 10.0f, 10.0f), vec3(11.0f, 11.0f, 11.0f));

    printf("Building LBVH...\n");
    BVHWrapper bvh;
    bvh.build(boxes);
    bvh.selfCheck();

    printf("Querying BVH...\n");
    auto results = bvh.query();

    printf("%zu collision pairs found.\n", results.size());
    for (const auto& p : results)
        printf("(%d %d)\n", p.x, p.y);

    return 0;
}
#pragma once
#include <vector>
#include "KittenEngine/includes/modules/Common.h"
#include "lbvh.cuh"

class BVHWrapper {
public:
    BVHWrapper();
    ~BVHWrapper();

    void build(const std::vector<Kitten::Bound<3, float>>& input);
    std::vector<glm::ivec2> query();
    std::vector<glm::ivec2> queryWith(const BVHWrapper& other);
    
    // NEW: Batched query method
    std::vector<std::vector<glm::ivec2>> queryWithBatched(const std::vector<BVHWrapper*>& others);
	
	void selfCheck();
    void translate(const glm::vec3& offset);

private:
    Kitten::LBVH bvh;
    Kitten::Bound<3, float>* d_bounds = nullptr;
    glm::ivec2* d_result = nullptr;

    size_t num_input = 0;
    size_t max_output = 0;
};
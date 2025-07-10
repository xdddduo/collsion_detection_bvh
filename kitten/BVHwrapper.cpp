#include "BVHwrapper.h"
#include <cuda_runtime.h>

BVHWrapper::BVHWrapper() {}

BVHWrapper::~BVHWrapper() {
    if (d_bounds) cudaFree(d_bounds);
    if (d_result) cudaFree(d_result);
}

void BVHWrapper::selfCheck() {
    bvh.bvhSelfCheck();
}

void BVHWrapper::build(const std::vector<Kitten::Bound<3, float>>& input) {
    num_input = input.size();
    max_output = 100 * num_input;

    cudaMalloc(&d_bounds, sizeof(Kitten::Bound<3, float>) * num_input);
    cudaMemcpy(d_bounds, input.data(), sizeof(Kitten::Bound<3, float>) * num_input, cudaMemcpyHostToDevice);

    cudaMalloc(&d_result, sizeof(glm::ivec2) * max_output);

    bvh.compute(d_bounds, static_cast<int>(num_input));
}

std::vector<glm::ivec2> BVHWrapper::query() {
    int numCols = bvh.query(d_result, max_output);

    std::vector<glm::ivec2> res(numCols);
    cudaMemcpy(res.data(), d_result, sizeof(glm::ivec2) * numCols, cudaMemcpyDeviceToHost);
    return res;
}

std::vector<glm::ivec2> BVHWrapper::queryWith(const BVHWrapper& other) {
    int max_output_this = 100 * num_input;
    glm::ivec2* d_result_cross = nullptr;
    cudaMalloc(&d_result_cross, sizeof(glm::ivec2) * max_output_this);

    int numCols = bvh.query(d_result_cross, max_output_this, const_cast<Kitten::LBVH*>(&other.bvh));

    std::vector<glm::ivec2> res(numCols);
    cudaMemcpy(res.data(), d_result_cross, sizeof(glm::ivec2) * numCols, cudaMemcpyDeviceToHost);
    cudaFree(d_result_cross);

    return res;
}

void BVHWrapper::translate(const glm::vec3& offset) {
    bvh.translate(offset);
}
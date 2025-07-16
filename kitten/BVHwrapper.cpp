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

std::vector<std::vector<glm::ivec2>> BVHWrapper::queryWithBatched(const std::vector<BVHWrapper*>& others) {
    if (others.empty()) return {};
    
    const int k = static_cast<int>(others.size());
    const int maxResPerTest = static_cast<int>(max_output);
    
    // Allocate device memory for results and counts
    glm::ivec2* d_results = nullptr;
    int* d_result_counts = nullptr;
    
    cudaMalloc(&d_results, sizeof(glm::ivec2) * k * maxResPerTest);
    cudaMalloc(&d_result_counts, sizeof(int) * k);
    
    // Prepare vector of LBVH pointers
    std::vector<Kitten::LBVH*> lbvh_others;
    for (const auto* other : others) {
        lbvh_others.push_back(const_cast<Kitten::LBVH*>(&other->bvh));
    }
    
    // Call the batched query
    bvh.queryBatched(d_results, d_result_counts, maxResPerTest, lbvh_others);
    
    // Copy results back to host
    std::vector<glm::ivec2> all_results(k * maxResPerTest);
    std::vector<int> result_counts(k);
    
    cudaMemcpy(all_results.data(), d_results, sizeof(glm::ivec2) * k * maxResPerTest, cudaMemcpyDeviceToHost);
    cudaMemcpy(result_counts.data(), d_result_counts, sizeof(int) * k, cudaMemcpyDeviceToHost);
    
    // Format results into vector of vectors
    std::vector<std::vector<glm::ivec2>> formatted_results;
    for (int i = 0; i < k; i++) {
        std::vector<glm::ivec2> test_results;
        int count = result_counts[i];
        for (int j = 0; j < count; j++) {
            test_results.push_back(all_results[i * maxResPerTest + j]);
        }
        formatted_results.push_back(test_results);
    }
    
    // Cleanup
    cudaFree(d_results);
    cudaFree(d_result_counts);
    
    return formatted_results;
}
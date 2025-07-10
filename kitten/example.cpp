#include <vector>
#include <cstdlib>
#include <cstdio>
#include "lbvh.cuh"

using namespace std;
using namespace Kitten;

int main() {
	const int N = 100000;
	const float R = 0.001f;

	printf("Generating synthetic data...\n");
	vector<Bound<3, float>> points(N);
	srand(1);
	for (size_t i = 0; i < N; i++) {
		Bound<3, float> b(vec3(
			rand() / (float)RAND_MAX,
			rand() / (float)RAND_MAX,
			rand() / (float)RAND_MAX
		));
		b.pad(R);
		points[i] = b;
	}

	// Allocate raw device memory for points
	Bound<3, float>* d_points = nullptr;
	cudaMalloc(&d_points, sizeof(Bound<3, float>) * N);
	cudaMemcpy(d_points, points.data(), sizeof(Bound<3, float>) * N, cudaMemcpyHostToDevice);

	// Allocate raw device memory for results
	ivec2* d_res = nullptr;
	size_t max_res = 100 * N;
	cudaMalloc(&d_res, sizeof(ivec2) * max_res);

	printf("Building LBVH...\n");
	LBVH bvh;
	bvh.compute(d_points, N);

	printf("Querying BVH...\n");
	int numCols = bvh.query(d_res, max_res);

	// Copy results back
	vector<ivec2> res(numCols);
	cudaMemcpy(res.data(), d_res, sizeof(ivec2) * numCols, cudaMemcpyDeviceToHost);

	printf("%d collision pairs found.\n", numCols);
	for (size_t i = 0; i < res.size(); i++)
		printf("(%d %d)\n", res[i].x, res[i].y);

	// Cleanup
	cudaFree(d_points);
	cudaFree(d_res);

	return 0;
}
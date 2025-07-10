#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <utility>

#include "../include/kernel_launcher.h"            // For launch_tridist_kernel
#include "../kitten/BVHwrapper.h"               // KittenGpuLBVH header

namespace py = pybind11;

// ========== 1. CUDA Triangle Collision Forward Binding ========== //

void launch_tridist_kernel(
    const float* triangles1,
    const float* triangles2,
    bool* collisions,
    int B, int T1, int T2,
    float threshold
);

torch::Tensor tridist_forward(
    torch::Tensor triangles1,  // [B, T1, 3, 3]
    torch::Tensor triangles2,  // [B, T2, 3, 3]
    float threshold
) {
    TORCH_CHECK(triangles1.is_cuda(), "triangles1 must be a CUDA tensor");
    TORCH_CHECK(triangles2.is_cuda(), "triangles2 must be a CUDA tensor");
    TORCH_CHECK(triangles1.dtype() == torch::kFloat32, "triangles1 must be float32");
    TORCH_CHECK(triangles2.dtype() == torch::kFloat32, "triangles2 must be float32");

    int B = triangles1.size(0);
    int T1 = triangles1.size(1);
    int T2 = triangles2.size(1);

    auto collisions = torch::zeros({B}, torch::dtype(torch::kBool).device(triangles1.device()));

    launch_tridist_kernel(
        triangles1.contiguous().data_ptr<float>(),
        triangles2.contiguous().data_ptr<float>(),
        collisions.data_ptr<bool>(),
        B, T1, T2, threshold
    );

    return collisions;
}

// ========== 2. KittenGpuLBVH BVHWrapper Binding ========== //

py::class_<BVHWrapper> bind_bvh(py::module &m) {
    return py::class_<BVHWrapper>(m, "BVHWrapper")
        .def(py::init<>())

        .def("build", [](BVHWrapper &self, const std::vector<std::pair<std::array<float, 3>, std::array<float, 3>>> &aabb_list) {
            std::vector<Kitten::Bound<3, float>> boxes;
            for (const auto& box : aabb_list) {
                boxes.emplace_back(
                    glm::vec3(box.first[0], box.first[1], box.first[2]),
                    glm::vec3(box.second[0], box.second[1], box.second[2])
                );
            }
            self.build(boxes);
        })

        .def("query", [](BVHWrapper &self) {
            std::vector<glm::ivec2> raw = self.query();
            std::vector<std::pair<int, int>> out;
            for (const auto &p : raw) {
                out.emplace_back(p.x, p.y);
            }
            return out;
        })

        .def("query_with", [](BVHWrapper &self, const BVHWrapper &other) {
            std::vector<glm::ivec2> raw = self.queryWith(other);
            std::vector<std::pair<int, int>> out;
            for (const auto &p : raw) {
                out.emplace_back(p.x, p.y);
            }
            return out;
        })

        .def("translate", [](BVHWrapper& self, std::vector<float> offset) {
            if (offset.size() != 3) throw std::runtime_error("Offset must be a 3D vector");
            self.translate(glm::vec3(offset[0], offset[1], offset[2]));
        })

        .def("selfCheck", &BVHWrapper::selfCheck);
}

// ========== 3. Python Module Entry ========== //

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tridist_forward", &tridist_forward, "Triangle collision detection (CUDA)");
    bind_bvh(m);
}
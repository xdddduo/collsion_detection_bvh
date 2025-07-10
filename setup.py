from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='collision_cuda',
    ext_modules=[
        CUDAExtension(
            name='collision_cuda',
            sources=[
                'python/collision.cpp',               # pybind11 + kernel entry
                'triangle/triangle_check.cu',         # your triangle collision CUDA kernel

                # 🟢 KittenGpuLBVH source files
                'kitten/BVHwrapper.cpp',
                'kitten/lbvh.cu',

                # 🔴 Optional: Add bvh_builder.cu, etc., if needed later
                # 'bvh/bvh_builder.cu',
                # 'bvh/bvh_traversal.cu',
            ],
            include_dirs=[
                'include',
                'triangle',
                'bvh',
                'kitten',  # KittenGpuLBVH headers like BVHwrapper.h, lbvh.cuh
                # Or custom glm path:
                # '/home/xiuping.zhu/libs/glm'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
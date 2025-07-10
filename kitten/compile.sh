# nvcc -std=c++17 -O2 \
#   --expt-relaxed-constexpr \
#   --extended-lambda \
#   -I./glm \
#   -I./kitten \
#   -I./KittenEngine/includes \
#   main.cpp lbvh.cu -o test_query


nvcc -std=c++17 -O2 \
  --expt-relaxed-constexpr \
  --extended-lambda \
  --cudart=shared \
  -I./glm \
  -I./kitten \
  -I./KittenEngine/includes \
  /hpc_stor03/sjtu_home/xiuping.zhu/courses/ece450/design/KittenGpuLBVH/BVHwrapper.cpp test.cpp lbvh.cu -o test
nvcc -std=c++17 -O2 \
  --expt-relaxed-constexpr \
  --extended-lambda \
  -I./glm \
  --cudart=shared \
  -I./kitten \
  -I./KittenEngine/includes \
  example.cpp lbvh.cu -o example
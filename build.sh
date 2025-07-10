#!/bin/bash

echo "🧹 Cleaning previous build artifacts..."

rm -rf build
rm -rf python/__pycache__/
rm -rf *.egg-info
find . -name "*.so" -delete
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -r {} +

echo "🔨 Building CUDA extension from root..."
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully."
else
    echo "❌ Build failed."
    exit 1
fi
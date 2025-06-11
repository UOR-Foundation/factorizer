#!/bin/bash
# Run the 64-bit factorizer benchmark with various configurations

echo "============================================="
echo "UOR/Prime Axioms Factorizer - 64-bit Benchmark"
echo "============================================="
echo ""

# Create results directory
mkdir -p benchmark_results
cd benchmark_results

# Get timestamp for unique filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Running benchmark WITH learning/acceleration..."
python ../factorizer_64bit_benchmark.py \
    --seed 42 \
    --output "factorizer_64bit_with_learning_${TIMESTAMP}.json"

echo ""
echo "Running benchmark WITHOUT learning/acceleration..."
python ../factorizer_64bit_benchmark.py \
    --no-learning \
    --seed 42 \
    --output "factorizer_64bit_no_learning_${TIMESTAMP}.json"

echo ""
echo "============================================="
echo "Benchmark complete! Results saved in benchmark_results/"
echo "============================================="

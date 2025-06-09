#!/bin/bash

# Comprehensive benchmark suite runner
# Runs all benchmarking tools and generates complete performance analysis

echo "UOR/Prime Axioms Factorizer - Complete Benchmark Suite"
echo "======================================================"
echo ""

# Create output directory
mkdir -p benchmark_output

# Run core benchmarks
echo "1. Running Core Benchmarks..."
python3 benchmark_runner.py | tee benchmark_output/core_benchmarks.log

echo ""
echo "2. Running Performance Profiler..."
python3 performance_profiler.py | tee benchmark_output/performance_profile.log

echo ""
echo "3. Running Quick Benchmark..."
python3 quick_benchmark.py | tee benchmark_output/quick_benchmark.log

echo ""
echo "4. Generating Summary Report..."

# Create comprehensive summary
cat > benchmark_output/benchmark_summary.md << EOF
# UOR/Prime Axioms Factorizer - Benchmark Summary

Generated on: $(date)

## Benchmark Components

### 1. Core Benchmarks (benchmark_runner.py)
- Tests all 5 axioms on standard test cases
- Measures success rates and execution times
- Provides axiom-by-axiom performance comparison

### 2. Performance Profiling (performance_profiler.py)  
- Detailed system metrics (CPU, memory usage)
- Algorithmic complexity analysis
- Cache performance evaluation
- Scaling behavior analysis

### 3. Quick Benchmark (quick_benchmark.py)
- Rapid performance assessment
- Essential metrics without full testing
- Quick health check for the system

## Key Findings

$(if [ -f benchmark_report.txt ]; then echo "### Core Benchmark Results"; tail -n 20 benchmark_report.txt; fi)

$(if [ -f performance_results.txt ]; then echo "### Performance Analysis"; tail -n 15 performance_results.txt; fi)

## Files Generated
- Core benchmarks: benchmark_report.txt
- Performance details: performance_results.txt  
- Execution logs: benchmark_output/*.log

## Recommendations
1. Review any axioms with success rates below 50%
2. Monitor memory usage patterns for optimization opportunities
3. Consider caching optimizations for frequently used operations

EOF

echo "Benchmark Summary created: benchmark_output/benchmark_summary.md"
echo ""
echo "All benchmarks completed!"
echo "Check the benchmark_output/ directory for detailed results."

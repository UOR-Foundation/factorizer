#!/bin/bash

# Run 64-bit benchmarks for Universal Ontological Factorizer

echo "Universal Ontological Factorizer - 64-bit Benchmark Suite"
echo "========================================================="
echo

# Check if we're in the right directory
if [ ! -f "factorizer.py" ]; then
    cd ..
    if [ ! -f "factorizer.py" ]; then
        echo "Error: Please run this script from the project root or benchmark directory"
        exit 1
    fi
fi

# Parse command line arguments
QUICK=false
COMPARE=false
COMPREHENSIVE=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK=true
            ;;
        --compare)
            COMPARE=true
            ;;
        --comprehensive)
            COMPREHENSIVE=true
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --quick          Run quick benchmark (default)"
            echo "  --compare        Compare original and optimized implementations"
            echo "  --comprehensive  Run comprehensive benchmark (slower)"
            echo "  --help           Show this help message"
            exit 0
            ;;
    esac
done

# Default to quick if nothing specified
if [ "$QUICK" = false ] && [ "$COMPREHENSIVE" = false ]; then
    QUICK=true
fi

# Run benchmarks
if [ "$QUICK" = true ]; then
    echo "Running quick 64-bit benchmark..."
    echo
    if [ "$COMPARE" = true ]; then
        python benchmark/benchmark_64bit_quick.py --compare
    else
        python benchmark/benchmark_64bit_quick.py
    fi
fi

if [ "$COMPREHENSIVE" = true ]; then
    echo
    echo "Running comprehensive 64-bit benchmark..."
    echo "(This may take several minutes)"
    echo
    python benchmark/benchmark_64bit_comprehensive.py
fi

echo
echo "Benchmark complete!"

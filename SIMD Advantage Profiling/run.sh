#!/bin/bash

OUTPUT_FILE="all_results.csv"
EXECUTABLE="kernal_code/kernel"
VECTORIZATION_REPORT="vectorization_report.txt"

# Mandatory Compiler Flags
COMPILER_FLAGS="-std=c++11 -O3 -march=native -fopenmp -fopt-info-vec-all -ftree-vectorizer-verbose=1"

# Clean previous results and reports before starting
rm -f $OUTPUT_FILE $VECTORIZATION_REPORT

# Build the C++ kernel and pipe the vectorization report to a file
echo "Compiling C++ kernel with vectorization analysis..."
g++ $COMPILER_FLAGS kernal_code/kernel.cpp -o $EXECUTABLE 2> $VECTORIZATION_REPORT
if [ $? -ne 0 ]; then
    echo "ERROR: C++ compilation failed."
    exit 1
fi

echo "Vectorization report saved to $VECTORIZATION_REPORT"

# Run 5 Times for Error Bar Statistics
NUM_RUNS=5
echo "Running $NUM_RUNS experiments to gather statistics..."

for i in $(seq 1 $NUM_RUNS); do
    $EXECUTABLE | sed "s/^/$i,/" >> $OUTPUT_FILE
done

echo "Finished $NUM_RUNS runs. Results saved to $OUTPUT_FILE"
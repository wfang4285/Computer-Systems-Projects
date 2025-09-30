#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

// Configuration Constants
#define ITERATIONS 10000; // Increased iterations for stable timing
#define PAGE_SIZE 4096

// Read Time Stamp Counter (TSC) using inline assembly
static inline uint64_t rdtsc() {
    uint32_t hi, lo;
    // Serializing instruction to ensure non-speculative reading
    asm volatile ("cpuid\n" "rdtsc" : "=a" (lo), "=d" (hi) :: "ebx", "ecx");
    return ((uint64_t)hi << 32) | lo;
}

// Global variable to prevent compiler optimizing away the loop
volatile int dummy_read = 0;

// Runs the kernel and returns the total cycles (TSC ticks)
uint64_t run_kernel(long size_elements, int stride) {
    int *arr = (int *)malloc(size_elements * sizeof(int));
    if (!arr) { perror("Malloc failed"); return 0; }

    // Warm-up and Initialization
    for (long j = 0; j < size_elements; j++) arr[j] = j;

    uint64_t start_cycles = rdtsc();

    // The measurement kernel (sequential R/W loop)
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (long j = 0; j < size_elements; j += stride) {
            arr[j] = arr[j] + 1;
            dummy_read = arr[j]; // Use result to avoid optimization
        }
    }

    uint64_t end_cycles = rdtsc();

    free(arr);
    return end_cycles - start_cycles;
}

// Experiment 6: Cache-Miss Impact. Varies the working set size.
void run_cache_tests() {
    printf("// EXPERIMENT 6: CACHE MISS IMPACT (Use perf stat for misses)\n");
    printf("// TestName, Size (MB), Stride (bytes), Cycles\n");

    long sizes[] = {
        32 * 1024,
        512 * 1024,
        8 * 1024 * 1024,
        32 * 1024 * 1024
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int STRIDE = 1;

    for (int i = 0; i < num_sizes; i++) {
        long size_bytes = sizes[i];
        long size_elements = size_bytes / sizeof(int);
        uint64_t cycles = run_kernel(size_elements, STRIDE);

        printf("E6_CACHE_SEQ, %ld, %ld, %llu\n",
               size_bytes / (1024 * 1024),
               STRIDE * sizeof(int),
               cycles
        );
        sleep(1);
    }
}

// Experiment 7: TLB-Miss Impact. Uses a page-sized stride to force DTLB misses.
void run_tlb_tests() {
    printf("\n// EXPERIMENT 7: TLB MISS IMPACT (Use perf stat for misses)\n");
    printf("// TestName, Size (MB), Stride (bytes), Cycles\n");

    long sizes_mb[] = {
        1, 2, 4, 8, 16
    };
    int num_sizes = sizeof(sizes_mb) / sizeof(sizes_mb[0]);
    int STRIDE = PAGE_SIZE / sizeof(int); // Stride is 4096 bytes

    for (int i = 0; i < num_sizes; i++) {
        long size_elements = sizes_mb[i] * 1024 * 1024 / sizeof(int);
        uint64_t cycles = run_kernel(size_elements, STRIDE);

        printf("E7_TLB_PAGESTRIDE, %ld, %ld, %llu\n",
               sizes_mb[i],
               STRIDE * sizeof(int),
               cycles
        );
        sleep(1);
    }
}


int main() {
    printf("// STARTING PERFORMANCE TESTS (TSC TIMER)\n\n");

    run_cache_tests();
    run_tlb_tests();

    printf("\n// END OF TESTS. Use perf stat to collect miss data.\n");
    return 0;
}

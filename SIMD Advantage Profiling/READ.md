# Project One 

## 1. Baseline & Correctness 

- Timing was validation against SIMD results using 10^-6 error tolerance.
  - See kernal.cpp code.
- Multiple repetitions used and median was chosen.
  - Graphs contain error bars/ ranges. 

## 2. Vectorization Verification 

- Vectorization report txt file outputted showing lines vectorized successfully.
  - Lines 136, 164, 185, 121, 143, and 171.
- Plots show 8X speed up for the float32 confirms the 8 lanes of the AVX2 instruction set used. 

## 3. Locality Sweep

- Plot indicates L1/L2 transition around 10^5 array size with slight drop. DRAM transition around 10^6 to 10^7 show a sharper drop.
- The 8X speed up is only possible when data is instantly accessed from the cache. As it becomes DRAM-bound speedup drops due to data access delays.
  - Vectorization doesn't solve this memory bottleneck.

## 4. Alignment & Tail Study 

- Plot shows that unaligned load is nearly as fast as aligned load because CPUs are very efficient. 
- There is a noticeable increase in execution time when the size decreases to 8191 due being misaligned.
  - 8191 is not a multiple of 8 so a secondary loop must clean up the elements using scalar instructions.

## 5. Stride/ Gather Effects 

- Plot shows performance sharply dropping off after S = 1. 
  - Vector elements are no long a single whole in memory after S = 1 forcing constantly keeping of track of each "node."

## 6. Data Type Comparison

- Float64 results show near zero deviation from the baseline none sped-up results. 
  - Float64 only having 4 lanes causes this lack of speed due to the high overhead of reduction.

## 7. Speedup & Throughput Plots

- NA; see plots folder. 

## 8. Roofline Analysis 

- Elements 
  - Peak Compute Limit: 1228.8 GFLOP/s
  - DRAM Bandwidth: 20GiB/s
  - SAXPY: 0.167 FLOP/Byte
  - Dot Product: 0.25 FLOP/Byte
  - Element Multiplication: 0.083 FLOP/Byte

- Conclusions 
  - SAXPY and Element Multiplication are memory-bound. Performance falls on or below the slanted memory roofline. It would not improve with further vectorizaiton due to speed of data transfer. 
  - Dot Product is compute-bound. Its DRAM performance is 4.0 GFLOP/s near the memory ceiling but its peak cache performance is compute-bound and falls far below the theoretical limit.

## 9. Reporting Quality 

- NA; see plots folder and document.
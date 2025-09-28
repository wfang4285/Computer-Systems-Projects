# Computer-Systems-Projects

## CPU Info

CPU: 11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz__
Architecture: x86_64, 8 Physical Cores, 16 Threads__
SIMD Support: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX-512F, AVX-512DQ, AVX-512BW, AVX-512VL, FMA__
Caches:__
  - L1d: 384 KiB (8 instances)__
  - L1i: 256 KiB (8 instances)__
  - L2: 10 MiB (8 instances)__
  - L3: 24 MiB (1 instance)__
NUMA Nodes: 1__ 

## System Info 

OS: Ubuntu 22.04__
Complier: GCC 11.4.8__
Complier Flags:__
  - Scalar baseline: -O0__
  - SIMD auto-vectorized: -O3 -march=native -ffast-math -fno-signed-zeros__
Timing: std::chrono, median of ≥3 runs, error bars included__

CPU Frequency: fixed using Performance governor__
SMT / Hyperthreading: Enabled, experiments run single-threaded


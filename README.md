# Computer-Systems-Projects

## CPU Info

CPU: 11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz
Architecture: x86_64, 8 Physical Cores, 16 Threads 
SIMD Support: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX-512F, AVX-512DQ, AVX-512BW, AVX-512VL, FMA
Caches:
  L1d: 384 KiB (8 instances)
  L1i: 256 KiB (8 instances)
  L2: 10 MiB (8 instances)
  L3: 24 MiB (1 instance)
NUMA Nodes: 1 

## System Info 

OS: Ubuntu 22.04
Complier: GCC 11.4.8
Complier Flags:
  - Scalar baseline: -O0
  - SIMD auto-vectorized: -O3 -march=native -ffast-math -fno-signed-zeros
Timing: std::chrono, median of ≥3 runs, error bars included

CPU Frequency: fixed using Performance governor
SMT / Hyperthreading: Enabled, experiments run single-threaded


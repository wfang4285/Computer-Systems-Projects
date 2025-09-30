## Project 2

# 1. Zero-Queue Baselines

- NA; see table in folder.
  - CPU frequency is 2.5GHz.
  - 36.7ns for fastest L1/L2 access and 42,992ns for DRAM. 

# 2. Pattern & Granularity Sweep

- See tables for results (Latency and Bandwidth Comparison).
  - Prefetcher masks the DRAM latency via predicting addresses needed.
  - When memory is accessed randomly the CPU needs to wait for every miss which shows the unmasked DRAM latency.
  - The difference in the graph is the prefetcher advantage.
  - Stride table also shows much superior bandwidth with 64B cache stride with quick drop off with 256 and 1024 strides.

# 3. Read/Write Mix Sweep

- See tables for results.
  - The 100% read is the fastest as data is only being streamed out which requires minimal overhead.
  - Meanwhile, the mix of half read and write commands does nearly as well as the 100% read.
    - This is due to the system being great at interweaving read and write commands. A 50/50 mix allows it to schedule efficiently, hiding latency. 
  - The 75% read and 25% write meanwhile takes a slight hit because it is harder to schedule it ideally resulting in buffer overhead. 
  - 100% write does the worst because write commands are more complex and have more overhead.
    - Writes sometimes need buffers to wait for memory to be ready.

# 4. Intensity Sweep

- See latency vs throughput table for plot and intensity sweep data for printed results from 3 trials.
  - Graph's shape/ curve is a representation of Little's Law.
    - Before the stagnant 28,000 MB/s point the throughput increases rapidly but latency remains stable.
    - But after it the throughput hits a limit and latency rapidly increases if continued. 
  - The knee is placed around 28,000 MB/s of throughput for the reasons above. There is a distinct before and after in the graph.
  - The highest recorded peak bandwidth is 40,463 MB/s, approximately 81% of the theoretical max capacity.
    - After the knee there is diminishing returns of increasing bandwidth that results in rapid latency growth; the scheduling limits make further returns almost zero.

# 5. Working-Set Size Sweep 

- See Cache Cliff graph and transition table.

# 6. Cache-Miss Impact

- For this requirement 0MB, 8MB, and 32MB sizes were used while keeping the stride at 4B. 
  - This controls the miss rate and allowing accurate measurement of pure L1/L2, L3, and then DRAM. 
- The average cycle rapidly increases making L1/L2 almost invisible in the Runtime_Penalty table. 
  - 83 billion cycles additionally consumed from L3 to DRAM relates directly to the L3 misses.
- The 8MB has a low AMAT because the miss rate for L3 is low leading to quicker L3 access.
  - Meanwhile, the 32MB run is obviously higher as DRAM will almost certainly be used due to the L3 misses.

# 7. TLB-Miss Impact

- A 4096B stride was used in the TLB miss impact experiment trials.
  - This ensures every memory access requires a new TLB entry. 
- The data shows a direct correlation of the set size and runtime.
  - The cycles used in each increase of working set size seem to be linear, with each doubling of size leading to double the cycles.
  - After 4MB size this doubling is still largely correct but some additional penalty of cycles is added as well.
- The more than doubling of cycles after 4MB indicates the DTLB limits.
  - Once this limit is exceeded a penalty in cycles occurs from the time being spent on DRAM access for page tables.
## Project 3

# Configuration

- Diskspd v2.2 on Microsoft Powershell
  - C:\test_data\disk_metrics.dat
    - 10GiB Size
  - 60s Duration
  
- Microsoft Windows 11 Education 26100
- 11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz
  - 8 Cores, 16 Logic Processors
- 34048368640 Total Physical Memory
- SAMSUNG MZVL2512HCJQ-00BL7 SSD
  - Capacity: 512110190592

- Powershell Commands
  - Zero Queue Baselines
    - .\diskspd.exe -o1 -t1 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o1 -t1 -r -w100 -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o1 -t1 -s -b128K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o1 -t1 -s -w100 -b128K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
  - Block Size Sweep
    - .\diskspd.exe -o16 -t4 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -r -b16K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -r -b32K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -r -b128K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -r -b256K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -r -b1M -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -s -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -s -b8K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -s -b16K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -s -b32K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -s -b64K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -s -b128K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -s -b256K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -s -b1M -d60 -Sh -D -L C:\test_data\disk_metrics.dat
  - Read/ Write
    - .\diskspd.exe -o16 -t4 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -r -w30 -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -r -w50 -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -r -w100 -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
  - Queue Depth Sweep
    - .\diskspd.exe -o1 -t4 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o2 -t4 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o4 -t4 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o8 -t4 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o16 -t4 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o32 -t4 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat
    - .\diskspd.exe -o64 -t4 -r -b4K -d60 -Sh -D -L C:\test_data\disk_metrics.dat

# 1. Zero Queue Baselines

- QD=1 latency isolated with -o1, -Sh, and -t1. Random parameter explicitly stated for 4 KiB while 128 KiB was sequential instead.
- See zero queue table for overall results and zero queue data for raw results.

# 2. Block Size and Pattern Sweep

- See graphs on IOPS and throughput vs block size and average latency vs block size. 
  - Also see block/ pattern table.

- IOPs dominated region is at the start from around 4K to 32K; throughput is able to rapidly rise.
  - Random I/O is rapidly decaying, peaking at the 4K start while sequential I/O IOPs is able to remain stable longer before a sharp drop off.
  - This is explained by queue coalescing in sequential workloads that combine 4-16K requests into larger reads; latency remains the same.
  - Sequential also has prefetching trying to predict what will be utilized next.
- As block size grows beyond 128K it is bandwidth dominated with data transfer time dominating the latency. 
  - The throughput for both random and sequential plateaus at 128K, with only marginal gains further on at huge cost. 
- Peak IOPS at 74.7K and throughput top of 3.7 GB/s shows the controller limits.

# 3. Read/ Write Mix Sweep

- See read/ write data file, table, and chart for information.
- The 100% write IOPs is superior in performance to the 100% read IOPs, which may be due to burst performance.
  - Write buffering on the SSD controller may explain most of the superior performance. SSDs contain high speed volatile memory; the SSD controller could immediately copy the data into the fast internal buffer. 
  - Meanwhile, write buffering makes things more complicated; writing new data to an already used page requires the controller to read the entire block, modify, and write a whole new block to a new location.
  - These two result in flushes where the controller will organize through data, do garbage collection, and flush the data to make sure the fast cache is not filled up.
    - This is seen in the 99th percentile write latency being 1.55ms compared to read's 0.231ms.

# 4. Queue-Depth and Parallelism Sweep

- See Queue data folder, QD chart, and QD summary table for information.
- From QD 4 to 16 there is rapid growth that then hits a plateau at QD 32, the knee point.
  - This can be justified with Little Law's with QD 32's IOPs times latency being 18.42, severely divergent from the actual number.
  - Throughput after QD 32 decreases slightly. 
- QD 4 average latency is 0.091ms and 99th percentile latency is 0.141ms. Meanwhile, at QD 32 it's 0.115ms and 0.213ms respectively.
  - The average latency is only 26% more between QD 4 and QD 32 while the 99th percentile latency is 51% higher, showing some degrading at the knee. 
  - QD 32 is the point of diminishing/ no returns as there is minimal to zero performance gains. 


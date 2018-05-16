Goal of the project : implement an LDPC decoder on a gpu

Tasks :

1. Benchmark the input/output performances
  - Test the throughput between GPU and CPU
  - Test the latency of the GPU (because LDPC decoder should be real-time)

Deliverables : 
  - a benchmark procedure
  - a presentation of the results of the benchmark

2. Implement the C LDPC algorithm on GPU using CUDA
  - use massive parallel computaton capacities of the GPU to speed up algorithm
  - minimize the data transfer between CPU and GPU 
  
The target throughput of the all application is 280 Mb/s * k (k the number of channels).

Documentary ressources :

For benchmarking the data transfer : https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
For optimizinf the data transfers : https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/

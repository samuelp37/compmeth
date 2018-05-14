#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"

__global__ void someKernel(int N)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx<N)
		printf("Hello from thread # %i (block #: %i)\n", idx, blockIdx.x);
}

extern void cuda_doStuff(void)
{
	int numberOfBlocks = 2;
	int threadsPerBlock = 5;
	int maxNumberOfThreads = 10;
	someKernel <<< numberOfBlocks, threadsPerBlock >>>(maxNumberOfThreads);
	cudaDeviceSynchronize();
}
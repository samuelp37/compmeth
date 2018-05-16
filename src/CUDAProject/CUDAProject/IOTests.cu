#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"

extern void cuda_upload(int *a, int N){

	int size = N*sizeof(int);
	int *d_a;
	cudaMalloc((void **)&d_a, size);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaFree(d_a);

}
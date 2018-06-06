#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"

__global__ void add_simple(int *a, int *b, int *c)
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void add_advanced(int *a, int *b, int *c,int N)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < N){
		c[index] = a[index] + b[index];
	}
}

extern void cuda_computeAddition(int *a, int *b, int *c,int N)
{
	int size = N*sizeof(int);
	int *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a,size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add_simple<<<N,1>>>(d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

}

extern void cuda_computeAddition_advanced(int *a, int *b, int *c, int N,int M)
{
	int size = N*sizeof(int);
	int *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add_advanced <<<N/M+1, M >> >(d_a, d_b, d_c,N);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

}
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"

__global__ void add_simple(int *a, int *b, int *c);

__global__ void add_advanced(int *a, int *b, int *c, int N);

extern void cuda_computeAddition(int *a, int *b, int *c, int N);

extern void cuda_computeAddition_advanced(int *a, int *b, int *c, int N, int M);
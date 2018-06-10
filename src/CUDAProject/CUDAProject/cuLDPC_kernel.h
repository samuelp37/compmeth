#ifndef LDPC_CUDA_KERNEL_CU
#define LDPC_CUDA_KERNEL_CU

// constant memory
//__device__ __constant__ int  dev_h_base[H_MATRIX];
__device__ __constant__ h_element dev_h_compact1[H_COMPACT1_COL][H_COMPACT1_ROW];  // used in kernel 1
__device__ __constant__ h_element dev_h_compact2[H_COMPACT2_ROW][H_COMPACT2_COL];  // used in kernel 2

// For cnp kernel
#if MODE == WIMAX
__device__ __constant__ char h_element_count1[BLK_ROW] = { 6, 7, 7, 6, 6, 7, 6, 6, 7, 6, 6, 6 };
__device__ __constant__ char h_element_count2[BLK_COL] = { 3, 3, 6, 3, 3, 6, 3, 6, 3, 6, 3, 6, \
3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
#else
__device__ __constant__ char h_element_count1[BLK_ROW] = { 7, 7, 7, 7, 7, 7, 8, 7, 7, 7, 7, 8 };
__device__ __constant__ char h_element_count2[BLK_COL] = { 11, 4, 3, 3, 11, 3, 3, 3, 11, 3, 3, 3, \
3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
#endif

__device__ float F_FUCN_MIN_SUM_DEV(float a, float b);
__global__ void ldpc_cnp_kernel_1st_iter(float * dev_llr, float * dev_dt, float * dev_R, int * dev_et);
__global__ void ldpc_cnp_kernel(float * dev_llr, float * dev_dt, float * dev_R, int * dev_et, int threadsPerBlock);
__global__ void ldpc_vnp_kernel_normal(float * dev_llr, float * dev_dt, int * dev_et);
__global__ void ldpc_vnp_kernel_last_iter(float * dev_llr, float * dev_dt, int * dev_hd, int * dev_et);
__global__ void ldpc_decoder_kernel3_early_termination(int * dev_hd, int * dev_et);
__global__ void conversion_Q8_float(float * dev_llr_float, char * dev_llr);

#endif
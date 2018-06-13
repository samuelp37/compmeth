CC=/usr/local/cuda-9.2/bin/nvcc
GCC_OPT=-O3
NVCC_FLAGS=-w #do not print warnings
NVCC_OPT=-Xptxas -O3
NVCC_LDFLAGS=-I ~/NVIDIA_CUDA-9.2_Samples/common/inc


all: LDPC_decoder 

LDPC_decoder: *.cu *.cpp *.h
	$(CC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) *.cu *.cpp -o LDPC_decoder

LDPC_optimized: *.cu *.cpp *.h
	$(CC) $(GCC_OPT) $(NVCC_OPT) $(NVCC_FLAGS) $(NVCC_LDFLAGS) *.cu *.cpp -o LDPC_optimized

clean: 
	rm -rf LDPC_decoder
	rm -rf LDPC_optimized


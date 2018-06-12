all: LDPC_decoder 

LDPC_decoder: *.cu *.cpp *.h
	/usr/local/cuda-9.2/bin/nvcc -I ~/NVIDIA_CUDA-9.2_Samples/common/inc *.cu *.cpp -o LDPC_decoder

clean: 
	rm -rf LDPC_decoder

all: output

output: *.cu *.cpp *.h
	/usr/local/cuda-9.2/bin/nvcc -I ~/NVIDIA_CUDA-9.2_Samples/common/inc *.cu *.cpp -o LDPC_decoder

mrporper: 
	rm -rf LDPC_decoder

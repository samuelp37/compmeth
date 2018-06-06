all: output

output: *.cu *.cpp
	nvcc -I ~/NVIDIA_CUDA-9.2_Samples/common/inc *.cu *.cpp -o output

mrporper: 
	rm -rf output

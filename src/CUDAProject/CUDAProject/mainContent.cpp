#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "random_generator.h"
#include "vectorized_operation.h"
#include "IOTests.h"
#include "cuLDPC.h"

#define N 3580 // size of the vector	
#define M 512 // threads per blocks

void perform_addition(){
	init_random();

	int *a, *b, *c;
	int size = N*sizeof(int);
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);
	random_ints(a, N);
	random_ints(b, N);
	printf("a0=%d\n", a[0]);
	printf("a1=%d\n", a[1]);
	printf("a2=%d\n", a[2]);

	printf("b0=%d\n", b[0]);
	printf("b1=%d\n", b[1]);
	printf("b2=%d\n", b[2]);

	//cuda_computeAddition(a, b, c, N);
	cuda_computeAddition_advanced(a, b, c, N, M);

	printf("c0=%d\n", c[0]);
	printf("c1=%d\n", c[1]);
	printf("c2=%d\n", c[2]);

	free(a); free(b); free(c);
}

void perform_IOTests(){

	int *a;
	int size;

	for (int nb = 1; nb < 1000000000; nb = nb * 2){

		size = nb*sizeof(int);
		a = (int *)malloc(size);
		random_ints(a, nb);
		clock_t t;
		t = clock();
		cuda_upload(a, nb);
		t = clock() - t;
		double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds

		FILE *f = fopen("result.txt", "a");
		printf("%d: %f\n", nb, time_taken);
		fprintf(f, "%d: %f\n", nb,time_taken);

		fclose(f);

		free(a);
	}

}

void advanced_IOTests(){

	FILE *f = fopen("result_without_malloc.txt", "a");
	printf("Bytes: Upload time (s) : Download time (s)\n");
	fprintf(f,"Bytes: Upload time (s) : Download time (s)\n");
	fclose(f);

	for (int nb = 1; nb < 10000000000; nb = nb * 2){

		const unsigned int bytes = nb * sizeof(int);
		int *h_a = (int*)malloc(bytes);
		int *d_a;
		cudaMalloc((int**)&d_a, bytes);

		memset(h_a, 0, bytes);
		clock_t t;
		t = clock();
		cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
		t = clock() - t;
		double time_upload = ((double)t) / CLOCKS_PER_SEC; // in seconds

		t = clock();
		cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
		t = clock() - t;
		double time_download = ((double)t) / CLOCKS_PER_SEC; // in seconds

		FILE *f = fopen("result_without_malloc.txt", "a");
		printf("%d: %f : %f\n", bytes, time_upload, time_download);
		fprintf(f, "%d: %f : %f\n", bytes, time_upload,time_download);
		fclose(f);

		cudaFree(d_a);
		free(h_a);
	}
	
}

void advanced_pinned_IOTests(){

	FILE *f = fopen("result_pinned.txt", "a");
	fprintf(f, "Advanced IO Test with pinned memory\n");
	printf("Bytes: Upload time (s) : Download time (s) : Upload time pinned (s) : Download time pinned (s)\n");
	fprintf(f, "Bytes: Upload time (s) : Download time (s) : Upload time pinned (s) : Download time pinned (s)\n");
	fclose(f);

	for (int nb = 1000; nb < 300000000; nb = nb + 5000000){

		const unsigned int bytes = nb * sizeof(int);
		int *h_aPinned;
		int *d_a, *h_a;

		h_a = (int*)malloc(bytes);
		cudaMallocHost((void**)&h_aPinned, bytes);
		cudaMalloc((void**)&d_a, bytes);

		for (int i = 0; i < nb; ++i) h_a[i] = i;
		memcpy(h_aPinned, h_a, bytes);

		clock_t t;
		t = clock();
		cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
		t = clock() - t;
		double time_upload = ((double)t) / CLOCKS_PER_SEC; // in seconds

		t = clock();
		cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
		t = clock() - t;
		double time_download = ((double)t) / CLOCKS_PER_SEC; // in seconds

		t = clock();
		cudaMemcpy(d_a, h_aPinned, bytes, cudaMemcpyHostToDevice);
		t = clock() - t;
		double time_upload_pinned = ((double)t) / CLOCKS_PER_SEC; // in seconds

		t = clock();
		cudaMemcpy(h_aPinned, d_a, bytes, cudaMemcpyDeviceToHost);
		t = clock() - t;
		double time_download_pinned = ((double)t) / CLOCKS_PER_SEC; // in seconds

		FILE *f = fopen("result_pinned.txt", "a");
		printf("%d: %f : %f : %f : %f\n", bytes, time_upload, time_download, time_upload_pinned, time_download_pinned);
		fprintf(f, "%d: %f : %f : %f : %f\n", bytes, time_upload, time_download, time_upload_pinned, time_download_pinned);
		fclose(f);

		free(h_a);
		cudaFreeHost(h_aPinned);
		cudaFree(d_a);
	}

}

void testLDPC(){
	mainOfLDPC();
}
#include <stdio.h>
#include <stdlib.h>
#include "random_generator.h"

#define N 3580 // size of the vector	
#define M 512 // threads per blocks

extern void cuda_computeAddition(int *a, int *b, int *c, int n);
extern void cuda_computeAddition_advanced(int *a, int *b, int *c, int n, int m);

int main(int argc, const char* argv[])
{

	init_random();

	int *a, *b, *c;
	int size = N*sizeof(int);
	a = (int *) malloc(size);
	b = (int *) malloc(size);
	c = (int *) malloc(size);
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
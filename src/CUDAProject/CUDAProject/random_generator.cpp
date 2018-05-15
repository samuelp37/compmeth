#include <stdlib.h>
#include <time.h>

void init_random(){
	srand(time(NULL));
}

void random_ints(int *a, int n)
{
	int i;
	for (i = 0; i < n; i++)
		a[i] = rand();
}
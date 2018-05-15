#include <stdlib.h>

void random_ints(int *a, int n)
{
	int i;
	for (i = 0; i < n; i++)
		a[i] = rand();
}
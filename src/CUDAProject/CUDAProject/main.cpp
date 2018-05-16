#include "mainContent.h"

extern void cuda_computeAddition(int *a, int *b, int *c, int n);
extern void cuda_computeAddition_advanced(int *a, int *b, int *c, int n, int m);

int main(int argc, const char* argv[])
{
	advanced_IOTests();
}
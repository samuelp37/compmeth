#include <stdio.h>

extern void cuda_doStuff(void);

int main(int argc, const char* argv[])
{
	cuda_doStuff();
}
/*	Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
All rights reserved. https://github.com/robertwgh/cuLDPC

CUDA implementation of LDPC decoding algorithm.
Created: 	10/1/2010
Revision:	08/01/2013
/4/20/2016 prepare for release on Github.
*/

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include "helper_cuda.h"

// custom header file
#include "cuLDPC.h"

extern "C"
{
	void structure_encode(int s[], int code[], int h[BLK_ROW][BLK_COL]);
	void info_gen(int info_bin[]);
	void modulation(int code[], float trans[]);
	void awgn(float trans[], float recv[]);
	void error_check(float trans[], float recv[]);
	void llr_init(float llr[], float recv[]);
	void llr_init_Q8(char llr[], float recv[]);
	int parity_check(float app[]);
	error_result cuda_error_check(int info[], int hard_decision[]);
};


// Variable declaration
extern "C" float sigma;
extern "C" int *info_bin;
extern "C" FILE * gfp;

//===================================
// Random info data generation
//===================================
void info_gen(int info_bin[])
{
	int i;
	// random number generation
	for (i = 0; i < INFO_LEN; i++)
		info_bin[i] = (rand()) % 2;
}

//===================================
// BPSK modulation
//===================================
void modulation(int code[], float trans[])
{
	int i;
	for (i = 0; i < CODEWORD_LEN; i++)
		if (code[i] == 0)
			trans[i] = 1.0;
		else
			trans[i] = -1.0;

}

//===================================
// AWGN modulation
//===================================
void awgn(float trans[], float recv[])
{
	float u1, u2, s, noise, randmum;
	int i;

	for (i = 0; i< CODEWORD_LEN; i++)
	{
		do
		{
#ifdef _WIN32
			randmum = (float)(rand()) / RAND_MAX;
#else
			randmum = drand48();
#endif
			u1 = randmum*2.0f - 1.0f;

#ifdef _WIN32
			randmum = (float)(rand()) / RAND_MAX;
#else
			randmum = drand48();
#endif
			u2 = randmum*2.0f - 1.0f;
			s = u1*u1 + u2*u2;
		} while (s >= 1);
		noise = u1 * sqrt((-2.0f*log(s)) / s);

#ifdef NONOISE
		recv[i] = trans[i];
#else
		recv[i] = trans[i] + noise * sigma;
#endif 
	}
}

//===================================
// data error checking
//===================================
void error_check(float trans[], float recv[])
{
	int i, cnt = 0;
	for (i = 0; i < CODEWORD_LEN; i++)
	{
		if (recv[i] * trans[i] < 0)
		{
			cnt++;
#if PRINT_MSG == 1
			fprintf(gfp, "Transmit error at %d: transmitted %f - received %f \n", i, trans[i], recv[i]);
#endif
		}
	}
#if PRINT_MSG == 1
	fprintf(gfp, "###############################################################\n");
	fprintf(gfp, "Total error is %d, percentage is %f%%\n", cnt, (float)(cnt * 100) / CODEWORD_LEN);
	fprintf(gfp, "###############################################################\n");
#endif
}


//===================================
// calc LLRs
//===================================
void llr_init(float llr[], float recv[])
{
	int i;
#if PRINT_MSG == 1
	FILE * fp;
#endif
	float llr_rev;

#if PRINT_MSG == 1
	fp = fopen("llr_fp.dat", "w");
#endif

	for (i = 0; i < CODEWORD_LEN; i++)
	{
		llr_rev = (recv[i] * 2) / (sigma*sigma);	// 2r/sigma^2 ;
		llr[i] = llr_rev;
#if PRINT_MSG == 1
		fprintf(fp, "recv[%d] = %f, LLR [%d] = %f\n", i, recv[i], i, llr[i]);
#endif
	}
#if PRINT_MSG == 1
	fclose(fp);
#endif
}


//===================================
// calc LLRs_Q8
//===================================
void llr_init_Q8(char llr[], float recv[])
{
	int i;
#if PRINT_MSG == 1
	FILE * fp;
#endif
	float llr_rev;

#if PRINT_MSG == 1
	fp = fopen("llr_fp.dat", "w");
#endif

	for (i = 0; i < CODEWORD_LEN; i++)
	{
		llr_rev = (recv[i] * 2) / (sigma*sigma);	// 2r/sigma^2 ;
		llr[i] = (char) llr_rev*(pow(2,8));
#if PRINT_MSG == 1
		fprintf(fp, "recv[%d] = %f, LLR [%d] = %f\n", i, recv[i], i, llr[i]);
#endif
	}
#if PRINT_MSG == 1
	fclose(fp);
#endif
}

//===================================
// parity check
//===================================
int parity_check(float app[])
{
	int * hbit = (int *)malloc(COL * sizeof(int));
	int error = 0;
	int i;

	// hard decision
	for (i = 0; i< INFO_LEN; i++)
	{
		if (app[i] >= 0)
			hbit[i] = 0;
		else
			hbit[i] = 1;
	}

	for (i = 0; i< INFO_LEN; i++)
	{
		if (hbit[i] != info_bin[i])
			error++;
	}
	//#if PRINT_MSG == 1
	//	fprintf(gfp, "After %d iteration, it has error %d\n", iter, error) ; 
	//#endif
	free(hbit);
	return error;
}


//===================================
// parity check
//===================================
error_result cuda_error_check(int info[], int hard_decision[])
{
	error_result this_error;
	this_error.bit_error = 0;
	this_error.frame_error = 0;

	int bit_error = 0;
	int frame_error = 0;
	int * hard_decision_t = 0;
	int * info_t = 0;

	for (int i = 0; i< CW * MCW; i++)
	{
		bit_error = 0;
		hard_decision_t = hard_decision + i * CODEWORD_LEN;
		info_t = info + i * INFO_LEN;
		for (int j = 0; j < INFO_LEN; j++)
		{
			if (info_t[j] != hard_decision_t[j])
				bit_error++;
		}
		if (bit_error != 0)
			frame_error++;
		this_error.bit_error += bit_error;
	}
	this_error.frame_error = frame_error;

	return this_error;
}


//===================================
// encoding
//===================================
void structure_encode(int s[], int code[], int h[BLK_ROW][BLK_COL])
{
	int i, j, k, sk, jj, id;
	int x[BLK_INFO][Z] = { 0 };
	int sum_x[Z] = { 0 };

	int shift;


	int p0[Z], p1[Z], p2[Z], p3[Z], p4[Z], p5[Z], p6[Z], p7[Z], p8[Z],
		p9[Z], p10[Z], pp[Z];

	for (i = 0; i <BLK_INFO; i++)
		for (j = 0; j< BLK_INFO; j++)
		{
		shift = h[i][j];
		if (shift >= 0)
		{
			for (k = 0; k < Z; k++)
			{
				sk = (k + shift) % Z; //Circular shifting, find the position for 1 in each sub-matrix
				jj = j * Z + sk; // calculate the index in the info sequence
				x[i][k] = (x[i][k] + s[jj]) % 2;  // block matrix multiplication
			}
		}
		}

	for (i = 0; i < Z; i++)
		for (j = 0; j < BLK_INFO; j++)
			sum_x[i] = (x[j][i] + sum_x[i]) % 2;

	id = INFO_LEN;

	// p0	
	for (i = 0; i < Z; i++)
		code[id++] = p0[i] = sum_x[i]; // why p0 = sum??

	shift = h[0][BLK_INFO];
	// h0
	for (i = 0; i < Z; i++)
	{
		if (shift != -1)
		{
			j = (i + shift) % Z;
			pp[i] = p0[j];
		}
		else
			pp[i] = p0[j];
	}

	// p1
	for (i = 0; i < Z; i++)
		code[id++] = p1[i] = (x[0][i] + pp[i]) % 2;

	// p2
	for (i = 0; i < Z; i++)
		code[id++] = p2[i] = (p1[i] + x[1][i]) % 2;

	// p3
	for (i = 0; i < Z; i++)
		code[id++] = p3[i] = (p2[i] + x[2][i]) % 2;


	// p4
	for (i = 0; i < Z; i++)
		code[id++] = p4[i] = (p3[i] + x[3][i]) % 2;

	// p5
	for (i = 0; i < Z; i++)
		code[id++] = p5[i] = (p4[i] + x[4][i]) % 2;

#if MODE == WIMAX
	// p6
	for (i = 0; i < Z; i++)
		code[id++] = p6[i] = (p5[i] + x[5][i] + p0[i]) % 2;

	// p7
	for (i = 0; i < Z; i++)
		code[id++] = p7[i] = (p6[i] + x[6][i]) % 2;
#else
	// p6
	for (i = 0; i < Z; i++)
		code[id++] = p6[i] = (p5[i] + x[5][i]) % 2;

	// p7
	for (i = 0; i < Z; i++)
		code[id++] = p7[i] = (p6[i] + x[6][i] + p0[i]) % 2;
#endif

	// p8
	for (i = 0; i < Z; i++)
		code[id++] = p8[i] = (p7[i] + x[7][i]) % 2;

	// p9
	for (i = 0; i < Z; i++)
		code[id++] = p9[i] = (p8[i] + x[8][i]) % 2;

	// p10
	for (i = 0; i < Z; i++)
		code[id++] = p10[i] = (p9[i] + x[9][i]) % 2;

	// p11
	for (i = 0; i < Z; i++)
		//code [id++] = p11 [i] = (p10 [i] + x[10][i]) % 2 ;
		code[id++] = (p10[i] + x[10][i]) % 2;

	// code word
	for (i = 0; i < INFO_LEN; i++)
		code[i] = s[i];
}

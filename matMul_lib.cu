#include <stdlib.h>
#include <stdio.h>
#include <stdexcept>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <sys/time.h>
#include "cublas_v2.h"

#ifndef N
#define N 32
#endif

using namespace std;

void lib_mmul(float* A, float* B, float* C) {
	
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);

	//switch A and B because this function uses column major order
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha, B, N, A, N, beta, C, N);

	cublasDestroy(handle);

}

int main(){

	size_t size = N * N * sizeof(float);
	struct timeval start_c, end_c, start_t, end_t;
	double time_total, time_calc;

	//host declaration and memory reservation
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size);

	//initialize arrays
	int i,j;
  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      h_A[N*i+j] = (float)(i);
      h_B[N*i+j] = (float)(i+j);
    }
  }

  //device initialization and memory allocation
	float *d_A, *d_B, *d_C;

	gettimeofday(&start_t,NULL);

	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	//copy memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	//setup launch configuration
	dim3 threadsPerBlock(N,N,1);
	dim3 blocksPerGrid(1,1,1);
  threadsPerBlock.x = 32;
  threadsPerBlock.y = 32;
  blocksPerGrid.x = ceil(double(N)/(double)threadsPerBlock.x);
  blocksPerGrid.y = ceil(double(N)/(double)threadsPerBlock.y);

	gettimeofday(&start_c,NULL);

	//kernel call
	lib_mmul(d_A, d_B, d_C);

  cudaDeviceSynchronize();

	gettimeofday(&end_c,NULL);

	//copy back
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	gettimeofday(&end_t,NULL);

  time_calc = ((end_c.tv_sec-start_c.tv_sec)*1.0)+ ((end_c.tv_usec - start_c.tv_usec) / 1000000.0);
  time_total = ((end_t.tv_sec-start_t.tv_sec)*1.0)+ ((end_t.tv_usec - start_t.tv_usec) / 1000000.0);

	//check by sum
	long double exp_sum = (long double)N;
	exp_sum = pow(exp_sum,3.0)*pow(exp_sum-1,2.0)/2.0;

	long double calc_sum = 0;
  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      calc_sum = calc_sum + h_C[N*i+j];
    }
  }

  printf("\nTesting %d x %d Matrix:\n",N,N);
  printf("------------------------------\n");
  printf("Expected Sum:        %Lf\n",exp_sum);
  printf("Calculated Sum:      %Lf\n",calc_sum);
  printf("------------------------------\n");
  printf("Total Time:          %f\n",time_total);  
  printf("Calculation Time:    %f\n\n",time_calc);

	//cleanup
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_C);

	return cudaDeviceSynchronize();
}



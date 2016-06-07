#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#ifndef N
#define N 32
#endif

#ifndef T
#define T 1
#endif

int main(){

	struct timeval start, end;
	size_t size = N * N * sizeof(float);
	clock_t start_c,end_c;
	double calc_time;

	//host declaration and memory reservation
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size);
	float temp = 0.0;

	//initialize arrays
	int i,j,k;
  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      h_A[N*i+j] = (float)(i);
      h_B[N*i+j] = (float)(i+j);
    }
  }

	gettimeofday(&start,NULL);

	//matrix multiplication
	omp_set_num_threads(T);
	#pragma omp parallel shared(h_A,h_B,h_C) private(i,j,k,temp)
	{
		#pragma omp for
		for(i=0; i < N; i++) {
			for(k=0; k < N; k++) {
				for(j=0; j < N; j++) {
					//C[i][j] += A[i][k] * B[k][j];
					temp = h_A[i*N+k] * h_B[k*N+j];
					#pragma omp atomic
					h_C[i*N+j]+=temp;
				}
			}
		}
	}

  gettimeofday(&end,NULL);

  calc_time = ((end.tv_sec-start.tv_sec)*1.0)+ ((end.tv_usec - start.tv_usec) / 1000000.0);

	//check by sum
	long double exp_sum = (long double)N;
	exp_sum = exp_sum*exp_sum*exp_sum*(exp_sum-1)*(exp_sum-1)/2.0;

	long double calc_sum = 0;
	  for(i = 0; i < N; i++){
	    for(j = 0; j < N; j++){
	      calc_sum = calc_sum + h_C[N*i+j];
	    }
	  }

  printf("\nTesting %d x %d Matrix with %d threads:\n",N,N,T);
  printf("------------------------------\n");
  printf("Expected Sum:        %Lf\n",exp_sum);
  printf("Calculated Sum:      %Lf\n",calc_sum);
  printf("------------------------------\n");
  printf("Calculation Time:      %f\n\n",calc_time);

	free(h_A);
	free(h_B);
	free(h_C);

}

/*
To compile this code, first enable a newer version of gcc with

scl enable devtoolset-7 bash

This is needed to support advanced OpenMP features

*/

#include "omp.h"
#define N 128 
#define BS 16
#define EPS 0.000001

#include "stdlib.h"
#include "stdio.h"

void matmul_tasks (float A[N][N], float B[N][N], float C[N][N])
{
   int i, j, k, ii, jj, kk;
#pragma omp parallel
#pragma omp single
{

   for (i = 0; i < N; i+=BS)
     for (j = 0; j < N; j+=BS)
       for (k = 0; k < N; k+=BS){
#pragma omp task firstprivate(i,j,k) private(ii, jj, kk) \
            depend ( in: A[i:BS][k:BS], B[k:BS][j:BS] ) \
            depend ( inout: C[i:BS][j:BS] )
{
            for (ii = i; ii < i+BS; ii++ )
              for (jj = j; jj < j+BS; jj++ )
                for (kk = k; kk < k+BS; kk++ )
                  C[ii][jj] = C[ii][jj] + A[ii][kk] * B[kk][jj];
printf("task worked on by thread %d \n",omp_get_thread_num());
}

}
}

}

int main ()
{
  float A[N][N], B[N][N], C[N][N];

  int i, j, s = -1;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
    {
      A[i][j] = i * j * s;
      B[i][j] = i + j;
      s = -s;
    }

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
    {
      C[i][j] = 0;
    }

  matmul_tasks (A, B, C);

  return 0;
}

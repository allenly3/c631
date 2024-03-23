/* Atomic reduction solution.
*/

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// Number of times to run the test (for better timings accuracy):
#define NTESTS 10

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 256

// Total number of threads (total number of elements to process in the kernel):
#define NMAX 131072

#define NBLOCKS NMAX/BLOCK_SIZE

// Input array (global host memory):
float h_A[NMAX];

/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */

// It messes up with y!

int
timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime, sum0;
  int error;
  float *d_A;

  cudaMalloc((void **) &d_A, NMAX*sizeof(float));

// Loop to run the timing test multiple times:
int kk;
for (kk=0; kk<NTESTS; kk++)
{

  // We don't initialize randoms, because we want to compare different strategies:
  // Initializing random number generator:
  //  srand((unsigned)time(0));

  // Initializing the input array:
  for (int i=0; i<NMAX; i++)
    {
      h_A[i] = (float)rand()/(float)RAND_MAX;
    }

  // Don't modify this: we need the double precision result to judge the accuracy:
  sum0 = 0.0;
  for (int i=0; i<NMAX; i++)
    sum0 = sum0 + (double)h_A[i];


  // Copying the data to device (we don't time it):
  if (error = cudaMemcpy( d_A, h_A, NMAX*sizeof(float), cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  //--------------------------------------------------------------------------------
  gettimeofday (&tdr0, NULL);

  thrust::device_ptr<float> d_ptr_A(d_A);
  float reduction_sum = thrust::reduce(d_ptr_A, d_ptr_A + NMAX);

  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);

  printf ("Sum: %e (relative error %e)\n", reduction_sum, fabs((double)reduction_sum-sum0)/sum0);

  printf ("Time: %e\n", restime);
  //--------------------------------------------------------------------------------

} // kk loop


  cudaFree(d_A);
  return 0;

}

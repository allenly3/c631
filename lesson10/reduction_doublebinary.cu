/* Two-level binary reduction solution.
*/

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Number of times to run the test (for better timings accuracy):
#define NTESTS 10

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 256

// Total number of threads (total number of elements to process in the kernel):
// For simplicity, use a power of two:
#define NMAX 131072

#define NBLOCKS NMAX/BLOCK_SIZE

// Input array (global host memory):
float h_A[NMAX];
// Copy of h_A on device:
__device__ float d_A[NMAX];
__device__ float d_sum1[NBLOCKS];

__device__ float d_sum;


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


// Kernel(s) should go here:


// First step in binary reduction:
__global__ void MyKernel1 ()
{
  __shared__ float  sum[BLOCK_SIZE];
 
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  // Not needed, because NMAX is a power of two:
  //  if (i >= NMAX)
  //    return;
 
  sum[threadIdx.x] = d_A[i];

  // To make sure all threads in a block have the sum[] value:
  __syncthreads();

  int nTotalThreads = blockDim.x;  // Total number of active threads;
  // only the first half of the threads will be active.
  
  while(nTotalThreads > 1)
    {
      int halfPoint = (nTotalThreads >> 1);	// divide by two
 
      if (threadIdx.x < halfPoint)
	{
	  int thread2 = threadIdx.x + halfPoint;
	  sum[threadIdx.x] += sum[thread2];  // Pairwise summation
	}
      __syncthreads();
      nTotalThreads = halfPoint;  // Reducing the binary tree size by two
    }

  if (threadIdx.x == 0)
    {
      d_sum1[blockIdx.x] = sum[0];
    }

  return;
}


// Second step in binary reduction (one block):
__global__ void MyKernel2 ()
{
  __shared__ float  sum[NBLOCKS];
  
  // Copying from global to shared memory: 
  sum[threadIdx.x] = d_sum1[threadIdx.x];

  // To make sure all threads in a block have the sum[] value:
  __syncthreads();

  int nTotalThreads = blockDim.x;  // Total number of active threads;
  // only the first half of the threads will be active.
  
  while(nTotalThreads > 1)
    {
      int halfPoint = (nTotalThreads >> 1);	// divide by two
 
      if (threadIdx.x < halfPoint)
	{
	  int thread2 = threadIdx.x + halfPoint;
	  sum[threadIdx.x] += sum[thread2];  // Pairwise summation
	}
      __syncthreads();
      nTotalThreads = halfPoint;  // Reducing the binary tree size by two
    }

  if (threadIdx.x == 0)
    {
      d_sum = sum[0];
    }

  return;
}




int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime, sum0;
  float sum;
  int devid, devcount, error;

  /* find number of device in current "context" */
  cudaGetDevice(&devid);
  /* find how many devices are available */
  if (cudaGetDeviceCount(&devcount) || devcount==0)
    {
      printf ("No CUDA devices!\n");
      exit (1);
    }
  else
    {
      cudaDeviceProp deviceProp; 
      cudaGetDeviceProperties (&deviceProp, devid);
      printf ("Device count, devid: %d %d\n", devcount, devid);
      printf ("Device: %s\n", deviceProp.name);
      printf("[deviceProp.major.deviceProp.minor] = [%d.%d]\n\n", deviceProp.major, deviceProp.minor);
    }

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
  if (error = cudaMemcpyToSymbol( d_A, h_A, NMAX*sizeof(float), 0, cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  //--------------------------------------------------------------------------------
  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr0, NULL);


  // First level binary reduction:
  MyKernel1 <<<NBLOCKS, BLOCK_SIZE>>> ();

  // Second level binary reduction (only one block):
  MyKernel2 <<<1, NBLOCKS>>> ();


  // Copying the result back to host (we time it):
  if (error = cudaMemcpyFromSymbol (&sum, d_sum, sizeof(float), 0, cudaMemcpyDeviceToHost))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);

  printf ("Sum: %e (relative error %e)\n", sum, fabs((double)sum-sum0)/sum0);

  printf ("Time: %e\n", restime);
  //--------------------------------------------------------------------------------

} // kk loop

  return 0;

}

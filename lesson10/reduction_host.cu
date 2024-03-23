/* This code can be used as a starting point for an exercise. 

   Try different strategies for reductions
   (summation in this case). This initial code does serial (on the
   host) reduction, but it is written as CUDA code, so should be
   compiled with nvcc . Your task is to write a kernel to
   do the reduction on GPU. You can try different approaches: 

    - atomic reduction, 

    - binary reduction (it should be two level, to be able to handle >
    1024 elements - meaning you'll have to write two kernels, and
    store intermediate results on device; you have to use a shared
    memory aray in both kernels); for simplicity, choose NMAX to be a
    power of two (so you don't have to use NearestPowerOf2 function),

    - if you have time, you can also try a hybrid approach: use the
    first kernel from the binary reduction program, but instead of
    assigning partial sums at the end to intermediate global array,
    add them up globally using atomicAdd function at the end of the
    first kernel.  (You don't need a second kernel).

   Study how speedup depends on NMAX.

   The code always computes the "exact result" sum0 (using double
   precision, serially) - don't touch this part, it is needed to
   estmate the accuracy of your computation.

   The initial copying of the array to device is not timed. We are
   only interested in timing different reduction approaches.

   At the end, you will have to copy the reduction result (sum) from
   device to host, using cudaMemcpyFromSymbol.

   You will discover that for large NMAX, atomic summation is much
   slower than serial code, and of a similar accuracy. How about
   binary reduction - is the accuracy better? What about the speedup?

   What if you use very small number of elements (NMAX=16,
   BLOCK_SIZE=16)?  Which way is faster now - atomics or binary? Why?


Hints (for binary reduction):

* Create a global device array to keep the intermediate results,
  __device__ d_sum1[NBLOCKS].

* First kernel should be called with NBLOCKS blocks and BLOCK_SIZE
  threads; inside the kernel, first the element should be copied from
  the global array d_A to the shared memory array, and then you should
  use thread synchronization, before the binary reduction loop. After
  the while loop, use "if(threadIdx.x == 0)" condition to assign the
  partial sum to the result of this block, which will be in the zeroth
  element of the shared memory array.

* Second kernel will be similar to the first one, but only contains
  one block, and should be called with NBLOCKS threads. Here the first
  line should copy the partial result from the previous kernel to the
  corresponding element of the shared memory array (sum):
  "sum[threadIdx.x] = d_sum1[threadIdx.x];". You have to sync threads
  after that, before the while loop. At the end, use the "if
  (threadIdx.x == 0)" condition to assign the global result (will be
  in the zeroth element of the shared array) to d_sum.




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

// Number of blocks
// This will be needed for the second kernel in two-step binary reduction
// (to declare a shared memory array)
#define NBLOCKS NMAX/BLOCK_SIZE


// Input array (global host memory):
float h_A[NMAX];
// Copy of h_A on device:
__device__ float d_A[NMAX];


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




int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime, sum0;
  float sum;
  int devid, devcount, error;

  if (BLOCK_SIZE>1024)
    {
      printf ("Bad BLOCK_SIZE: %d\n", BLOCK_SIZE);
      exit (1);
    }

  if (NBLOCKS>1024)
    {
      printf ("Bad NBLOCKS: %d\n", NBLOCKS);
      exit (1);
    }


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
  if (error = cudaMemcpyToSymbol (d_A, h_A, NMAX*sizeof(float), 0, cudaMemcpyHostToDevice))
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


  // This serial summation will have to be replaced by calls to kernel(s):
  sum = 0.0;
  for (int i=0; i<NMAX; i++)
    sum = sum + h_A[i];


  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);

  // We are printing the result here, after cudaDeviceSynchronize (this will matter
  // for CUDA code - why?)
  printf ("Sum: %e (relative error %e)\n", sum, fabs((double)sum-sum0)/sum0);

  printf ("Time: %e\n", restime);
  //--------------------------------------------------------------------------------

} // kk loop

  return 0;

}

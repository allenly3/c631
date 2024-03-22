/*

This code computes a visualization of the Julia set.  Specifically, it computes a 2D array of pixels. 

The data can be viewed with gnuplot.

The Julia set iteration is:

z= z**2 + C

If it converges, then the initial point z is in the Julia set.

Code compiles with:

nvcc -arch=sm_60 -O2 julia_gpu.cu

To profile

nvprof  ./a.out

*/

#include "stdio.h"
#define DIM 1000

__device__ int julia(int x, int y){
    const float scaling = 1.5;
    float scaled_x = scaling * (float)(DIM/2 - x)/(DIM/2);
    float scaled_y = scaling * (float)(DIM/2 - y)/(DIM/2);

    float c_real=-0.8f;
    float c_imag=0.156f;

    float z_real=scaled_x;
    float z_imag=scaled_y;
    float z_real_tmp;

    int iter=0;
    for(iter=0; iter<200; iter++){

        z_real_tmp = z_real;
        z_real =(z_real*z_real-z_imag*z_imag) +c_real;
        z_imag = 2.0f*z_real_tmp*z_imag + c_imag;

        if( (z_real*z_real+z_imag*z_imag) > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(int *arr ) {
    // map from blockIdx to pixel position
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int offset = x + y * DIM;

    // now calculate the value at that position
    if(x < DIM && y<DIM){
    int juliaValue = julia( x, y );
    arr[offset] = juliaValue;
    }
}


int main( void ) {
    int x,y;
    int arr[DIM*DIM]; 
    FILE *out;
    int *arr_dev;
    size_t memsize;
    int error;
    int blocksize=32;

    memsize = DIM * DIM * sizeof(int);

    if(error = cudaMalloc( (void **) &arr_dev,memsize ) )
    {
        printf ("Error in cudaMalloc %d\n", error);
        exit (error);
    }

    dim3    grid(DIM/blocksize+1,DIM/blocksize+1);
    dim3    block(blocksize,blocksize);
    kernel<<<grid,block>>>( arr_dev );

    if(error = cudaMemcpy(arr, arr_dev,memsize, cudaMemcpyDeviceToHost ) )
    {
        printf ("Error in cudaMemcpy %d\n", error);
        exit (error);
    }

    /* guarantee synchronization */
    cudaDeviceSynchronize();
                              
    cudaFree( arr_dev );

    out = fopen( "julia.dat", "w" );
    for (y=0; y<DIM; y++) {
        for (x=0; x<DIM; x++) {
            int offset = x + y * DIM;
            if(arr[offset]==1)
                fprintf(out,"%d %d \n",x,y);  
        }
    }
    fclose(out);

}


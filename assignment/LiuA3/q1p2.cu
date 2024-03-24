/*

Compile:

nvcc -arch=sm_60 -O2 q1p2.cu 

nvprof ./a.out

*/


#include <stdio.h>
#include <math.h>
#include <float.h>

#define N 1000 // Number of particles
#define BLOCK_SIZE 256 // Block size

__global__ void findMinimumDistance(float *particles, float *minDistance) {
    __shared__ float distances[BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize the minimum distance to a large value
    float myMinDistance = FLT_MAX;

    // Calculate the distance between the current particle and all other particles
    for (int i = 0; i < N - 1; i += 2) {

        if(i == tid){
            continue;
        }

        float dx = particles[tid] - particles[i];
        float dy = particles[tid + 1] - particles[i + 1];
        float distance = sqrtf(dx * dx + dy * dy);
        myMinDistance = fminf(myMinDistance, distance);
    }

    // Store the minimum distance computed by this thread in shared memory
    distances[threadIdx.x] = myMinDistance;
    __syncthreads();

    // Reduction to find the minimum distance among threads in the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            distances[threadIdx.x] = fminf(distances[threadIdx.x], distances[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Store the minimum distance of this block in global memory
    if (threadIdx.x == 0) {
        minDistance[blockIdx.x] = distances[0];
    }
}

int main() {
    float *particles_host, *minDistance_host;
    float *particles_dev, *minDistance_dev;

    // memory for host 
    particles_host = (float *)malloc(N * 2 * sizeof(float));
    minDistance_host = (float *)malloc((N / BLOCK_SIZE + 1) * sizeof(float));

    //  memory for device 
    cudaMalloc((void **)&particles_dev, N * 2 * sizeof(float));
    cudaMalloc((void **)&minDistance_dev, (N / BLOCK_SIZE + 1) * sizeof(float));

    // Initialize particle coordinates on the host
    /*
        Data structure
        particle[INDEX]:
        INDEX % 2 == 0 : x
        INDEX % 2 == 1 : y
    */
    for (int i = 0; i < N * 2; ++i) {
        particles_host[i] = rand() / (float)RAND_MAX; 
        //printf("particle %f \n", particles_host[i]);
    }
    //printf(" \n\n\n\n");

     /* copy arrays to device memory (synchronous) */
    cudaMemcpy(particles_dev, particles_host, N * 2 * sizeof(float), cudaMemcpyHostToDevice);

    /* set up device execution configuration */
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(numBlocks, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

     /* execute kernel (asynchronous!) */
    findMinimumDistance<<<grid, block>>>(particles_dev, minDistance_dev);

     /* retrieve results from device (synchronous) */
    cudaMemcpy(minDistance_host, minDistance_dev, (N / BLOCK_SIZE + 1) * sizeof(float), cudaMemcpyDeviceToHost);

   
   //find min
    float minDistance = minDistance_host[0];
    for (int i = 1; i < (N / BLOCK_SIZE + 1); ++i) {
        minDistance = fminf(minDistance, minDistance_host[i]);
    }

    /* check results */
    printf("Minimum distance between particles: %f\n", minDistance);

    /* free memory */
    cudaFree(particles_dev);
    cudaFree(minDistance_dev);
    free(particles_host);
    free(minDistance_host);

    return 0;
}

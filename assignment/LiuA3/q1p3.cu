/*
    Compile on graham with:

    nvcc -arch=sm_60 -O2 q1p3.cu 

    nvprof ./a.out

    if want to see cpu profiling to compare GPU and CPU performance

    nvprof  --cpu-profiling on   ./a.out

*/
#include <stdio.h>
#include <math.h>
#include <float.h>

#define N 1024*1024  // 100 particles
#define BLOCK_SIZE 256 // Block size

__global__ void findMinimumDistance(double *particles, double *minDistance) {
    __shared__ double sharedMinDistances[BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    double myMinDistance = DBL_MAX;

    // Calculate the indices of the particles for this thread
    int particle1_index = tid * 2;
    int particle2_index = particle1_index + 2;

    // Ensure the indices are within bounds
    if (particle1_index < N * 2 && particle2_index < N * 2) {
        // Compute the distance between the particles
        double dx = particles[particle1_index] - particles[particle2_index];
        double dy = particles[particle1_index + 1] - particles[particle2_index + 1];
        double distance = sqrt(dx * dx + dy * dy);
        myMinDistance = distance;
    }

    sharedMinDistances[threadIdx.x] = myMinDistance;
    __syncthreads();

    // Reduction to find minimum distance among threads in the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sharedMinDistances[threadIdx.x] = fmin(sharedMinDistances[threadIdx.x], sharedMinDistances[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        minDistance[blockIdx.x] = sharedMinDistances[0];
    }
}

int main() {
    double *particles_dev, *minDistance_dev;
    double particles_host[N * 2];
    double minDistance_host[N / BLOCK_SIZE + 1];

    srand(time(NULL));
    for (int i = 0; i < N * 2; ++i) {
        particles_host[i] = (double)rand() / RAND_MAX;
    }

    cudaMalloc((void **)&particles_dev, N * 2 * sizeof(double));
    cudaMalloc((void **)&minDistance_dev, (N / BLOCK_SIZE + 1) * sizeof(double));

    cudaMemcpy(particles_dev, particles_host, N * 2 * sizeof(double), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    findMinimumDistance<<<numBlocks, BLOCK_SIZE>>>(particles_dev, minDistance_dev);

    cudaMemcpy(minDistance_host, minDistance_dev, (N / BLOCK_SIZE + 1) * sizeof(double), cudaMemcpyDeviceToHost);

    double minDistance = minDistance_host[0];
    for (int i = 1; i < (N / BLOCK_SIZE + 1); ++i) {
        minDistance = fmin(minDistance, minDistance_host[i]);
    }

    printf("Minimum distance between particles (GPU): %lf\n", minDistance);

    cudaFree(particles_dev);
    cudaFree(minDistance_dev);

    return 0;
}

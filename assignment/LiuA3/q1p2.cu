#include <stdio.h>
#include <math.h>
#include <float.h>

#define N 100  // 100 particles
#define BLOCK_SIZE 256 // Block size

__global__ void findMinimumDistance(double *particles, double *minDistance) {
    __shared__ double sharedMinDistances[BLOCK_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int index = tid * 2;

    double myMinDistance = DBL_MAX;

    if (index < N * 2) {
        double myX = particles[index];
        double myY = particles[index + 1];

        for (int i = index + 2; i < N * 2; i += 2) {
            double dx = myX - particles[i];
            double dy = myY - particles[i + 1];
            double distance = sqrt(dx * dx + dy * dy);
            myMinDistance = fmin(myMinDistance, distance);
        }
    }

    sharedMinDistances[threadIdx.x] = myMinDistance;
    __syncthreads();

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
    double *d_particles, *d_minDistance;
    double h_particles[N * 2];
    double h_minDistance[N / BLOCK_SIZE + 1];

    srand(time(NULL));
    for (int i = 0; i < N * 2; ++i) {
        h_particles[i] = (double)rand() / RAND_MAX;
    }

    cudaMalloc((void **)&d_particles, N * 2 * sizeof(double));
    cudaMalloc((void **)&d_minDistance, (N / BLOCK_SIZE + 1) * sizeof(double));

    cudaMemcpy(d_particles, h_particles, N * 2 * sizeof(double), cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    findMinimumDistance<<<numBlocks, BLOCK_SIZE>>>(d_particles, d_minDistance);

    cudaMemcpy(h_minDistance, d_minDistance, (N / BLOCK_SIZE + 1) * sizeof(double), cudaMemcpyDeviceToHost);

    double minDistance = h_minDistance[0];
    for (int i = 1; i < (N / BLOCK_SIZE + 1); ++i) {
        minDistance = fmin(minDistance, h_minDistance[i]);
    }

    printf("Minimum distance between particles (GPU): %lf\n", minDistance);

    cudaFree(d_particles);
    cudaFree(d_minDistance);

    return 0;
}

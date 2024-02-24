#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define NUM_POINTS 1000000

int main(int argc, char *argv[]) {
    int i, rank, size;
    int num_points_inside_circle = 0;
    double x, y, pi, local_pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process uses a different seed for the random number generator
    srand(time(NULL) + rank);

    // Generate random points and count how many fall inside the circle
    for (i = 0; i < NUM_POINTS / size; i++) {
        x = (double)rand() / RAND_MAX; // Generate random x-coordinate
        y = (double)rand() / RAND_MAX; // Generate random y-coordinate

        // Check if the point falls inside the circle (radius = 1)
        if (x * x + y * y <= 1) {
            num_points_inside_circle++;
        }
    }

    // Each process calculates its local estimate of Pi
    local_pi = 4.0 * num_points_inside_circle / (NUM_POINTS / size);

    // Gather local estimates to process 0
    MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Process 0 calculates the average of all estimates
        pi /= size;
        printf("Estimated value of Pi: %f\n", pi);
    }

    MPI_Finalize();

    return 0;
}
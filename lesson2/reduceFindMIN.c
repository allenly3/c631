#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 10000

double eggholder(double x, double y) {
    return -(y + 47) * sin(sqrt(fabs(x / 2 + (y + 47)))) - x * sin(sqrt(fabs(x - (y + 47))));
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //A different random sequence
    srand(time(NULL) + rank);

    double global_min_value = INFINITY;
    double local_min_value = INFINITY;
    double x_min, y_min;

    for (int i = 0; i < N; i++) {
        // Generate random (x, y) values in the range [-512.0, 512.0]
        double x = ((double)rand() / RAND_MAX) * 1024.0 - 512.0;
        double y = ((double)rand() / RAND_MAX) * 1024.0 - 512.0;

        // Evaluate the Eggholder function
        double value = eggholder(x, y);

        // Update local minimum
        if (value < local_min_value) {
            local_min_value = value;
            x_min = x;
            y_min = y;
        }
    }

    // Reduce local minimum values to find the global minimum
    MPI_Reduce(&local_min_value, &global_min_value, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // Process 0 prints the result
    if (rank == 0) {
        printf("Global minimum value: %f\n", global_min_value);
        printf("Corresponding (x, y): (%f, %f)\n", x_min, y_min);
        printf("Varify:  %f", eggholder(x_min, y_min));
    }

    MPI_Finalize();
    return 0;
}
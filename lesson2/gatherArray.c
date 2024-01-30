#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process generates its own array
    double local_data[3] = {2.1 + rank, 3.4 + rank, 6.6 + rank};

    // The root process will gather all arrays into this global 2D array
    double global_data[size][3];

    MPI_Gather(local_data, 3, MPI_DOUBLE, global_data, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Gathered data at root process:\n");
        for (int i = 0; i < size; i++) {
            printf("[%f, %f, %f]\n", global_data[i][0], global_data[i][1], global_data[i][2]);
        }
    }

    MPI_Finalize();

    return 0;
}
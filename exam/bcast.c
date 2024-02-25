#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, data = 66;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // All processes call MPI_Bcast to receive the broadcasted data from process 0
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received data: %d\n", rank, data);

    MPI_Finalize();

    return 0;
}
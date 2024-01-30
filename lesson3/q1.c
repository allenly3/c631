#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int tag = 123;

    for (int message_size = 1; ; message_size *= 2) {
        char *message = (char *)malloc(message_size);

        if (rank == 0) {
            printf("Trying message size: %d\n", message_size);
            MPI_Send(message, message_size, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(message, message_size, MPI_CHAR, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(message, message_size, MPI_CHAR, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(message, message_size, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
        }

        free(message);
    }

    MPI_Finalize();
    return 0;
}
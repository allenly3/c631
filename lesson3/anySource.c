#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        printf("This program is designed to run with 4 processes. Exiting.\n");
        MPI_Finalize();
        return 1;
    }

    int data;
    MPI_Status status;

    if (rank == 0) {
        data = 100;
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent %d to Process 1\n", data);

        MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("Process 0 received %d from Process %d\n", data, status.MPI_SOURCE);

        // Now, process 0 sends to process 3
        MPI_Send(&data, 1, MPI_INT, 3, 3, MPI_COMM_WORLD);
        printf("Process 0 sent %d to Process 3\n", data);
    } else if (rank == 1) {
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process 1 received %d from Process 0\n", data);
        MPI_Send(&data, 1, MPI_INT, 2, 1, MPI_COMM_WORLD);
        printf("Process 1 sent %d to Process 2\n", data);
    } else if (rank == 2) {
        MPI_Recv(&data, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
        printf("Process 2 received %d from Process 1\n", data);
        MPI_Send(&data, 1, MPI_INT, 3, 2, MPI_COMM_WORLD);
        printf("Process 2 sent %d to Process 3\n", data);
    } else if (rank == 3) {
        MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("Process 3 received %d from Process %d\n", data, status.MPI_SOURCE);

        // Now, process 3 sends to process 0
        MPI_Send(&data, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        printf("Process 3 sent %d to Process 0\n", data);
    }

    MPI_Finalize();
    return 0;
}
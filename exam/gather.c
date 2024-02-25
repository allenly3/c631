#include <stdio.h>
#include <mpi.h>

#define MAX_SIZE 100

int main(int argc, char *argv[]) {
    int my_rank, size, send_data, recv_data[MAX_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process initializes its own send_data
    send_data = my_rank + 1;
    // Gather data from all processes to process 0
    MPI_Gather(&send_data, 1, MPI_INT, recv_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Process 0: Received data:\n");
        for (int i = 0; i < size; i++) {
            printf("%d ", recv_data[i]);
        }
        printf("\n");
    }

    MPI_Finalize();

    return 0;
}
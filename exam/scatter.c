#include <stdio.h>
#include <mpi.h>

#define ARRAY_SIZE 8

int main(int argc, char *argv[]) {
    int my_rank, size;
    int send_data[ARRAY_SIZE];
    int recv_data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the send_data array in process 0
    if (my_rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            send_data[i] = i + 1;
        }
    }

    // Scatter data from process 0 to all other processes
    MPI_Scatter(send_data, 1, MPI_INT, &recv_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Print received data in each process
    printf("Process %d received data: %d\n", my_rank, recv_data);

    MPI_Finalize();

    return 0;
}
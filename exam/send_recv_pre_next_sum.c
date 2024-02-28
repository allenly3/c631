#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int my_rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int data = 0;
    if (my_rank == p - 1) {
        // Process p3 initializes the data
        data = my_rank;
        MPI_Send(&data, 1, MPI_INT, my_rank -1, 0, MPI_COMM_WORLD);
    }

    if (my_rank > 0 && my_rank != p - 1) {
        // Receive data from the previous process
        MPI_Recv(&data, 1, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data = data + my_rank;
        MPI_Send(&data, 1, MPI_INT, my_rank -1, 0, MPI_COMM_WORLD);
    }
    //MPI_Barrier(MPI_COMM_WORLD); 
    if (my_rank == 0 ) {
        // Send data to the next process
        MPI_Recv(&data, 1, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (my_rank == 0) {
        printf("Received data at process 0: %d\n", data);
    }

    MPI_Finalize();
    return 0;
}
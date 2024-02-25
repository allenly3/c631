#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int my_rank, size, data = 0, sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process has its own data
    data = my_rank +1;

    /*
    MPI_MAX: Computes the maximum value.
    MPI_MIN: Computes the minimum value.
    MPI_PROD: Computes the product of all values.
    */
    // Each process adds its own data to the sum
    MPI_Reduce(&data, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Process 0: Sum of all data = %d\n", sum);
    }else{
        printf("Process %d: sent data : %d\n", my_rank,data);
    }

    MPI_Finalize();

    return 0;
}
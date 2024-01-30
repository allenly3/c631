#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Seed the random number generator with a unique seed for each process
    srand(time(NULL) + rank);

    // Generate and print 5 random numbers for each process
    for (int i = 0; i < 5; i++) {
        int random_number = rand();
        printf("Process %d: Random number %d: %d\n", rank, i + 1, random_number);
    }

    MPI_Finalize();
    return 0;
}
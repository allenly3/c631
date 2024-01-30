#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double local_values[3] = {10.2, 2.2, 13.4};  // Replace with your local values
    double global_values[3 * size];  // Array to store all gathered values
      if (rank == 1) {
        local_values[0] = 6.66;
        local_values[1] = 6.66;
        local_values[2] = 6.66;
        
      }
            if (rank == 2) {
        local_values[0] = 68.44;
        local_values[1] = 8.4;
        local_values[2] = 4.2;
        
      }
    // Gather arrays from all processes to process 0
    MPI_Gather(local_values, 3, MPI_DOUBLE, global_values, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Process 0 prints the result
    if (rank == 0) {
        printf("All gathered values:\n");
        for (int i = 0; i < 3 * size; i++) {
            printf("%f ", global_values[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
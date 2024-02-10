#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIDE_LENGTH 2 // Cube side length

int main(int argc, char *argv[]) {
    int rank, size, i ;
    int coords[3]; // Array to store Cartesian coordinates
    int cube_size[3] = {SIDE_LENGTH, SIDE_LENGTH, SIDE_LENGTH}; // Cube dimensions
    int periods[3] = {0, 0, 0}; // Set periodicity to false in all dimensions
    MPI_Comm cart_comm; // Cartesian communicator
    int side_sums[3] = {0}; // Array to store sums for each side

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create a 3D Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 3, cube_size, periods, 0, &cart_comm);

    // Get the Cartesian coordinates of the current process
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    // Determine the sides that the current process belongs to
    for (i = 0; i < 3; i++) {
        if (coords[i] == 0) {
            // Process belongs to side 0 in dimension i
            side_sums[i] += rank + 1; // Assuming each process holds a unique number
        } else if (coords[i] == SIDE_LENGTH - 1) {
            // Process belongs to side SIDE_LENGTH - 1 in dimension i
            side_sums[i] += rank + 1; // Assuming each process holds a unique number
        }
    }

    // Create communicators for each side of the cube
    MPI_Comm side_comms[3];
    for (i = 0; i < 3; i++) {
        int remain_dims[3] = {1, 1, 1}; // Keep all dimensions for this side
        remain_dims[i] = 0; // Remove dimension i
        MPI_Cart_sub(cart_comm, remain_dims, &side_comms[i]);
    }

    // Perform collective communication to compute sums for each side
    for (i = 0; i < 3; i++) {
        int side_sum;
        MPI_Reduce(&side_sums[i], &side_sum, 1, MPI_INT, MPI_SUM, 0, side_comms[i]);

        // Print the sum for each side (only for processes on side 0 in each communicator)
        if (coords[i] == 0) {
            printf("Process %d on side %d in dimension %d has sum: %d\n", rank, coords[i], i, side_sum);
        }
    }

    // Clean up
    for (i = 0; i < 3; i++) {
        MPI_Comm_free(&side_comms[i]);
    }
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}

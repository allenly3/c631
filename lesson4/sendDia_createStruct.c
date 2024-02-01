#include "stdio.h"
#include "mpi.h"

#define n 10

int main(int argc, char* argv[]) {
    int p;
    int my_rank;
    float         A[n][n];          /* Complete Matrix */
    float         diagonal[n];       /* Diagonal Elements */
    MPI_Datatype  struct_mpi_t;
    int           i, j;
    MPI_Status    status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Aint displacements[n];
    int block_lengths[n];
    MPI_Datatype types[n];

    // Create MPI datatype for diagonal elements using MPI_Type_create_struct
    for (i = 0; i < n; i++) {
        block_lengths[i] = 1;
        MPI_Get_address(&diagonal[i], &displacements[i]);
        displacements[i] -= (MPI_Aint)&diagonal[0];
        types[i] = MPI_FLOAT;
    }

    MPI_Type_create_struct(n, block_lengths, displacements, types, &struct_mpi_t);
    MPI_Type_commit(&struct_mpi_t);

    if (my_rank == 0) {
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                A[i][j] = (float) i + j;

        // Extract diagonal elements and send
        for (i = 0; i < n; i++)
            diagonal[i] = A[i][i];

        MPI_Send(diagonal, 1, struct_mpi_t, 1, 0, MPI_COMM_WORLD);
    } else { /* my_rank == 1 */
        for (i = 0; i < n; i++)
            diagonal[i] = 0.0;

        MPI_Recv(diagonal, 1, struct_mpi_t, 0, 0, MPI_COMM_WORLD, &status);

        // Print received diagonal elements
        for (i = 0; i < n; i++)
            printf("%4.1f ", diagonal[i]);

        printf("\n");
    }

    MPI_Type_free(&struct_mpi_t);
    MPI_Finalize();
    return 0;
}
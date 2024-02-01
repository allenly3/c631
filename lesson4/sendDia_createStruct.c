#include "stdio.h"
#include "mpi.h"

#define n 10

int main(int argc, char* argv[]) {
    int p;
    int my_rank;
    float         A[n][n];          /* Complete Matrix */
    float         T[n][n];          /* Upper Triangle  */
    int           displacements[n];
    int           block_lengths[n];
    MPI_Datatype  index_mpi_t;
    MPI_Datatype  diagonal_mpi_t;   /* MPI datatype for diagonal elements */
    int           i, j;
    MPI_Status    status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    for (i = 0; i < n; i++) {
        block_lengths[i] = 1;
        displacements[i] = (n+1)*i;
    }
    MPI_Type_indexed(n, block_lengths, displacements,
        MPI_FLOAT, &index_mpi_t);
    MPI_Type_commit(&index_mpi_t);

    /* Define MPI datatype for diagonal elements */
    MPI_Type_vector(n, 1, n+1, MPI_FLOAT, &diagonal_mpi_t);
    MPI_Type_commit(&diagonal_mpi_t);

    if (my_rank == 0) {
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                A[i][j] = (float) i + j;

        /* Send only the diagonal elements using the new datatype */
        MPI_Send(&A[0][0], 1, diagonal_mpi_t, 1, 0, MPI_COMM_WORLD);
    } else { /* my_rank == 1 */
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                T[i][j] = 0.0;

        /* Receive only the diagonal elements using the new datatype */
        MPI_Recv(&T[0][0], 1, diagonal_mpi_t, 0, 0, MPI_COMM_WORLD, &status);

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                printf("%4.1f ", T[i][j]);
            printf("\n");
        }
    }

    MPI_Type_free(&diagonal_mpi_t);
    MPI_Finalize();
    return 0;
}
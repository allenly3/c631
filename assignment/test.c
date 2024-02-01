#include "stdio.h"
#include "mpi.h"

#define n 10

int main(int argc, char* argv[]) {
    int p;
    int my_rank;
    float         A[n][n];          /* Complete Matrix */
    float         T[n][n];          /* Upper Triangle */
    MPI_Datatype  index_mpi_t;
    int           i, j;
    MPI_Status    status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int block_lengths[n];
    int displacements[n];
    MPI_Aint displacements_mpi_a[n];

    for (i = 0; i < n; i++) {
        block_lengths[i] = 1;
        displacements[i] = i * n;
        MPI_Get_address(&A[i][0], &displacements_mpi_a[i]);
        displacements_mpi_a[i] -= (MPI_Aint)&A[0][0];
    }

    MPI_Type_create_struct(n, block_lengths, displacements_mpi_a, MPI_FLOAT, &index_mpi_t);
    MPI_Type_commit(&index_mpi_t);

    if (my_rank == 0) {
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                A[i][j] = (float)i + j;

        MPI_Send(A, 1, index_mpi_t, 1, 0, MPI_COMM_WORLD);
    } else { /* my_rank == 1 */
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                T[i][j] = 0.0;

        MPI_Recv(T, 1, index_mpi_t, 0, 0, MPI_COMM_WORLD, &status);

        // Print the full matrix in Process 1
        printf("Process 1 received full matrix:\n");
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                printf("%4.1f ", T[i][j]);
            printf("\n");
        }
    }

    MPI_Type_free(&index_mpi_t);
    MPI_Finalize();
    return 0;
}
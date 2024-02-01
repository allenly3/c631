#include "stdio.h"
#include "mpi.h"
#define n 10

int main(int argc, char* argv[]) {
    int p;
    int my_rank;
    float A[n][n];
    float T[n][n];
    MPI_Status status;
    MPI_Datatype column_mpi_t;
    int i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Type_vector(10, 1, 10, MPI_FLOAT, &column_mpi_t);
    MPI_Type_commit(&column_mpi_t);

    if (my_rank == 0) {
        for (i = 0; i < 10; i++)
            for (j = 0; j < 10; j++)
                A[i][j] = (float) j;
        MPI_Send(&(A[0][2]), 1, column_mpi_t, 1, 0,MPI_COMM_WORLD);
    } else { 
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                T[i][j] = 0.0;

        MPI_Recv(&(T[0][2]), 1, column_mpi_t, 0, 0,MPI_COMM_WORLD, &status);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                printf("%4.1f ", T[i][j]);
            printf("\n");
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0 ; 
}  /* main */
	
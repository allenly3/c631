#include "stdio.h"
#include "mpi.h"
#include "unistd.h" 

#define n 10

int main(int argc, char* argv[]) {
    int p;
    int my_rank;
    float         A[n][n];          /* Complete Matrix */
    float         T[n][n];          /* Upper Triangle  */
    int           displacements[n];
    int           block_lengths[n];
    MPI_Datatype  index_mpi_t;
    int           i, j;
    MPI_Status    status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    for (i = 0; i < n; i++) {
        block_lengths[i] = 1;
        displacements[i] = 9*(i+1);  // diagonal:  displacements[i] = i * (n + 1); 
        //printf("%d ",(n+1)*i);
    }
    MPI_Type_indexed(n, block_lengths, displacements,
        MPI_FLOAT, &index_mpi_t);
    MPI_Type_commit(&index_mpi_t);


    if (my_rank == 0) {
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                A[i][j] = (float) i + j;
        MPI_Send(A, 1, index_mpi_t, 1, 0, MPI_COMM_WORLD);
    } else {/* my_rank == 1 */
        usleep(100000);
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                T[i][j] = 0.0;
        MPI_Recv(T, 1, index_mpi_t, 0, 0, MPI_COMM_WORLD, &status);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                printf("%4.1f ", T[i][j]);
            printf("\n");
        }
    }

    MPI_Finalize();
}
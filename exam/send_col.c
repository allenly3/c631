#include "stdio.h"
#include "mpi.h"
#include "unistd.h" 

int main(int argc, char* argv[]) {
    int p;
    int my_rank;
    float A[10][10];
    float B[10][10];
    MPI_Status status;
    MPI_Datatype column_mpi_t;
    int i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Type_vector(10, 2, 10, MPI_FLOAT, &column_mpi_t);
    MPI_Type_commit(&column_mpi_t);

    if (my_rank == 0) {
        for (i = 0; i < 10; i++){
             for (j = 0; j < 10; j++){
                A[i][j] = (float) j;
                printf("%3.1f ",(float) j);
             } 
             printf("\n");
        }
           

        MPI_Send(&(A[0][2]), 1, column_mpi_t, 1, 0,
            MPI_COMM_WORLD);
    } else { /* my_rank = 1 */
        MPI_Recv(&(A[0][2]), 1, column_mpi_t, 0, 0,
            MPI_COMM_WORLD, &status);

        usleep(100000);
        printf("starts:\n");
        for (i = 0; i < 10; i++)
            printf("%3.1f %3.1f  \n", A[i][2], A[i][3]);
    }

    MPI_Finalize();
}
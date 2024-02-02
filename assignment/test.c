#include "stdio.h"
#include "mpi.h"
#define n 16

int main(int argc, char* argv[]) {
    int p;
    int my_rank;
    float A[n][n];
    float T[n][n];
    MPI_Status status;
    MPI_Datatype column_mpi_t;
    int i, j, k ;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Type_vector(4, 4, 16, MPI_FLOAT, &column_mpi_t);
    MPI_Type_commit(&column_mpi_t);

for(k = 0 ; k < 4; k++){
    if (my_rank == 0) {
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                A[i][j] = (float) j;

    
        if(k == 0 ){
            MPI_Send(&(A[0][0]), 1, column_mpi_t, k, 0,MPI_COMM_WORLD);
            MPI_Recv(&(T[0][0]), 1, column_mpi_t, k, 0,MPI_COMM_WORLD, &status);
        }else{
            MPI_Send(&(A[0][k*4]), 1, column_mpi_t, k, 0,MPI_COMM_WORLD);
        }

        
    } else { 
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                T[i][j] = 0.0;
                
        MPI_Recv(&(T[0][k*4]), 1, column_mpi_t, 0, 0,MPI_COMM_WORLD, &status);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                printf("%4.1f ", T[i][j]);
            printf("\n");
        }
        printf("\n");
    }
}

    MPI_Finalize();
    return 0 ; 
}  /* main */
	
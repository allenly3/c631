#include "stdio.h"
#include "mpi.h"
#include "unistd.h" 

#define N 12

int main(int argc, char* argv[]) {
    int p;
    int my_rank;
    float A[N][N];
    float B[N][N]; // In p0 to receive data.
    int group ; //  To find how many col in each process
    MPI_Status status;
    MPI_Datatype column_mpi_t;
    int i, j;
    float sum = 0 ;

    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    group = N/p; // in each process
    MPI_Type_vector(N, group, N, MPI_FLOAT, &column_mpi_t);
    MPI_Type_commit(&column_mpi_t);


 
        for (i = 0; i < N; i++){
             for (j = 0; j < N; j++){
                A[i][j] = (float) j+i;
                //printf("%3.1f ",(float) A[i][j]);
             } 
             //printf("\n");
        }

        MPI_Send(&(A[0][my_rank*group]), 1, column_mpi_t, 0, 0,
            MPI_COMM_WORLD);

    if(my_rank==0){ /* my_rank = 0 */
        for (i = 0; i < N; i++){
             for (j = 0; j < N; j++){
                B[i][j] = (float) 0;
                //printf("%3.1f ",(float) A[i][j]);
             } 
             //printf("\n");
        }

        for(i = 0; i < p; i++){
            MPI_Recv(&(B[0][i*group]), 1, column_mpi_t, i, 0,
            MPI_COMM_WORLD, &status); 
        }
       
        //sum row
        for ( i = 0; i < N;i++){
            for ( j = 0;j < N;j++){
                printf("%3.1f ",(float) B[i][j]);
                sum = sum + B[i][j];
            }
            printf(" = %3.1f\n", sum);
            sum = 0 ;
        }

    }

    MPI_Finalize();
    return 0 ;
}
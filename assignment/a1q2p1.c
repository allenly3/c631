#include "stdio.h"
#include "mpi.h"
#define n 16
#define block_size 4  // 4x4 block

int main(int argc, char* argv[]) {
    int my_rank, num_procs;
    float A[n][n];
    float T[block_size][block_size] = { 0.0 };  // Temporary block to receive data
    MPI_Status status;
    MPI_Datatype block_mpi_t;
    int i,j,time,start_row,start_col;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create a datatype for the 4x4 block
    MPI_Type_vector(block_size, block_size, n, MPI_FLOAT, &block_mpi_t);
    MPI_Type_commit(&block_mpi_t);

    if (my_rank == 0) {
        // Initialize the matrix in process 0
        for ( i = 0; i < n; i++) {
            for ( j = 0; j < n; j++) {
                A[i][j] = (float)(i * n + j);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before starting the communication
    
    for(time = 0 ; time < n/block_size; time ++){ // 16/4 = 4 , so need 4 times to 
     MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before starting the communication
     //printf("Sendinng Row:  %d :\n", time);
        if (my_rank < 4) {  // Only the first 4 processes participate
            start_row = (my_rank / block_size + time) * block_size;
            start_col = (my_rank % block_size) * block_size;

            if (my_rank == 0) {
                // Process 0 sends and receives the same block to itself
                MPI_Sendrecv(&A[start_row][start_col], 1, block_mpi_t, 0, 0,
                    &T, block_size * block_size, MPI_FLOAT, 0, 0,
                    MPI_COMM_WORLD, &status);
            }
            else {
                // Other processes only receive
                MPI_Recv(&T, block_size * block_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            }

            // Print the received block
            printf("Row %d, Process %d received:\n",time,  my_rank);
            for (i = 0; i < block_size; i++) {
                for (j = 0; j < block_size; j++) {
                    printf("%4.1f ", T[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }

        if (my_rank == 0) {
            for (i = 1; i < num_procs && i < 4; i++) {   // Process 0 sends blocks to other processes
                start_row = (i / block_size + time) * block_size;
                start_col = (i % block_size) * block_size;
                MPI_Send(&A[start_row][start_col], 1, block_mpi_t, i, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Type_free(&block_mpi_t);  // Cleanup the custom datatype
    MPI_Finalize();
    return 0;
}

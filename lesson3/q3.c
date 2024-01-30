#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int data = 100;
    MPI_Request req;
    int flag = 0;
    MPI_Status status;

    if (rank == 0) {
        printf("Process 0  sends first MSG\n");
        MPI_Isend(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &req);

        // Check if the send operation has completed
        MPI_Test(&req, &flag, &status);

        while (!flag){
            sleep(3);
            MPI_Test(&req, &flag, &status);
        }


        printf("Process 1 idle, send another MSG\n");
        data = 999;
        MPI_Isend(&data, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &req);

    } else if (rank == 1) {
	  MPI_Recv(&data, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
       printf("Process 1 recv first MSG\n");
    }

    MPI_Finalize();
    return 0;
}
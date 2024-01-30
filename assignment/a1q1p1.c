#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100000

double eggholder(double x, double y) {
    return -(y + 47) * sin(sqrt(fabs(x / 2 + (y + 47)))) - x * sin(sqrt(fabs(x - (y + 47))));
}

int main(int argc, char *argv[]) {
    int rank, size, i ;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //A different random sequence
    srand(time(NULL) + rank);

    double local_min_value = INFINITY;
    double minHolder = INFINITY; // used for comparing 
    double minX, minY;
    double local_data[3] = {0,0,0}; //  [0] eggholder value, [1] x, [2] y
    
    double global_data[size][3]; // 2 dimension 

    for ( i = 0; i < N; i++) {
        // Generate random (x, y) values in the range [-512.0, 512.0]
        double x = ((double)rand() / RAND_MAX) * 1024.0 - 512.0;
        double y = ((double)rand() / RAND_MAX) * 1024.0 - 512.0;

        // Evaluate the Eggholder function
        double value = eggholder(x, y);

        // Update local minimum
        if (value < local_min_value) {
            local_min_value = value;
            local_data[0] = local_min_value;
            local_data[1] = x;
            local_data[2] = y;
        }
    }

    // Gather local data
    MPI_Gather(local_data, 3, MPI_DOUBLE, global_data, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("After Gathering data at root process:\n");
        for ( i = 0; i < size; i++) {
            printf("[%f, %f, %f]\n", global_data[i][0], global_data[i][1], global_data[i][2]);
            if(global_data[i][0] < minHolder){
                minHolder = global_data[i][0];
                minX = global_data[i][1];
                minY = global_data[i][2];
            }
        }
        printf("Min Value is %f, x is %f, y is %f:\n", minHolder, minX,minY);
    }

    MPI_Finalize();
    return 0;
}
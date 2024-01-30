#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// #define N 100000

double eggholder(double x, double y) {
    return -(y + 47) * sin(sqrt(fabs(x / 2 + (y + 47)))) - x * sin(sqrt(fabs(x - (y + 47))));
}

int main(int argc, char *argv[]) {
    double p0MinValue = INFINITY; // Used to check if change is less than 0.1

    //  1 stands for Improvement greater than 0.1 and keep while looping
    // process 0 will broadcast flag value 0 if improvement less than 0.1 
    int flag = 1;

    int check = 1;// used to avoid doubleCheck runs
    int rank, size, i;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // A different random sequence
    srand(time(NULL) + rank);

    double local_min_value = INFINITY;
    double minHolder = INFINITY; // used for comparing
    double minX, minY;
    double local_data[3] = {0, 0, 0}; //  [0] eggholder value, [1] x, [2] y

    double global_data[size][3]; // 2 dimension

    time_t start_time, current_time;
    double gapTime; 
    double runTime = 30.0; // Set the maximum execution time to 30 seconds
    time(&start_time);

    while (gapTime < runTime && flag) {
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

            // check time Running
            time(&current_time);
            gapTime = difftime(current_time, start_time);
            //printf("-------------current Running Time: %f\n", gapTime);
            if( (int)gapTime % 5 == 0 && gapTime > 0 && check) { // every 5 secs to check
                printf("****************P%d  Time to Check  ***************\n", rank);
                check = 0;// already run
                MPI_Gather(local_data, 3, MPI_DOUBLE, global_data, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                if (rank == 0) {  // check the if the minValue change is less than 0.1
                    //printf("After Gathering data at root process:\n");
                    for (i = 0; i < size; i++) {
                        //printf("[%f, %f, %f]\n", global_data[i][0], global_data[i][1], global_data[i][2]);
                        if (global_data[i][0] < minHolder) {
                            minHolder = global_data[i][0];
                            minX = global_data[i][1];
                            minY = global_data[i][2];
                        }
                    }
          
                    if( p0MinValue - minHolder < 0.1){
                        printf("Current Round, Improvement less than 0.1, Program Terminates\n");
                        flag = 0 ;
                        
                    }else{
                        printf("Got big Improvement, Go Next Round.\n\n");
                        p0MinValue = minHolder;
                    }
                      
                }  
               
                            

            }else if((int)gapTime % 5 == 1){
                check = 1;
            }
            
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD); 
        }

    if(rank == 0) printf("Min Value is %f, x is %f, y is %f:\n", minHolder, minX, minY);  
    
    MPI_Finalize();
    return 0;
}
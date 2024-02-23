/*

This code computes a visualization of the Julia set.  Specifically, it computes a 2D array of pixels.

The data can be viewed with gnuplot.

The Julia set iteration is:

z= z**2 + C

If it converges, then the initial point z is in the Julia set.

Compile with:

mpicc -fopenmp -O2 julia_mpi.c

Example run:

mpirun -bind-to-core -np 12 ./a.out

Output files can be combined with:

cat julia.dat.* > julia.dat

*/


#include "stdio.h"
#include "omp.h"
#include "mpi.h"


#define DIM 1000

int julia(int x, int y){
    const float scaling = 1.5;
    float scaled_x = scaling * (float)(DIM/2 - x)/(DIM/2);
    float scaled_y = scaling * (float)(DIM/2 - y)/(DIM/2);

    float c_real=-0.8f;
    float c_imag=0.156f;

    float z_real=scaled_x;
    float z_imag=scaled_y;
    float z_real_tmp;

    int iter=0;
    for(iter=0; iter<200; iter++){

        z_real_tmp = z_real;
        z_real =(z_real*z_real-z_imag*z_imag) +c_real;
        z_imag = 2.0f*z_real_tmp*z_imag + c_imag;

        if( (z_real*z_real+z_imag*z_imag) > 1000)
            return 0;
    }

    return 1;
}

void kernel( int *arr,int ystart,int yend ){

int x,y;

    for (y=ystart; y<yend; y++) {
        for (x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            arr[offset] = juliaValue;
        }
    }
}

int main(int argc, char** argv){
    int arr[DIM*DIM];
    FILE *out;
    int x,y;
    double start,end;
    int         my_rank_mpi;   /* My process rank           */
    int         p_mpi;         /* The number of processes   */
    char filename[128];
    int ystart,yend;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &p_mpi);

    ystart=my_rank_mpi*DIM/p_mpi;
    yend=ystart+DIM/p_mpi;
    if(my_rank_mpi==p_mpi-1) yend=DIM;

//    printf("test %d %d %d \n",my_rank_mpi,ystart,yend);

    start=omp_get_wtime();
    kernel(arr,ystart,yend);
    end=omp_get_wtime();

    MPI_Finalize();

    printf("total time diff = %.6g on process %d \n",  end - start,my_rank_mpi);  

    sprintf(filename,"julia.dat.%d",my_rank_mpi);
    out = fopen( filename, "w" );
    for (y=ystart; y<yend; y++) {
        for (x=0; x<DIM; x++) {
            int offset = x + y * DIM;
            if(arr[offset]==1)
                fprintf(out,"%d %d \n",x,y);  
        } 
    } 
    fclose(out);

}


/*
compile and run with:

mpicc -fopenmp -O2 openmp_mpi.c
OMP_NUM_THREADS=2 mpirun -np 4 ./a.out

*/

#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "omp.h"

int main(int argc, char** argv)
{
    int         my_rank_mpi;   /* My process rank           */
    int         p_mpi;         /* The number of processes   */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &p_mpi);

#pragma omp parallel
    {
    int ID_openmp = omp_get_thread_num();
    int p_openmp=omp_get_num_threads();
    printf("Hi from thread %d of %d threads, on MPI process %d of %d \n",
ID_openmp,p_openmp,my_rank_mpi,p_mpi);
    }

    MPI_Finalize();

    return 0;
}

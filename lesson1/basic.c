/* basic.c */
#include "mpi.h"

int main(int argc, char* argv[])
{
/* ... */

/* This must be the first MPI call */
 MPI_Init(&argc, &argv);

/* Identify the process, find total number of processes active */

/* Do computation (with MPI communications) */

 MPI_Finalize();
/* No MPI calls after this line */

/* ... */

  return 0;
}


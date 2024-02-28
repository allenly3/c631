#include "stdio.h"
#include "mpi.h"
#include "math.h"

int main(int argc, char* argv[])
{
  int       p, my_rank;
  MPI_Comm  my_row_comm;
  int       my_row, my_rank_in_row;
  int       q, test;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        
  MPI_Comm_split(MPI_COMM_WORLD, my_rank%2, my_rank,
                 &my_row_comm);

  /* Test the new communicators */
  MPI_Comm_rank(my_row_comm, &my_rank_in_row);
  if (my_rank_in_row == 0) test = my_row;
  else test = 0;

  MPI_Bcast(&test, 1, MPI_INT, 0, my_row_comm);

  printf("Process %d > my_row = %d,"
         "my_rank_in_row = %d, test = %d\n",
         my_rank, my_row, my_rank_in_row, test);
  MPI_Finalize();
  return 0; }

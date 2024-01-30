/* nondet.c */
#include <stdio.h>
#include "mpi.h"
#include <unistd.h>

int compute_task()
{
sleep(1);
return 0;
}

int main(int argc, char* argv[])
{
  int         my_rank;       /* rank of process      */
  int         p;             /* number of processes  */
  int         iter;
  int         itermax=100;
  int         flag;
  int         signal;
  MPI_Request req;

  /* Start up MPI */
  MPI_Init(&argc, &argv);

  /* Find out process rank  */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Find out number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &p);


  if (my_rank == 0)
  {
    sleep(3);
    printf("process 0 sending termination signal to process 1 \n");
    /* send to process 1 */
    MPI_Isend(&signal,1,MPI_INT,1,0,MPI_COMM_WORLD,&req);
    MPI_Wait(&req,MPI_STATUS_IGNORE);
  }

  else
  {
    /* receive from process 0 */
    MPI_Irecv(&signal,1,MPI_INT,0,0,MPI_COMM_WORLD,&req); 

    for (iter=0;iter<itermax;iter++)
      {
        MPI_Test(&req,&flag,MPI_STATUS_IGNORE);
        if(flag)
          {
            printf("process 1 stop inside iteration %d  \n",iter);
            break;
          }
        compute_task();

      printf("process 1 finished iteration  %d \n",iter);
    }

  }

  /* Shut down MPI */
  MPI_Finalize();

  return 0;

}
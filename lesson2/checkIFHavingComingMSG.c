/* 
/* broadcast.c
 * Broadcast and integer from process 0 to all processes with rank != 0
 */
#include "stdio.h"
#include "string.h"
#include "mpi.h"

int main(int argc, char* argv[])
{
  int         my_rank;       /* rank of process      */
  int         p;             /* number of processes  */
  int         source;        /* rank of sender       */
  int         dest;          /* rank of receiver     */
  int         tag = 0;       /* tag for messages     */
  MPI_Status  status;        /* status for receive   */
  int broadcast_integer=555; /*integer for broadcasting */
  int spacing; /*distance between sending processes*/
  int stage; /* stage of algorithm */
  int flag = 0 ; // keep waiting 

  /* Start up MPI */
  MPI_Init(&argc, &argv);

  /* Find out process rank  */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Find out number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &p);

printf("my_rank %d \n", my_rank);
  if(my_rank == 0){
    MPI_Send(&broadcast_integer, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
    printf("sent");
  }

    while(my_rank == 1 && !flag){
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG , MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            printf("Coming MSG \n");
        }
        else{
            printf("Nothing\n");
        }

    }


  /* Shut down MPI */
  MPI_Finalize();



  return 0;
}
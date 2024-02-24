/* greetings.c
 * Send a message from all processes with rank != 0 
 * to process 0.
 * Process 0 prints the messages received.

Compile and run:

mpicc greetings.c -o greetings.x
mpirun -np 4 ./greetings.x

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
  char        message[100];  /* storage for message  */
  MPI_Status  status;        /* status for receive   */
  int  data = -1;
  int i , j; 



  /* Start up MPI */
  MPI_Init(&argc, &argv);
  
  /* Find out process rank  */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  /* Find out number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &p);


  if(my_rank==p-1){
    MPI_Send(&my_rank, 1, MPI_INT, 
      	       0, tag, MPI_COMM_WORLD);
               printf( " p %d sends data\n",my_rank);
  }else{
    MPI_Send(&my_rank, 1, MPI_INT, 
      	       my_rank+1, tag, MPI_COMM_WORLD);
               printf( " p %d sends data\n",my_rank);
  }

    if(my_rank==0){
            MPI_Recv(&data, 1, MPI_INT, p-1, tag, 
                MPI_COMM_WORLD, &status);
                printf( " Rank %d recv %d \n",my_rank,data);
  }else{
            MPI_Recv(&data, 1, MPI_INT, my_rank-1, tag, 
                MPI_COMM_WORLD, &status);
                printf( " Rank %d recv %d \n", my_rank,data);
  }

            

 


 


  
  /* Shut down MPI */
  MPI_Finalize();

  return 0;
} 

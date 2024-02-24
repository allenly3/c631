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
  int         data = 33;
  int         p;             /* number of processes  */
  int         source;        /* rank of sender       */
  int         dest;          /* rank of receiver     */
  int         tag = 0;       /* tag for messages     */
  char        message[100];  /* storage for message  */
  MPI_Status  status;        /* status for receive   */
  
  /* Start up MPI */
  MPI_Init(&argc, &argv);
  
  /* Find out process rank  */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  /* Find out number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (my_rank != 0) 
    {
      /* Create message */
      printf( " p %d sends data\n",my_rank);
      dest = 0;
      data = my_rank;
      /* Use strlen+1 so that '\0' gets transmitted */
      MPI_Send(&my_rank, 1, MPI_INT, 
      	       dest, tag, MPI_COMM_WORLD);
    } 

  else 
    { /* my_rank == 0 */
      for (source = 1; source < p; source++) 
	    {
            MPI_Recv(&data, 1, MPI_INT, source, tag, 
                MPI_COMM_WORLD, &status);
            printf("p0 recv data :%d\n", data);
        }
    }
  
  /* Shut down MPI */
  MPI_Finalize();

  return 0;
} 

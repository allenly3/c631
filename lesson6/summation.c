
#include "stdio.h"
#include "string.h"
#include "mpi.h"

int main(int argc, char* argv[])
{
  int         N=256;
  float       A[N];
  int         my_rank;       /* rank of process      */
  int         p;             /* number of processes  */
  int         source;        /* rank of sender       */
  int         dest;          /* rank of receiver     */
  int         tag = 0;       /* tag for messages     */
  MPI_Status  status;        /* status for receive   */
  float mysum,neighboursum;
  int currentN;
  int i;

  /* Start up MPI */
  MPI_Init(&argc, &argv);

  /* Find out process rank  */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Find out number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &p);

// number of processes must be a power of 2

//initialize A with some arbitrary data
  for (i=0;i<N/p;i++){
    A[i]=1.0;
  }

  mysum=0.0;
  neighboursum=0.0;
  currentN=p;

// start timing

  for (i=0;i<N/p;i++){
    mysum=mysum+A[i];
  }

// communication part
  while(currentN>1){

  if(my_rank<currentN/2){
    source=my_rank+currentN/2;
    MPI_Recv(&neighboursum, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
  }

  else if(my_rank<currentN){
    dest=my_rank-currentN/2;
    MPI_Send(&mysum, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
  }

  currentN=currentN/2;
//end of communication part

// summation part
  mysum=mysum+neighboursum;

}

  if(my_rank==0){
    printf("total sum %f \n",mysum);
  }

  /* Shut down MPI */
  MPI_Finalize();

  return 0;
}
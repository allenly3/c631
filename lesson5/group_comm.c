#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

#define NPROCS 4 //8

int main(int argc, char *argv[]) {
  int rank, new_rank,
    sendbuf, recvbuf, numtasks;
  int ranks1[2]={0,1};
  int ranks2[2]={2,3};

  MPI_Group  orig_group, new_group;
  MPI_Comm   new_comm;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

if (numtasks != NPROCS && rank==0) {
      printf("Must specify MP_PROCS = %d.\n", NPROCS);
      MPI_Finalize(); exit(0); }

  /* store the global rank in sendbuf */
  sendbuf = rank;

  /* Extract the original group handle */
  MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

  /* Divide tasks into two distinct groups based upon rank */
  if (rank < numtasks/2)
    /* if rank = 0,1,2,3, put original processes 0,1,2,3
       into new_group */
    MPI_Group_incl(orig_group, 2, ranks1, &new_group);
  else
    /* if rank = 4,5,6,7, put original processes 4,5,6,7
       into new_group */
    MPI_Group_incl(orig_group, 2, ranks2, &new_group);

/* Create new new communicator and then perform collective
     communications */
  MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

  /* new_comm contains a group with processes 0,1,2,3
     on processes 0,1,2,3 */
  /* new_comm contains a group with processes 4,5,6,7
     on processes 4,5,6,7 */
  MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_INT,
                MPI_SUM, new_comm);

  /* new_rank is the rank of my processs in the new group */
  MPI_Group_rank (new_group, &new_rank);

  printf("rank= %d newrank= %d recvbuf= %d\n",
         rank,new_rank,recvbuf);

  MPI_Finalize();
  return 0; }
		
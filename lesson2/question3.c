/* main_parallel_matrix_vector.c */
#include <stdio.h>
#include "mpi.h"
#include "matrix_vector.h"

void Parallel_matrix_vector_prod
( LOCAL_MATRIX_T  local_A, int m, int n,
  float local_x[], float global_x[], float local_y[],
  int   local_m, int local_n)
{
  /* local_m = m/p, local_n = n/p */
  int i, j;

  MPI_Allgather(local_x, local_n, MPI_FLOAT,
                global_x, local_n, MPI_FLOAT,
                MPI_COMM_WORLD);

  for (i = 0; i < local_m; i++)
    {
      local_y[i] = 0.0;
      for (j = 0; j < n; j++)
        local_y[i] = local_y[i] +
          local_A[i][j]*global_x[j];
    }
}

void Read_vector(char *prompt, float local_x[], int local_n,
                 int my_rank, int p)
{
  int   i;
  float temp_vector[MAX_ORDER];

  if (my_rank == 0)
    {
      printf("%s\n", prompt);
      for (i = 0; i < p*local_n; i++)
        scanf("%f", &temp_vector[i]);
    }

  MPI_Scatter(temp_vector, local_n, MPI_FLOAT,
              local_x, local_n, MPI_FLOAT,
              0, MPI_COMM_WORLD);

}

void Read_matrix(char *prompt, LOCAL_MATRIX_T  local_A,
                 int local_m, int n, int my_rank,
                 int p)
{
  int i, j;
  LOCAL_MATRIX_T  temp_matrix;

  /* Fill entries in temp_matrix with zeros, for subsequent overwrite */
  for (i = 0; i < p*local_m; i++)
    for (j = n; j < MAX_ORDER; j++)
      temp_matrix[i][j] = 0.0;

  if (my_rank == 0)
    {
      printf("%s\n", prompt);
      for (i = 0; i < p*local_m; i++)
        for (j = 0; j < n; j++)
          scanf("%f",&temp_matrix[i][j]);
    }
    MPI_Scatter(temp_matrix, local_m*MAX_ORDER, MPI_FLOAT,
              local_A, local_m*MAX_ORDER, MPI_FLOAT,
              0, MPI_COMM_WORLD);
}
	
void Print_vector(char *title, float  local_y[] ,
                  int local_m, int my_rank,
                  int p)
{
  int   i;
  float temp_vector[MAX_ORDER];

  MPI_Gather(local_y, local_m, MPI_FLOAT,
             temp_vector, local_m, MPI_FLOAT,
             0, MPI_COMM_WORLD);

  if (my_rank == 0)
    {
      printf("%s\n", title);
      for (i = 0; i < p*local_m; i++)
        printf("%4.1f ", temp_vector[i]);
      printf("\n");
    }
}

 void Print_matrix(char *title, LOCAL_MATRIX_T  local_A,
                  int local_m, int n, int my_rank, int p)
{
  int   i, j;
  float temp_matrix[MAX_ORDER][MAX_ORDER];

  MPI_Gather(local_A, local_m*MAX_ORDER, MPI_FLOAT,
             temp_matrix, local_m*MAX_ORDER, MPI_FLOAT,
             0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    printf("%s\n", title);
    for (i = 0; i < p*local_m; i++)
      {
        for (j = 0; j < n; j++)
          printf("%4.1f ", temp_matrix[i][j]);
        printf("\n");
      }
  }
}      

int main(int argc, char* argv[])
{
  int             my_rank, p;
  LOCAL_MATRIX_T  local_A;
  float           global_x[MAX_ORDER];
  float           local_x[MAX_ORDER];
  float           local_y[MAX_ORDER];
  int             m, n;
  int             local_m, local_n;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0)
    {
      printf("Enter the dimensions of the matrix (m n)\n");
      scanf("%d %d", &m, &n);
    }
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  local_m = m/p;
  local_n = n/p;

  Read_matrix("Enter the matrix values",
              local_A, local_m, n, my_rank, p);
  Print_matrix("Printing matrix for verification",
               local_A, local_m, n, my_rank, p);

  Read_vector("Enter the vector values",
              local_x, local_n, my_rank, p);
  Print_vector("Printing vector for verification",
               local_x, local_n, my_rank, p);

  Parallel_matrix_vector_prod(local_A, m, n, local_x,
                              global_x, local_y, local_m,
                              local_n);
  Print_vector("The resulting product vector is", local_y, local_m,
               my_rank, p);

  MPI_Finalize();
  return 0;
}


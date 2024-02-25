#include <stdio.h>
#include "mpi.h"

float f(float x) 
{
    return x*x;
} 


void Get_data(float* a_ptr, float* b_ptr, int* n_ptr,
               int my_rank)
{
    if (my_rank == 0)
    {
        printf("Enter a, b, and n\n");
        scanf("%f %f %d", a_ptr, b_ptr, n_ptr);
    }
    MPI_Bcast(a_ptr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_ptr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_ptr, 1, MPI_INT,   0, MPI_COMM_WORLD);
}

float Trapezoid(float a, float b, int n, float h)
{
    float integral;   /* result of integration  */
    float x;
    int i;
  
    integral = (f(a) + f(b))/2.0;
  
    x = a;
    for ( i = 1; i <= n-1; i++ ) 
    {
        x = x + h;
        integral = integral + f(x);
    }
  
    return integral*h;
} 


int main(int argc, char** argv)
{
  int         my_rank, p;
  float       a, b, h;
  int         n;
  float       local_a, local_b, local_n;
  float       integral;  /* Integral over my interval */
  float       total;     /* Total integral            */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  Get_data(&a, &b, &n, my_rank);

  h = (b-a)/n;
  local_n = n/p;

  local_a = a + my_rank*local_n*h;
  local_b = local_a + local_n*h;
  integral = Trapezoid(local_a, local_b, local_n, h);

  /* Add up the integrals calculated by each process */
  MPI_Reduce(&integral, &total, 1, MPI_FLOAT,
             MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0)
    {
      printf("With n = %d trapezoids, our estimate\n", n);
      printf("of the integral from %f to %f = %f\n",
             a, b, total);
    }

  MPI_Finalize();
  return 0;
}

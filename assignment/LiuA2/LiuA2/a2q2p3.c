#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define BLOCK_SIZE 10 // block size for the computation

// compute a block of the cumulative sum array B in parallel using OpenMP tasks
void computeBlock(int start_row, int end_row, int start_col, int end_col, int **A, int **B) {
    int i , j ;
    for ( i = start_row; i < end_row; ++i) {
        for ( j = start_col; j < end_col; ++j) {
            if (i == 0 && j == 0) {
                B[i][j] = A[i][j];
            } else if (i == 0) {
                B[i][j] = B[i][j - 1] + A[i][j];
            } else if (j == 0) {
                B[i][j] = B[i - 1][j] + A[i][j];
            } else {
                B[i][j] = A[i][j] + B[i - 1][j] + B[i][j - 1] - B[i - 1][j - 1];
            }
        }
    }
}

void computeCumulativeSumParallel(int rows, int cols, int **A, int **B)
{
    int i , j ; 
    #pragma omp parallel
    #pragma omp single
    {
        for ( i = 0; i < rows; i += BLOCK_SIZE) {
            for (j = 0; j < cols; j += BLOCK_SIZE) {
                #pragma omp task firstprivate(i, j) shared(A, B)
                {
                    int start_row = i;
                    int end_row = (i + BLOCK_SIZE < rows) ? i + BLOCK_SIZE : rows;
                    int start_col = j;
                    int end_col = (j + BLOCK_SIZE < cols) ? j + BLOCK_SIZE : cols;
                    computeBlock(start_row, end_row, start_col, end_col, A, B);
                }
            }
        }
    }
}

// print the 2D array
void printArray(int rows, int cols, int **array) {
    int i , j;
    for ( i = 0; i < rows; i++) {
        for ( j = 0; j < cols; j++) {
            printf("%4d ", array[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int i ,j ;
    double start_time, end_time;
    if (argc < 3) {
        printf("Usage: %s <rows> <cols>\n", argv[0]);
        return 1;
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);

    if (rows <= 0 || cols <= 0) {
        printf("Invalid dimensions. Please enter positive dimensions.\n");
        return 1;
    }

    // allocate memory for the 2D arrays
    int **A = (int **)malloc(rows * sizeof(int *));
    int **B = (int **)malloc(rows * sizeof(int *));
    for ( i = 0; i < rows; i++) {
        A[i] = (int *)malloc(cols * sizeof(int));
        B[i] = (int *)malloc(cols * sizeof(int));
    }

    // seed the random number generator
    srand((unsigned int)time(NULL));

    // generate a random matrix A
    for ( i = 0; i < rows; i++) {
        for ( j = 0; j < cols; j++) {
            A[i][j] = rand() % 11; // Random number between 0 and 10
        }
    }

    start_time = MPI_Wtime();;
    // compute the cumulative sum array B in parallel
    computeCumulativeSumParallel(rows, cols, A, B);
    end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    printf("Time cost: %fs.\n", elapsed);

    // print the original array A
    // printf("Array A:\n");
    // printArray(rows, cols, A);

    // print the computed array B
    // printf("\nCumulative Sum Array B:\n");
    // printArray(rows, cols, B);

    // free memory
    for (i = 0; i < rows; i++) {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);

    return 0;
}

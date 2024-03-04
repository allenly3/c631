#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// Function to compute the cumulative sum array B
void computeCumulativeSum(int rows, int cols, int **A, int **B) {
    int i, j;

    // Initialize the first element of B
    B[0][0] = A[0][0];

    // Initialize the first column of B
    for (i = 1; i < rows; i++) {
        B[i][0] = B[i-1][0] + A[i][0];
    }

    // Initialize the first row of B
    for (j = 1; j < cols; j++) {
        B[0][j] = B[0][j-1] + A[0][j];
    }

    // Compute the rest of B
    for (i = 1; i < rows; i++) {
        for (j = 1; j < cols; j++) {
            // Add current element, and subtract the overlap to avoid double-counting
            B[i][j] = A[i][j] + B[i-1][j] + B[i][j-1] - B[i-1][j-1];
        }
    }
}

// print the 2D array
void printArray(int rows, int cols, int **array) {
    int i , j;
    for (i = 0; i < rows; i++) {
        for ( j = 0; j < cols; j++) {
            printf("%4d ", array[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    double start_time, end_time;
    int i , j;

    if (argc < 3) {
        printf("NEED 2 Arguments: %s <rows> <cols>\n" );
        return 1;
    }

    // int rows = 300;
    // int cols = 300;
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
        for (j = 0; j < cols; j++) {
            A[i][j] = rand() % 11; // Random number between 0 and 10
        }
    }


    start_time = MPI_Wtime();
    // compute the cumulative sum array B in parallel
    computeCumulativeSum(rows, cols, A, B);
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

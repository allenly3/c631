#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define N 10 // Size of the grid
#define ITERATIONS 10 // Number of iterations

// Function to initialize the grid with random values
void initialize_grid(int *grid, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        grid[i] = rand() % 2; // Randomly set cell to 0 or 1
    }
}

// Function to print the grid
void print_grid(int *grid, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", grid[i]);
        if ((i + 1) % N == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

// Function to count the number of live neighbors for a cell
int count_live_neighbors(int *grid, int size, int index) {
    int count = 0;
    int row = index / N;
    int col = index % N;
    for (int i = row - 1; i <= row + 1; i++) {
        for (int j = col - 1; j <= col + 1; j++) {
            if (i >= 0 && i < N && j >= 0 && j < N && (i != row || j != col)) {
                count += grid[i * N + j];
            }
        }
    }
    return count;
}

// Function to update the grid based on Conway's rules
void update_grid(int *grid, int size) {
    int *new_grid = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        int live_neighbors = count_live_neighbors(grid, size, i);
        if (grid[i] == 1) {
            if (live_neighbors < 2 || live_neighbors > 3) {
                new_grid[i] = 0; // Cell dies
            } else {
                new_grid[i] = 1; // Cell survives
            }
        } else {
            if (live_neighbors == 3) {
                new_grid[i] = 1; // Cell becomes alive
            } else {
                new_grid[i] = 0; // Cell remains dead
            }
        }
    }
    // Copy new grid back to original grid
    for (int i = 0; i < size; i++) {
        grid[i] = new_grid[i];
    }
    free(new_grid);
}

int main(int argc, char **argv) {
    int my_rank, num_procs;
    int *grid = (int *)malloc(N * N * sizeof(int));

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Initialize grid on process 0
    if (my_rank == 0) {
        initialize_grid(grid, N * N);
    }

    // Scatter grid to all processes
    MPI_Bcast(grid, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform iterations of the game
    for (int iter = 0; iter < ITERATIONS; iter++) {
        update_grid(grid, N * N);
    }

    // Gather final grid from all processes to process 0
    MPI_Gather((my_rank == 0) ? MPI_IN_PLACE : grid, N * N, MPI_INT, grid, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Print final grid on process 0
    if (my_rank == 0) {
        printf("Final grid:\n");
        print_grid(grid, N * N);
    }

    // Finalize MPI
    MPI_Finalize();

    // Free allocated memory
    free(grid);

    return 0;
}
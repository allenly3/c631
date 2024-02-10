#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to determine the range of indices for a subtree rooted at root_index
void get_subtree_indices(int root_index, int* start_index, int* end_index) {
    *start_index = 2 * root_index + 1; // Left child index
    *end_index = 2 * root_index + 2;   // Right child index
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int root_index = 0;
    int start_index, end_index;
    get_subtree_indices(root_index, &start_index, &end_index);

   
    int in_subtree = (rank >= start_index && rank <= end_index);


    MPI_Comm subtree_comm;
    MPI_Comm_split(MPI_COMM_WORLD, in_subtree, rank, &subtree_comm);

    if (in_subtree) {
        int subtree_rank;
        MPI_Comm_rank(subtree_comm, &subtree_rank);
        printf("Rank %d belongs to the subtree. Subtree rank: %d\n", rank, subtree_rank);
    }

    MPI_Finalize();
    return 0;
}
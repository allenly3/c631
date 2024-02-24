#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_POINTS 1000000

int main() {
    int i;
    int num_points_inside_circle = 0;
    double x, y, pi;

    // Seed the random number generator
    srand(time(NULL));

    // Generate random points and count how many fall inside the circle
    for (i = 0; i < NUM_POINTS; i++) {
        x = (double)rand() / RAND_MAX; // Generate random x-coordinate
        y = (double)rand() / RAND_MAX; // Generate random y-coordinate

        // Check if the point falls inside the circle (radius = 1)
        if (x * x + y * y <= 1) {
            num_points_inside_circle++;
        }
    }

    // Estimate the value of Pi
    pi = 4.0 * num_points_inside_circle / NUM_POINTS;

    printf("Estimated value of Pi: %f\n", pi);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//gcc agents.c -o ag -lm -Ofast -fsanitize=address -static-libasan -g -Wall && ./ag

double bindist(int n, int k, double p) {
    double comb = tgamma(n + 1) / (tgamma(k + 1) * tgamma(n - k + 1));
    return comb * pow(p, k) * pow(1 - p, n - k);
}


int main() {
    int nindivs = 10; // set the number of individuals
    int nloci = 5; // set the number of loci
    float skew = 0.2; // set the skew parameter
    int i, j;
    
    int **pop = (int **)malloc(nindivs * sizeof(int *)); // allocate memory for the population
    
    for (i = 0; i < nindivs; i++) {
        pop[i] = (int *)malloc(nloci * sizeof(int));
    }
    
    srand(time(0)); // seed the random number generator with the current time
    
    for (i = 0; i < nindivs; i++) {
        for (j = 0; j < nloci; j++) {
            float r = (float)rand() / RAND_MAX; // generate a random float between 0 and 1
            if (r < skew) {
                pop[i][j] = 1; // set the locus to 1 with probability skew
            } else {
                pop[i][j] = 0; // set the locus to 0 with probability 1 - skew
            }
        }
    }
    
    // print the population
    printf("Population:\n");
    for (i = 0; i < nindivs; i++) {
        for (j = 0; j < nloci; j++) {
            printf("%d ", pop[i][j]);
        }
        printf("\n");
    }
    
    // free the allocated memory
    for (i = 0; i < nindivs; i++) {
        free(pop[i]);
    }
    free(pop);
    
    return 0;
}
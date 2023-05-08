#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//gcc oc_tensor_generator.c -o oc_TG -lm -Ofast -fsanitize=address -static-libasan -g -Wall && ./oc_TG

double bindist(int n, int k, double p) {
    double comb = tgamma(n + 1) / (tgamma(k + 1) * tgamma(n - k + 1));
    return comb * pow(p, k) * pow(1 - p, n - k);
}

double oc(double v, double n, double i, double j) {
    double sumvar = 0;
    int v_int = (int)v;
    int n_int = (int)n;
    int i_int = (int)i;
    int j_int = (int)j;
    for (int x = 0; x <= i_int; x++) {
        double comb1 = tgamma(i_int + 1) / (tgamma(x + 1) * tgamma(i_int - x + 1));
        double comb2 = tgamma(n_int - i_int + 1) / (tgamma(j_int - x + 1) * tgamma(n_int - i_int - j_int + x + 1));
        double comb3 = tgamma(n_int + 1) / (tgamma(j_int + 1) * tgamma(n_int - j_int + 1));
        sumvar += comb1 * comb2 / comb3 * bindist(i_int + j_int - 2 * x, v_int - x, 0.5);
    }
    return sumvar;
}

int main() {

    int nloci = 20;
    int nstates = nloci + 1;


    // create 2D array for x and y
    double *x_data = (double *)malloc(nstates * nstates * sizeof(double));
    double *y_data = (double *)malloc(nstates * nstates * sizeof(double));

    double **x = (double **)malloc(nstates * sizeof(double *));
    double **y = (double **)malloc(nstates * sizeof(double *));
    for (int i = 0; i < nstates; i++) {
        x[i] = x_data + i * nstates;
        y[i] = y_data + i * nstates;
        for (int j = 0; j < nstates; j++) {
            x[i][j] = i;
            y[i][j] = j;
        }
    }


    // create 1D array for n_list
    double *n_list = (double *)malloc(nstates * nstates * sizeof(double));
    for (int i = 0; i < nstates * nstates; i++) {
        n_list[i] = nloci;
    }

    // create 3D array for oc_tensor
    double *oc_tensor = (double *)malloc(nstates * nstates * nstates * sizeof(double));

    for (int v = 0; v < nstates; v++) {
        printf("v=%d\n", v);
        double *v_list = (double *)malloc(nstates * nstates * sizeof(double));
        for (int i = 0; i < nstates * nstates; i++) {
            v_list[i] = v;
        }

        for (int i = 0; i < nstates; i++) {
            for (int j = 0; j < nstates; j++) {
                oc_tensor[v * nstates * nstates + i * nstates + j] = oc(v_list[i * nstates + j], n_list[i * nstates + j], x[i][j], y[i][j]);
            }
        }
        free(v_list);
    }

    // Open file for writing
    FILE* f = fopen("data.bin", "wb");

    // Write data to file
    fwrite(oc_tensor, sizeof(double), nstates*nstates*nstates, f);

    // Close file
    fclose(f);

    free(x);
    free(y);
    free(x_data);
    free(y_data);
    free(oc_tensor);
    free(n_list);

    return 0;
}

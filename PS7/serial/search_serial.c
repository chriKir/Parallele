// search.c
// generates a random array and searches one element
// number of elements and repetitions are passed as parameters
// at least 1000 repetitions should be used to get a statistically relevant result
// Peter Thoman

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include "time_ms.h"
#include "dSFMT.h"

int main(int argc, char** argv) {
    if(argc != 3) {
        printf("Usage: search [elements] [iterations]\nExample: search 1000000 1000\n");
        return -1;
    }

    int elems = atoi(argv[1]), iters = atoi(argv[2]);

    // allocate memory for data set
    double *data = (double*)malloc(elems*sizeof(double));

    // initialize random number generator
    dsfmt_t rand_state;
    dsfmt_init_gen_rand(&rand_state, (uint32_t)time(NULL));

    // initialize data set (fill randomly)
    for(int j=0; j<elems; ++j) {
        data[j] = dsfmt_genrand_close1_open2(&rand_state);
    }

    double epsilon = 0.4/(double)elems;
    unsigned long long total_time = 0, total_found = 0;

    for(int i=0; i<iters; ++i) {
        // search
        unsigned long long start_time = time_ms();
        double val = dsfmt_genrand_close1_open2(&rand_state);
        bool found = false;
        for(int j=0; j<elems; ++j) {
            if(fabs(data[j] - val) < epsilon) {
                found = true;
                total_found++;
                break;
            }
        }
        total_time += time_ms() - start_time;
    }

    printf("Done, took %12llu ms, found %12llu\n", total_time, total_found);

    free(data);
}

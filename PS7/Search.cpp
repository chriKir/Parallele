//
// Created by Patrick Lanzinger on 28.11.16.
//

#include <ClWrapper.h>
#include "Search.h"

extern "C" {
#include <dSFMT.h>
}


#define SIZE 1000
#define VALUE double

dsfmt_t rand_state;

Search::Search() {
    //init random number generator
    dsfmt_init_gen_rand(&rand_state, (uint32_t)time(NULL));
}

void Search::fill_array_with_random_numbers(double *array) {
    for(int i = 0; i < SIZE;i++) {
        array[i] = dsfmt_genrand_close1_open2(&rand_state);
    }
}

void Search::execute(int iterations) {
    try {
        //init
        ClWrapper wrapper("../../PS7/search.cl", -1);
        wrapper.Build("search");

        VALUE data[SIZE];
        float found = 0;
        double epsilon = 0.4/(double)SIZE;

        //create buffer and set args
        cl::Buffer data_buffer = wrapper.AddBuffer(CL_READ_ONLY_CACHE, 0, sizeof(VALUE) * SIZE);
        cl::Buffer found_buffer = wrapper.AddBuffer(CL_READ_WRITE_CACHE, 1, sizeof(float));
        cl::Buffer epsilon_buffer = wrapper.AddBuffer(CL_READ_ONLY_CACHE,2, sizeof(double));

        //------------------------------------------

        unsigned long long total_found = 0;
        for(int i = 0;i < iterations;i++) {
            //fill data array
            fill_array_with_random_numbers(data);

            //write in buffer
            wrapper.WriteBuffer(data_buffer, data, 0);
            wrapper.WriteBuffer(found_buffer, &found, 1);
            wrapper.WriteBuffer(epsilon_buffer, &epsilon, 2);

            //execute buffer
            cl::NDRange global(SIZE);

            wrapper.Run(cl::NullRange, global);
            wrapper.ReadBuffer(found_buffer, 1, &found);

            if(found == 1) {
                total_found++;
            }
        }

        //---------------------------------------------
        std::cout << total_found << "\n";

    } catch (const cl::Error &e) {
        std::cerr << "OpenCL exception: " << e.what() << " : " << ClWrapper::get_error_string(e.err()) << "\n";

    } catch (const std::exception &e) {
        std::cout << std::flush;
        std::cerr << std::flush << "Exception thrown: " << e.what() << "\n";
        exit(1);
    }



}

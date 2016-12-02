//
// Created by Patrick Lanzinger on 28.11.16.
//

#include <ClWrapper.h>
#include <include/time_ms.h>
#include "Search.h"

extern "C" {
#include <dSFMT.h>
}


#define VALUE double

dsfmt_t rand_state;

Search::Search() {
    //init random number generator
    dsfmt_init_gen_rand(&rand_state, (uint32_t) time(NULL));
}

void Search::fill_array_with_random_numbers(double *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = dsfmt_genrand_close1_open2(&rand_state);
    }
}

void Search::execute(int iterations, int size) {
    try {
        //init
        ClWrapper wrapper("search.cl", 0);
        wrapper.Build("search");

        VALUE data[size];
        float found = 0;
        double epsilon = 0.4 / (double) size;

        //create buffer and set args
        cl::Buffer data_buffer = wrapper.AddBuffer(CL_READ_ONLY_CACHE, 0, sizeof(VALUE) * size);
        cl::Buffer val_buffer = wrapper.AddBuffer(CL_READ_ONLY_CACHE, 1, sizeof(VALUE));
        cl::Buffer found_buffer = wrapper.AddBuffer(CL_READ_WRITE_CACHE, 2, sizeof(float));
        cl::Buffer epsilon_buffer = wrapper.AddBuffer(CL_READ_ONLY_CACHE, 3, sizeof(double));

        //------------------------------------------

        unsigned long long total_host_time = 0;
        unsigned long long start_time = time_ms();

        unsigned long long total_found = 0;
        for (int i = 0; i < iterations; i++) {
            //fill data array
            fill_array_with_random_numbers(data, size);
            double val = dsfmt_genrand_close1_open2(&rand_state);
            found = 0;

            //write in buffer
            wrapper.WriteBuffer(data_buffer, data, 0);
            wrapper.WriteBuffer(val_buffer, &val, 1);
            wrapper.WriteBuffer(found_buffer, &found, 2);
            wrapper.WriteBuffer(epsilon_buffer, &epsilon, 3);

            //execute buffer
            cl::NDRange global(size);
            wrapper.Run(cl::NullRange, global);
            wrapper.ReadBuffer(found_buffer, 2, &found);


            if (found == 1) {
                total_found++;
            }

        }

        //---------------------------------------------
        total_host_time += time_ms() - start_time;
        double total_cl_time = wrapper.getTotalExecutionTime();
        std::cout << std::fixed << "Total OpenCl Profiling: " << total_cl_time << "ms, found " << total_found
                  << std::endl;
        std::cout << std::fixed << "Total host ex. time: " << total_host_time << "ms" << std::endl;
        wrapper.printProfilingInfo();

    } catch (const cl::Error &e) {
        std::cerr << "OpenCL exception: " << e.what() << " : " << ClWrapper::get_error_string(e.err()) << "\n";

    } catch (const std::exception &e) {
        std::cout << std::flush;
        std::cerr << std::flush << "Exception thrown: " << e.what() << "\n";
        exit(1);
    }


}

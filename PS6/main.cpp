//
// Created by roland on 11/23/16.
//
#include <iostream>
#include <iomanip>

#include "ClWrapper.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"
#define VALUE cl_int

int main() {

    try {

        ClWrapper cl("kernel.c", -1);

        std::cout << std::setw(7) << "version" << std::setw(7) << "N" << std::setw(15) << "Time" << std::endl;

        cl_uint N = 1024;

        for (cl_uint n = 4; n < N; n *= 2) {
            VALUE *vec = (VALUE *) malloc(sizeof(VALUE) * N);
            VALUE *result = (VALUE *) malloc(sizeof(VALUE) * N);

            for (size_t i = 0; i < N; i++) { result[i] = 0; }


            cl.Build("reduction_v1");

            cl::Buffer b_array = cl.AddBuffer(CL_READ_ONLY_CACHE, 0, sizeof(VALUE) * N);
            cl::Buffer b_result = cl.AddBuffer(CL_READ_WRITE_CACHE, 1, sizeof(VALUE) * N);

            cl.WriteBuffer(b_array, vec, 0, sizeof(VALUE) * N);
            cl.WriteBuffer(b_result, result, 1, sizeof(VALUE) * N);

            cl::NDRange global(N);

            cl.Run(cl::NullRange, global);

            cl.ReadBuffer(b_result, 2, sizeof(VALUE) * N, result);

            std::cout << std::setw(7) << "1" << std::setw(7) << N << std::setw(15) << cl.getTotalExecutionTime() << "ms"
                      << std::setw(25) << std::endl;
        }
        return 0;

    } catch (const cl::Error &e) {
        std::cerr << "OpenCL exception: " << e.what() << " : " << ClWrapper::get_error_string(e.err());

    } catch (const std::exception &e) {
        std::cout << std::flush;
        std::cerr << std::flush << "Exception thrown: " << e.what();
        return -1;
    }

}

#pragma clang diagnostic pop

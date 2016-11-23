//
// Created by Roland Gritzer on 13.10.16.
//

#include <iostream>
#include <cmath>
#include <malloc.h>

#include "ClWrapper.h"

//#define PRINT_MTX
#define PROFILING

void printMatrix(int rows, int columns, cl_float *matrix) {

    for (int i = 0; i < rows; ++i) {
        std::string line = "";

        for (int j = 0; j < columns; ++j) {
            line.append("[");
            line.append(std::to_string(matrix[i * columns + j]));
            line.append("]\t");
        }
        std::cout << line << "\n";
    }
}

float init_func(int x, int y) {
    return 40 * std::sin((float) (16 * (2 * x - 1) * y));
}


int main() {

    try {

        ClWrapper *loader = new ClWrapper("jacobi.c", 0);

        loader->Build("jacobi");

        for (cl_uint matrix_size = 4; matrix_size < 1025; matrix_size *= 2) {
//        for (cl_uint matrix_size = 4; matrix_size < 5; matrix_size *= 2) {

            for (int iterations = 100; iterations < 100000; iterations *= 10) {
//            for (int iterations = 100; iterations < 101; iterations *= 10) {

                cl_float factor = (cl_float) std::pow((cl_float) 1 / matrix_size, 2);

                std::cout << std::endl << (int) matrix_size << "x" << (int) matrix_size << "/" << iterations << ": ";

                cl_float *f = (cl_float *) std::malloc(matrix_size * matrix_size * sizeof(cl_float));
                cl_float *u1 = (cl_float *) std::malloc(matrix_size * matrix_size * sizeof(cl_float));
                cl_float *u2 = (cl_float *) std::malloc(matrix_size * matrix_size * sizeof(cl_float));

                for (cl_uint i = 0; i < matrix_size; i++)
                    for (cl_uint j = 0; j < matrix_size; j++)
                        f[i * matrix_size + j] = init_func(i, j);

                for (cl_uint i = 0; i < matrix_size * matrix_size; i++) { u1[i] = (cl_float) 0; }
                for (cl_uint i = 0; i < matrix_size * matrix_size; i++) { u2[i] = (cl_float) 0; }

                cl::Buffer buffer_f = loader->AddBuffer(CL_MEM_READ_ONLY, 0, matrix_size * matrix_size * sizeof(cl_float));
                cl::Buffer buffer_u1 = loader->AddBuffer(CL_MEM_READ_ONLY, 1, matrix_size * matrix_size * sizeof(cl_float));
                cl::Buffer buffer_u2 = loader->AddBuffer(CL_MEM_READ_WRITE, 2, matrix_size * matrix_size * sizeof(cl_float));

                loader->setKernelArg(&factor, 3, sizeof(cl_float));
                loader->setKernelArg(&matrix_size, 4, sizeof(cl_uint));

                cl::NDRange global(matrix_size, matrix_size);

                for (int i = 0; i < iterations; i++) {

                    loader->Run(cl::NullRange, global);

                    void* b_u1 = i % 2 == 0 ? &buffer_u1 : &buffer_u2;
                    void* b_u2 = i % 2 == 0 ? &buffer_u2 : &buffer_u1;

                    loader->setKernelArg(b_u1, 1, sizeof(cl::Buffer));
                    loader->setKernelArg(b_u2, 2, sizeof(cl::Buffer));
                }

                loader->ReadBuffer((iterations % 2 == 0) ? buffer_u1 : buffer_u2, (iterations % 2 == 0) ? 1 : 2, matrix_size * matrix_size * sizeof(cl_float), u1);

#ifdef PRINT_MTX
                std::cout << "\n\nx=\n";
                printMatrix(matrix_size, matrix_size, u1);
#endif


                loader->PrintProfileInfo();

                free(f);
                free(u1);
                free(u2);

            }
        }

        delete loader;

        return 0;

    } catch (const std::exception &e) {

        std::cout << std::flush;
        std::cerr << std::flush << "Exception thrown: " << e.what();

        return -1;
    }
}


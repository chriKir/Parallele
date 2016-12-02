//
// Created by Roland Gritzer on 13.10.16.
//

#include <iostream>
#include <cmath>

#include "ClWrapper.h"

//#define PRINT_MTX

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
    return 40 * std::sin((float)(16 * (2 * x - 1) * y));
}


int main() {

    try {

        ClWrapper *loader = new ClWrapper("jacobi.cl", -1);

        loader->Build("jacobi");

        for (cl_uint matrix_size = 4; matrix_size < 1025; matrix_size *= 2) {

            for (int iterations = 100; iterations < 100000; iterations *= 10) {

                cl_float factor = std::pow((float)(1/matrix_size), 2);

                loader->setKernelArg(&factor, 3, sizeof(cl_float));
                loader->setKernelArg(&matrix_size, 4, sizeof(cl_uint));

                std::cout << std::endl << (int) matrix_size << "x" << (int) matrix_size << "/" << iterations << ": ";

                cl_float *f = (cl_float *) std::malloc(matrix_size * matrix_size * sizeof(cl_float));
                cl_float *tmp = (cl_float *) std::malloc(matrix_size * matrix_size * sizeof(cl_float));
                cl_float *u = (cl_float *) std::malloc(matrix_size  * matrix_size* sizeof(cl_float));

                for (cl_uint i = 0; i < matrix_size; i++)
                    for (cl_uint j = 0; j < matrix_size; j++)
                        f[i + j * matrix_size] = init_func(i, j);


                for (cl_uint i = 0; i < matrix_size * matrix_size; i++) { u[i] = (cl_float) 0; }

#ifdef PRINT_MTX
                printMatrix(matrix_size, matrix_size, A);
                std::cout << "*x=\n";
                printMatrix(matrix_size, 1, b);
#endif

                cl::Buffer buffer_f = loader->AddBuffer(CL_MEM_READ_ONLY, 0, matrix_size * matrix_size * sizeof(cl_float));
                cl::Buffer buffer_tmp = loader->AddBuffer(CL_MEM_READ_WRITE, 1, matrix_size * matrix_size * sizeof(cl_float));
                cl::Buffer buffer_u = loader->AddBuffer(CL_MEM_READ_ONLY, 2, matrix_size * matrix_size * sizeof(cl_float));

                int iteration = 0;

                cl::NDRange global(matrix_size, matrix_size);

                do {

                    loader->Run(cl::NullRange, global);

                    loader->ReadBuffer(buffer_tmp, 1, matrix_size * matrix_size * sizeof(cl_float), tmp);

#ifdef PRINT_MTX
                    std::cout << "\n\nx=\n";
                    printMatrix(matrix_size, 1, x2);
#endif


                    loader->ReWriteBuffer(buffer_u, tmp, 2, matrix_size * matrix_size* sizeof(cl_float));

                    iteration++;

                } while (iteration < iterations);

                loader->printProfilingInfo();

                free(f);
                free(tmp);
                free(u);

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


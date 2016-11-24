#include <iostream>
#include <iomanip>

#include "ClWrapper.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"
#define DTYPE_COLOR_VALUE cl_int

void fillArray(DTYPE_COLOR_VALUE *vec, size_t N);

void printArray(DTYPE_COLOR_VALUE *vec, size_t N);

bool validateSum(DTYPE_COLOR_VALUE sum, size_t N);

void executev1(ClWrapper cl, cl_uint N);

void executev2(ClWrapper cl, cl_uint N);

void executev3(ClWrapper cl, cl_uint N, cl_uint WORKGROUP_SIZE);

int main() {

    try {

        ClWrapper cl("reduction.cl", -1);

        std::cout << std::setw(7) << "version" << std::setw(7) << "N" << std::setw(15) << "Time" << std::endl;

        cl_uint N = 1024;
        cl_uint WORKGROUP_SIZE = 4;

        for (cl_uint n = 4; n < N; n *= 2) {
            executev1(cl, n);
            executev2(cl, n);
            executev3(cl, n, WORKGROUP_SIZE);
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

void executev1(ClWrapper cl, cl_uint N) {

    DTYPE_COLOR_VALUE *vec = (DTYPE_COLOR_VALUE *) malloc(sizeof(DTYPE_COLOR_VALUE) * N);
    DTYPE_COLOR_VALUE *result = (DTYPE_COLOR_VALUE *) malloc(sizeof(DTYPE_COLOR_VALUE) * N);

    fillArray(vec, N);
    for (size_t i = 0; i < N; i++) { result[i] = 0; }


    cl.Build("reduction_v1");

    cl::Buffer b_array = cl.AddBuffer(CL_READ_ONLY_CACHE, 0, sizeof(DTYPE_COLOR_VALUE) * N);
    cl::Buffer b_result = cl.AddBuffer(CL_READ_WRITE_CACHE, 1, sizeof(DTYPE_COLOR_VALUE) * N);

    cl.WriteBuffer(b_array, vec, 0, sizeof(DTYPE_COLOR_VALUE) * N);
    cl.WriteBuffer(b_result, result, 1, sizeof(DTYPE_COLOR_VALUE) * N);

    cl::NDRange global(N);

    cl.Run(cl::NullRange, global);

    cl.ReadBuffer(b_result, 2, sizeof(DTYPE_COLOR_VALUE) * N, result);

    validateSum(result[0], N);
    std::cout << std::setw(7) << "1" << std::setw(7) << N << std::setw(15) << cl.getTotalExecutionTime() << "ms"
              << std::setw(25) << (validateSum(result[0], N) ? "" : " ERROR: validation failed") << std::endl;
};

void executev2(ClWrapper cl, cl_uint N) {

    DTYPE_COLOR_VALUE *vec = (DTYPE_COLOR_VALUE *) malloc(sizeof(DTYPE_COLOR_VALUE) * N);
    DTYPE_COLOR_VALUE *result = (DTYPE_COLOR_VALUE *) malloc(sizeof(DTYPE_COLOR_VALUE) * N);

    fillArray(vec, N);
    for (size_t i = 0; i < N; i++) { result[i] = 0; }


    cl.Build("reduction_v2");

    cl::Buffer b_array = cl.AddBuffer(CL_READ_ONLY_CACHE, 0, sizeof(DTYPE_COLOR_VALUE) * N);
    cl::Buffer b_result = cl.AddBuffer(CL_READ_WRITE_CACHE, 1, sizeof(DTYPE_COLOR_VALUE) * N);

    cl.WriteBuffer(b_array, vec, 0, sizeof(DTYPE_COLOR_VALUE) * N);
    cl.WriteBuffer(b_result, result, 1, sizeof(DTYPE_COLOR_VALUE) * N);

    cl.kernel.setArg(2, sizeof(DTYPE_COLOR_VALUE) * N, NULL);

    cl::NDRange global(N);
    cl::NDRange local(N);
    cl.Run(local, global);

    cl.ReadBuffer(b_result, 2, sizeof(DTYPE_COLOR_VALUE) * N, result);
//    printArray(result, N);

    validateSum(result[0], N);
    std::cout << std::setw(7) << "2" << std::setw(7) << N << std::setw(15) << cl.getTotalExecutionTime() << "ms"
              << std::setw(25) << (validateSum(result[0], N) ? "" : "ERROR: validation failed") << std::endl;

};

void executev3(ClWrapper cl, cl_uint N, cl_uint WORKGROUP_SIZE) {

    DTYPE_COLOR_VALUE *vec = (DTYPE_COLOR_VALUE *) malloc(sizeof(DTYPE_COLOR_VALUE) * N);
    DTYPE_COLOR_VALUE *result = (DTYPE_COLOR_VALUE *) malloc(sizeof(DTYPE_COLOR_VALUE) * N);

    fillArray(vec, N);
    for (size_t i = 0; i < N; i++) { result[i] = 0; }


    cl.Build("reduction_v2");

    cl::Buffer b_array = cl.AddBuffer(CL_READ_ONLY_CACHE, 0, sizeof(DTYPE_COLOR_VALUE) * N);
    cl::Buffer b_result = cl.AddBuffer(CL_READ_WRITE_CACHE, 1, sizeof(DTYPE_COLOR_VALUE) * N);

    cl.WriteBuffer(b_array, vec, 0, sizeof(DTYPE_COLOR_VALUE) * N);
    cl.WriteBuffer(b_result, result, 1, sizeof(DTYPE_COLOR_VALUE) * N);

    cl.kernel.setArg(2, sizeof(DTYPE_COLOR_VALUE) * N, NULL);

    for (int i = N; i > 0; i /= WORKGROUP_SIZE) {
        cl::NDRange global(N);
        cl::NDRange local(WORKGROUP_SIZE);
        cl.Run(local, global);

        cl.ReadBuffer(b_result, 2, sizeof(DTYPE_COLOR_VALUE) * N, result);

        cl::Buffer b_temp = b_result;
        b_result = b_array;
        b_array = b_temp;

        cl.kernel.setArg(0, b_array);
        cl.kernel.setArg(1, b_result);
    }

    validateSum(result[0], N);
    std::cout << std::setw(7) << "3" << std::setw(7) << N << std::setw(15) << cl.getTotalExecutionTime() << "ms"
              << std::setw(25) << (validateSum(result[0], N) ? "" : "ERROR: validation failed") << std::endl;

};

void printArray(DTYPE_COLOR_VALUE *vec, size_t N) {
    size_t i = 0;
    printf("Array: { ");
    for (; i < N - 1; ++i) {
        printf("%d, ", vec[i]);
    }
    printf("%d}\n", vec[i]);
}

void fillArray(DTYPE_COLOR_VALUE *vec, size_t N) {
    for (size_t i = 1; i <= N; ++i) {
        vec[i - 1] = (DTYPE_COLOR_VALUE) i;
    }
}

bool validateSum(int sum, size_t N) {
    DTYPE_COLOR_VALUE check = (DTYPE_COLOR_VALUE) (N * (N + 1) / 2);
    if (sum == check) {
        return true;
    } else {
        std::cout << sum << " != " << check << std::endl;
        return false;
    }
}

#pragma clang diagnostic pop
#include <iostream>
#include <iomanip>

#include "ClWrapper.h"

#include "time_ms.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"
#define VALUE cl_int

void fillArray(VALUE *vec, size_t N);

void printArray(VALUE *vec, size_t N);

bool validateSum(VALUE sum, size_t N);

VALUE iterativeReduction(VALUE *vector, size_t N);

void executev1(ClWrapper cl, cl_uint N);

void executev2(ClWrapper cl, cl_uint N, cl_uint WORKGROUP_SIZE);

int main() {

    try {

        ClWrapper cl("../reduction.c", -1);

        std::cout << std::setw(7) << "version" << std::setw(7) << "N" << std::setw(15) << "Time" << std::endl;

        cl_uint N = 16384;
        cl_uint WORKGROUP_SIZE = 64;

        for (cl_uint n = 1024; n < N; n *= 2) {
            executev1(cl, n);
            executev2(cl, n, WORKGROUP_SIZE);
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

    VALUE *vec = (VALUE *) malloc(sizeof(VALUE) * N);
    VALUE *result = (VALUE *) malloc(sizeof(VALUE) * N);

    fillArray(vec, N);
    for (size_t i = 0; i < N; i++) { result[i] = 0; }


    cl.Build("reduction_v1");

    cl::Buffer b_array = cl.AddBuffer(CL_READ_ONLY_CACHE, 0, sizeof(VALUE) * N);
    cl::Buffer b_result = cl.AddBuffer(CL_READ_WRITE_CACHE, 1, sizeof(VALUE) * N);

    cl.WriteBuffer(b_array, vec, 0, sizeof(VALUE) * N);
    cl.WriteBuffer(b_result, result, 1, sizeof(VALUE) * N);

    cl::NDRange global(N);

    cl.Run(cl::NullRange, global);

    cl.ReadBuffer(b_result, 2, sizeof(VALUE) * N, result);

    validateSum(result[0], N);
    std::cout << std::setw(7) << "1" << std::setw(7) << N << std::setw(15) << cl.getTotalExecutionTime() << "ms"
              << std::setw(25)<< (validateSum(result[0], N) ? "" : " ERROR: validation failed") << std::endl;
};

void executev2(ClWrapper cl, cl_uint N, cl_uint WORKGROUP_SIZE) {

    VALUE *vec = (VALUE *) malloc(sizeof(VALUE) * N);
    VALUE *result = (VALUE *) malloc(sizeof(VALUE) * N);

    fillArray(vec, N);
    for (size_t i = 0; i < N; i++) { result[i] = 0; }


    cl.Build("reduction_v2");

    cl::Buffer b_array = cl.AddBuffer(CL_READ_ONLY_CACHE, 0, sizeof(VALUE) * N);
    cl::Buffer b_result = cl.AddBuffer(CL_READ_WRITE_CACHE, 1, sizeof(VALUE) * N);

    cl.WriteBuffer(b_array, vec, 0, sizeof(VALUE) * N);
    cl.WriteBuffer(b_result, result, 1, sizeof(VALUE) * N);

    cl.kernel.setArg(2, sizeof(VALUE) * N, NULL);

    for (int i = N; i > 0; i /= WORKGROUP_SIZE) {
        cl::NDRange global(N);
        cl::NDRange local(WORKGROUP_SIZE);
        cl.Run(local, global);

        cl.ReadBuffer(b_result, 2, sizeof(VALUE) * N, result);

        cl.kernel.setArg(0, b_result);
    }

    std::cout << std::setw(7) << "2" << std::setw(7) << N << std::setw(15) << cl.getTotalExecutionTime() << "ms"
              << std::setw(25)<< (validateSum(result[0], N) ? "" : "ERROR: validation failed") << std::endl;

};

VALUE iterativeReduction(VALUE *vector, size_t N) {

    unsigned long start_time = time_ms();
    VALUE sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += vector[i];
    }

    printf("Time for iterative reduction: %9lu ms\n", time_ms() - start_time);

    return sum;
}


void printArray(VALUE *vec, size_t N) {
    size_t i = 0;
    printf("Array: { ");
    for (; i < N - 1; ++i) {
        printf("%d, ", vec[i]);
    }
    printf("%d}\n", vec[i]);
}


void fillArray(VALUE *vec, size_t N) {
    for (size_t i = 1; i <= N; ++i) {
        vec[i - 1] = (VALUE) i;
    }
}

bool validateSum(int sum, size_t N) {
    VALUE check = (VALUE) (N * (N + 1) / 2);
    if (sum == check) {
        return true;
    } else {
        std::cout << sum << " != " << check << std::endl;
        return false;
    }
}

#pragma clang diagnostic pop
//
// Created by roland on 13.10.16.
//

#ifndef PARALLELE_CLLOADER_H
#define PARALLELE_CLLOADER_H

#include <stdlib.h>
#include <iostream>
#include <malloc.h>

#include <vector>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#define MAX_SOURCE_SIZE 1024*1024*4

struct ClException : public std::exception
{
   std::string s;
   ClException(std::string ss) : s(ss) {}
   ~ClException() throw () {}
   const char* what() const throw() { return s.c_str(); }
};

// check __err for ocl success and print message in case of error
#define CL_ERRCHECK(__err) \
if(__err != CL_SUCCESS) { \
	fprintf(stderr, "OpenCL Assertion failure in %s:%d:\n", __FILE__, __LINE__); \
	fprintf(stderr, "Error code: %s\n",  ClLoader::get_error_string(__err)); \
	throw ClException("ClException"); \
}

class ClLoader {
private:
    cl_device_id device_id_ = NULL;
    cl_context context_ = NULL;
    cl_command_queue command_queue_ = NULL;

    cl_program program_ = NULL;
    cl_platform_id * platforms_ = NULL;

    std::vector<cl_mem> buffer_;
    std::vector<size_t> buffer_size_;

    cl_uint ret_num_devices_;
    cl_uint ret_num_platforms_;
    cl_uint kernel_arg_count_ = 0;
    cl_int ret_;

    std::string kernel_path_;
    const char * kernel_source_string_;

    size_t kernel_source_size_ = 0;

    cl_kernel kernel_ = NULL;

    void LoadKernelFile();

    void GetContext();

public:

    ClLoader(const char *kernel_path, cl_uint num);
    ~ClLoader();

    void Build();

    void AddParameter(void *parameter, size_t size);

    cl_mem AddBuffer(cl_mem_flags flags, size_t buffer_size);

    void Run(const size_t * local_work_size, const size_t * global_work_size);

    void GetResult(cl_mem buffer, size_t buffer_size, cl_float * result);

    void WriteBuffer(cl_mem buffer, cl_float *array, size_t size);

		static const char * get_error_string(cl_int error);

    const char* GetDeviceDescription(const cl_device_id device);
};


#endif //PARALLELE_CLLOADER_H

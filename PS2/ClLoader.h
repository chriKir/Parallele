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

// check __err for ocl success and print message in case of error
#define CL_ERRCHECK(__err) \
if(__err != CL_SUCCESS) { \
	fprintf(stderr, "OpenCL Assertion failure in %s:%d:\n", __FILE__, __LINE__); \
	fprintf(stderr, "Error code: %s\n",  ClLoader::get_error_string(__err)); \
	throw ClException("ClException"); \
}

class ClLoader {
private:
		cl_device_id * devices_ = NULL;
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

public:

    /**
     * init OpenCL Device. If device_nr = -1 asks which device
     * @param kernel_path path to kernel file
     * @param device_nr available devices are nubered starting from 0. -1 means ask for device
     * @throws ClException if device_nr is not available
     */
    ClLoader(const char *kernel_path, int device_nr);

		/**
		 * destructor frees memory
		 */
    ~ClLoader();

		/**
		 * builds the kernel and prints compile errors
		 */
    void Build();

		/**
		 * Binds basic parameter to the OpenCL Kernel. Use only for basic datatypes as int, float, ...
		 * @param parameter cast to void*
		 * @param size size of the data in byte
		 */
    void AddParameter(void *parameter, size_t size);

		/**
		 * Binds a Buffer to the OpenCL Kernel.
		 * @param flags read/write access
		 * @param buffer_size size of the array
		 * @return returns the cl_mem reference
		 */
    cl_mem AddBuffer(cl_mem_flags flags, size_t buffer_size);

		/**
		 * Writes Data into a buffer
		 * @param buffer cl_mem reference of the buffer
		 * @param array array containing data
		 * @param size size of the array
		 */
		void WriteBuffer(cl_mem buffer, cl_float *array, size_t size);
		// TODO: should accept more than cl_float arrays

    /**
     * Runs the OpenCL Kernel.
     * @param local_work_size
     * @param global_work_size
     */
    void Run(const size_t * local_work_size, const size_t * global_work_size);

		/**
		 * Reads Data from Buffer
		 * @param buffer
		 * @param buffer_size
		 * @param result
		 */
    void GetResult(cl_mem buffer, size_t buffer_size, cl_float * result);


		// Helper functions

		/**
		 * returns the OpenCL error string
		 * @param error nr
		 * @return
		 */
		static const char * get_error_string(cl_int error);

		/**
		 * Returns description Name, Vendor and Type of the OpenCL Device
		 * @param device
		 * @return
		 */
    const char* get_device_description(const cl_device_id device);

		const char* device_type_string(cl_device_type type);

		cl_device_type get_device_type(cl_device_id device);

};

struct ClException : public std::exception
{
		std::string s;
		ClException(std::string ss) : s(ss) {}
		~ClException() throw () {}
		const char* what() const throw() { return s.c_str(); }
};

#endif //PARALLELE_CLLOADER_H

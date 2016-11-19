//
// Created by roland on 13.10.16.
//

#ifndef PARALLELE_CLLOADER_H
#define PARALLELE_CLLOADER_H

#include <stdlib.h>
#include <iostream>
#include <malloc.h>

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <map>

#define BUILD_OPTIONS "-cl-std=CL1.2" // -cl-nv-verbose -Werror "

#define CL_ERRCHECK(__err) \
if(__err != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL Assertion failure in %s:%d:\n", __FILE__, __LINE__); \
    fprintf(stderr, "Error code: %s\n",  ClLoader::get_error_string(__err)); \
    throw ClException("ClException"); \
}

class ClLoader {
private:
    std::vector<cl::Platform> platforms_;
    std::vector<cl::Device> devices_;
    std::vector<cl::Device> all_devices_;

    cl::Device device_;

    cl::Context context_;
    cl::CommandQueue command_queue_;

    cl::Program program_;

    std::map<int, cl::Event> buffer_write_events_;
    std::map<int, cl::Event> buffer_read_events_;

    double kernel_execution_time_ = 0;
    std::map<int, double> buffer_write_times_;
    std::map<int, double> buffer_read_times_;

    std::string kernel_path_;
    std::string kernel_source_string_;
    cl::Program::Sources sources_;

    cl::Kernel kernel_;
    std::vector<cl::Event> kernel_event_;

    void LoadKernelFile();

public:

    /**
     * init OpenCL Device. If device_nr = -1 asks which device
     * @param kernel_path path to kernel file
     * @param platform_nr available devices are nubered starting from 0. -1 means ask for device
     * @throws ClException if device_nr is not available
     */
    ClLoader(const char *kernel_path, int platform_nr);

    /**
     * builds the kernel and prints compile errors
     * @param kernelFunctionName name of kernel function
     */
    void Build(const char *kernelFunctionName);

    /**
     * Binds basic argument to the OpenCL Kernel. Use only for basic datatypes as int, float, ...
     * ADD BUFFERS BEFORE ARGUMENTS
     * @param parameter cast to void*
     * @param arg_index index of the argument
     * @param size size of the data in byte
     */
    void setKernelArg(void *parameter, cl_uint arg_index, size_t size);

    /**
     * Binds a Buffer to the OpenCL Kernel.
     * @param flags read/write access
     * @param arg_index index of the argument
     * @param buffer_size size of the array
     * @return returns the cl::Buffer reference
     */
    cl::Buffer AddBuffer(cl_mem_flags flags, cl_uint arg_index, size_t buffer_size);

    /**
     * Writes Data into a buffer
     * @param buffer cl::Buffer reference of the buffer
     * @param array array containing data
     * @param arg_index index of the argument
     * @param size size of the array
     */
    void WriteBuffer(cl::Buffer buffer, cl_float *array, cl_uint arg_index, size_t size);

    /**
     * Writes Data into a buffer. Waits for previous kernel execution to finish
     * @param buffer
     * @param array
     * @param arg_index
     * @param size
     */
    void ReWriteBuffer(cl::Buffer buffer, cl_float *array, cl_uint arg_index, size_t size);

    /**
     * Runs the OpenCL Kernel.
     * @param work_dim
     * @param local_work_size
     * @param global_work_size
     */
    void Run(const cl::NDRange local_work_size, const cl::NDRange global_work_size);

    /**
     * Reads Data from Buffer
     * @param buffer
     * @param buffer_size
     * @param result
     */
    void ReadBuffer(cl::Buffer buffer, cl_uint arg_index, size_t buffer_size, cl_float *result);

    void PrintProfileInfo();


    // Helper functions

    /**
     * returns the OpenCL error string
     * @param error nr
     * @return
     */
    static const char *get_error_string(cl_int error);

    /**
     * Returns description Name, Vendor and Type of the OpenCL Device
     * @param device
     * @return
     */
    std::string get_device_description(const cl::Device device);

    std::vector<cl::Event> getEvents(std::map<int, cl::Event> myMap);

    const char *device_type_string(cl_device_type type);

    /**
     * returns time between start & end of event in ms
     * @param event
     * @return
     */
    double getDuration(cl::Event event);

};

struct ClException : public std::exception {
    std::string s;

    ClException(std::string ss) : s(ss) {}

    ~ClException() throw() {}

    const char *what() const throw() { return s.c_str(); }
};

#endif //PARALLELE_CLLOADER_H

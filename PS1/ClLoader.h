//
// Created by roland on 13.10.16.
//

#ifndef PARALLELE_CLLOADER_H
#define PARALLELE_CLLOADER_H


#include <CL/cl.hpp>

class ClLoader {
private:
    cl_device_id device_id_ = NULL;
    cl_context context_ = NULL;
    cl_command_queue command_queue_ = NULL;

    cl_program program_ = NULL;
    cl_platform_id platform_id_ = NULL;

    std::vector<cl_mem> buffer_;
    std::vector<size_t> buffer_size_;

    cl_uint ret_num_devices_;
    cl_uint ret_num_platforms_;
    cl_uint kernel_arg_count_ = 0;
    cl_int ret_;

    std::string kernel_path_;
    const char *kernel_source_string_;

    //TODO: try without
    size_t kernel_source_size_ = 0;

    cl_kernel kernel_ = NULL;

    void LoadKernelFile();

    void GetContext();

public:
    ClLoader(const char *kernel_path);

    void Build();

    void AddParameter(void *parameter, size_t size);

    cl_mem AddBuffer(cl_mem_flags flags, size_t buffer_size);

    void Run(const size_t local_work_size[2], const size_t global_work_size[2]);

    void GetResult(cl_mem &buffer, size_t buffer_size, void *result);

    static void check_for_errors(cl_int error);
};


#endif //PARALLELE_CLLOADER_H

//
// Created by roland on 13.10.16.
//

#include <stdlib.h>
#include <iostream>
#include <malloc.h>
#include "ClLoader.h"

#define MAX_SOURCE_SIZE (0x100000)
#define BUILD_OPTIONS "-Werror -cl-std=CL1.2"

ClLoader::ClLoader(const char * kernel_path):
  kernel_path_(kernel_path)
{
  this->LoadKernelFile();
  this->GetContext();

}

ClLoader::~ClLoader() {

  ret_ = clFlush(command_queue_);
  ret_ = clFinish(command_queue_);
  ret_ = clReleaseKernel(kernel_);
  ret_ = clReleaseProgram(program_);


  for (cl_mem buffer : buffer_) {
    ret_ = clReleaseMemObject(buffer);
  }

  ret_ = clReleaseCommandQueue(command_queue_);
  ret_ = clReleaseContext(context_);

  free((void *) kernel_source_string_);
}

void ClLoader::LoadKernelFile() {

  FILE *fp;

  /* Load the source code containing the kernel*/
  fp = fopen(kernel_path_.c_str(), "r");

  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
  }

  kernel_source_string_ = (char *) malloc(MAX_SOURCE_SIZE);
  kernel_source_size_ = fread((void *) kernel_source_string_, 1, MAX_SOURCE_SIZE, fp);

  fclose(fp);
}

void ClLoader::GetContext() {

  // get OpenCL Platforms
  ret_ = clGetPlatformIDs(1, &platform_id_, &ret_num_platforms_);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);

  // get OpenCL Device
  ret_ = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_DEFAULT, 1, &device_id_, &ret_num_devices_);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);

  // Create OpenCL context_
  context_ = clCreateContext(NULL, 1, &device_id_, NULL, NULL, &ret_);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);

}

void ClLoader::Build() {

  // Create Command Queue
  command_queue_ = clCreateCommandQueue(context_, device_id_, 0, &ret_);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);


  // Create Kernel Program from the source
  program_ = clCreateProgramWithSource(
      context_,
      1,
      &kernel_source_string_,
      &kernel_source_size_,
      &ret_);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);

  // Build Kernel Program
  ret_ = clBuildProgram(program_, 1, &device_id_, BUILD_OPTIONS, NULL, NULL);

  // check for build errors
  if (ret_ != CL_SUCCESS) {

    cl_build_status status;
    size_t log_size;
    char * program_log;

    // check build error and build status first
    clGetProgramBuildInfo(program_, device_id_, CL_PROGRAM_BUILD_STATUS,
                          sizeof(cl_build_status), &status, NULL);

    // check build log
    clGetProgramBuildInfo(program_, device_id_,
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    program_log = (char *) calloc(log_size + 1, sizeof(char));
    clGetProgramBuildInfo(program_, device_id_,
                          CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);

    printf("Build failed; error=%d, status=%d, programLog:nn%s",
           ret_, status, program_log);

    free(program_log);
  }

  // Create OpenCL Kernel
  kernel_ = clCreateKernel(program_, "matrix", &ret_);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);
}

void ClLoader::AddParameter(void * parameter, size_t size) {

  ret_ = clSetKernelArg(kernel_,
                        kernel_arg_count_++,
                        size,
                        parameter);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);
}

cl_mem ClLoader::AddBuffer(cl_mem_flags flags, size_t buffer_size) {

  //TODO: needed??
  buffer_size_.insert(buffer_size_.end(), buffer_size);

  // Create Memory Buffers
  buffer_.push_back(clCreateBuffer(context_, flags, buffer_size, NULL, &ret_));
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);

  // Set OpenCL Kernel Parameters
  ret_ = clSetKernelArg(kernel_,
                        kernel_arg_count_++,
                        sizeof(cl_mem),
                        &buffer_.back());
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);

  return buffer_.back();
}

void ClLoader::WriteBuffer(cl_mem buffer, cl_float * array, size_t size) {
  ret_ = clEnqueueWriteBuffer(command_queue_,
                              buffer,
                              CL_TRUE,    // blocking write
                              0,          // offset
                              size,
                              array,
                              0,
                              NULL,
                              NULL
  );

  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);
}

void ClLoader::Run(const size_t * local_work_size, const size_t * global_work_size) {
  // Execute OpenCL Kernel
  cl_event event;


  ret_ = clEnqueueNDRangeKernel(command_queue_,
                                kernel_,
                                2,
                                NULL,
                                global_work_size,
                                local_work_size,
                                0,
                                NULL,
                                &event);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);

  ret_ = clWaitForEvents(1, &event);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);
}

void ClLoader::GetResult(cl_mem buffer, size_t buffer_size, cl_float * result) {

  // Copy results from the memory buffer
  ret_ = clEnqueueReadBuffer(command_queue_,
                            buffer,
                            CL_TRUE,
                            0,
                            buffer_size,
                            result,
                            0,
                            NULL,
                            NULL);
  ClLoader::check_for_errors(ret_, __LINE__, __FILE__);
}

void ClLoader::check_for_errors(cl_int error, int line, const char * file) {

  if (error == CL_SUCCESS) {
    return;
  } else {
    std::string err;

    switch(error){
      case -1: err = "CL_DEVICE_NOT_FOUND"; break;
      case -2: err = "CL_DEVICE_NOT_AVAILABLE"; break;
      case -3: err = "CL_COMPILER_NOT_AVAILABLE"; break;
      case -4: err = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
      case -5: err = "CL_OUT_OF_RESOURCES"; break;
      case -6: err = "CL_OUT_OF_HOST_MEMORY"; break;
      case -7: err = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
      case -8: err = "CL_MEM_COPY_OVERLAP"; break;
      case -9: err = "CL_IMAGE_FORMAT_MISMATCH"; break;
      case -10: err = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
      case -11: err = "CL_BUILD_PROGRAM_FAILURE"; break;
      case -12: err = "CL_MAP_FAILURE"; break;
      case -13: err = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
      case -14: err = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
      case -15: err = "CL_COMPILE_PROGRAM_FAILURE"; break;
      case -16: err = "CL_LINKER_NOT_AVAILABLE"; break;
      case -17: err = "CL_LINK_PROGRAM_FAILURE"; break;
      case -18: err = "CL_DEVICE_PARTITION_FAILED"; break;
      case -19: err = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;

      // compile-time errors break;
      case -30: err = "CL_INVALID_VALUE"; break;
      case -31: err = "CL_INVALID_DEVICE_TYPE"; break;
      case -32: err = "CL_INVALID_PLATFORM"; break;
      case -33: err = "CL_INVALID_DEVICE"; break;
      case -34: err = "CL_INVALID_CONTEXT"; break;
      case -35: err = "CL_INVALID_QUEUE_PROPERTIES"; break;
      case -36: err = "CL_INVALID_COMMAND_QUEUE"; break;
      case -37: err = "CL_INVALID_HOST_PTR"; break;
      case -38: err = "CL_INVALID_MEM_OBJECT"; break;
      case -39: err = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
      case -40: err = "CL_INVALID_IMAGE_SIZE"; break;
      case -41: err = "CL_INVALID_SAMPLER"; break;
      case -42: err = "CL_INVALID_BINARY"; break;
      case -43: err = "CL_INVALID_BUILD_OPTIONS"; break;
      case -44: err = "CL_INVALID_PROGRAM"; break;
      case -45: err = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
      case -46: err = "CL_INVALID_KERNEL_NAME"; break;
      case -47: err = "CL_INVALID_KERNEL_DEFINITION"; break;
      case -48: err = "CL_INVALID_KERNEL"; break;
      case -49: err = "CL_INVALID_ARG_INDEX"; break;
      case -50: err = "CL_INVALID_ARG_VALUE"; break;
      case -51: err = "CL_INVALID_ARG_SIZE"; break;
      case -52: err = "CL_INVALID_KERNEL_ARGS"; break;
      case -53: err = "CL_INVALID_WORK_DIMENSION"; break;
      case -54: err = "CL_INVALID_WORK_GROUP_SIZE"; break;
      case -55: err = "CL_INVALID_WORK_ITEM_SIZE"; break;
      case -56: err = "CL_INVALID_GLOBAL_OFFSET"; break;
      case -57: err = "CL_INVALID_EVENT_WAIT_LIST"; break;
      case -58: err = "CL_INVALID_EVENT"; break;
      case -59: err = "CL_INVALID_OPERATION"; break;
      case -60: err = "CL_INVALID_GL_OBJECT"; break;
      case -61: err = "CL_INVALID_BUFFER_SIZE"; break;
      case -62: err = "CL_INVALID_MIP_LEVEL"; break;
      case -63: err = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
      case -64: err = "CL_INVALID_PROPERTY"; break;
      case -65: err = "CL_INVALID_IMAGE_DESCRIPTOR"; break;
      case -66: err = "CL_INVALID_COMPILER_OPTIONS"; break;
      case -67: err = "CL_INVALID_LINKER_OPTIONS"; break;
      case -68: err = "CL_INVALID_DEVICE_PARTITION_COUNT"; break;

      // extension errors
      case -1000: err = "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR"; break;
      case -1001: err = "CL_PLATFORM_NOT_FOUND_KHR"; break;
      case -1002: err = "CL_INVALID_D3D10_DEVICE_KHR"; break;
      case -1003: err = "CL_INVALID_D3D10_RESOURCE_KHR"; break;
      case -1004: err = "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR"; break;
      case -1005: err = "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR"; break;
      default: err = "Unknown OpenCL error"; break;
    }

    std::cerr << "OpenCL Error in " << file << ":" << line << " " << err << "\n";
  }
}





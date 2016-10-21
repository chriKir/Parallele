//
// Created by roland on 13.10.16.
//

#include <assert.h>
#include <sstream>
#include "ClLoader.h"

//#define MAX_SOURCE_SIZE (0x100000)
#define BUILD_OPTIONS "-Werror -cl-std=CL1.2"

/**
 * initializes OpenCl Device
 * @param kernel_path
 * @param num device number
 * @return
 */
ClLoader::ClLoader(const char * kernel_path, cl_uint num):
  kernel_path_(kernel_path)
{
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

  free(device_id_);
  free(platforms_);
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

  // get number of Platforms
  ret_ = clGetPlatformIDs(0, NULL, &ret_num_platforms_);
  CL_ERRCHECK(ret_);

  platforms_ = (cl_platform_id *) malloc(sizeof(cl_platform_id) * ret_num_platforms_);

  // get OpenCL Platforms
  ret_ = clGetPlatformIDs(ret_num_platforms_, platforms_, NULL);
  CL_ERRCHECK(ret_);

  // list all devices
  for (cl_uint i = 0; i < ret_num_platforms_; ++i) {

    // get number of devices for this platform
    ret_ = clGetDeviceIDs(platforms_[i], CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices_);
    CL_ERRCHECK(ret_);


      //
      cl_device_id * devices = (cl_device_id *) malloc(sizeof(cl_device_id) * ret_num_devices_);

      ret_ = clGetDeviceIDs(platforms_[i], CL_DEVICE_TYPE_ALL, ret_num_devices_, devices, NULL);
      CL_ERRCHECK(ret_);

    for (cl_uint j = 0; j < ret_num_devices_; ++j) {
      std::cout << "(" << i << ")" << GetDeviceDescription(devices[j]) << std::endl;
    }
  }

  std::string input = "";
  int choise = 0;

  while (true) {
    std::cout << "Please choose device:";
    std::getline(std::cin, input);

   std::stringstream myStream(input);
   if (myStream >> choise)                  // TODO: check if valid
     break;

   std::cout << "Invalid number, please try again" << std::endl;
  }

  // Create OpenCL context
  context_ = clCreateContext(NULL, 1, &device_id_, NULL, NULL, &ret_);
  CL_ERRCHECK(ret_);

  // Create Command Queue
  command_queue_ = clCreateCommandQueue(context_, device_id_, 0, &ret_);
  CL_ERRCHECK(ret_);

}

const char* device_type_string(cl_device_type type) {
  switch(type){
    case CL_DEVICE_TYPE_CPU: return "CPU";
    case CL_DEVICE_TYPE_GPU: return "GPU";
    case CL_DEVICE_TYPE_ACCELERATOR: return "ACC";
    default: return "UNKNOWN";
  }

}

cl_device_type get_device_type(cl_device_id device) {
  cl_device_type retval;
  CL_ERRCHECK(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(retval), &retval, NULL));
  return retval;
}


#define MAX_DEVICES 16
const char* ClLoader::GetDeviceDescription(const cl_device_id device) {
  static char description[128];

    char name[255], vendor[255];
    CL_ERRCHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, 255, name, NULL));
    CL_ERRCHECK(clGetDeviceInfo(device, CL_DEVICE_VENDOR, 255, vendor, NULL));
    sprintf(description, "%s  |  %s  |  %s", name, vendor, device_type_string(get_device_type(device)));

  return description;
}


void ClLoader::Build() {

  this->LoadKernelFile();

  // Create Kernel Program from the source
  program_ = clCreateProgramWithSource(
      context_,
      1,
      &kernel_source_string_,
      &kernel_source_size_,
      &ret_);
  CL_ERRCHECK(ret_);

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
  CL_ERRCHECK(ret_);
}

void ClLoader::AddParameter(void * parameter, size_t size) {

  ret_ = clSetKernelArg(kernel_,
                        kernel_arg_count_++,
                        size,
                        parameter);
  CL_ERRCHECK(ret_);
}

cl_mem ClLoader::AddBuffer(cl_mem_flags flags, size_t buffer_size) {

  buffer_size_.insert(buffer_size_.end(), buffer_size);

  // Create Memory Buffers
  buffer_.push_back(clCreateBuffer(context_, flags, buffer_size, NULL, &ret_));
  CL_ERRCHECK(ret_);

  // Set OpenCL Kernel Parameters
  ret_ = clSetKernelArg(kernel_,
                        kernel_arg_count_++,
                        sizeof(cl_mem),
                        &buffer_.back());
  CL_ERRCHECK(ret_);

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

  CL_ERRCHECK(ret_);
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
  CL_ERRCHECK(ret_);

  ret_ = clWaitForEvents(1, &event);
  CL_ERRCHECK(ret_);
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
  CL_ERRCHECK(ret_);
}

const char * ClLoader::get_error_string(cl_int error) {

  switch (error) {
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

      // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGe";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

      // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";

  }
}




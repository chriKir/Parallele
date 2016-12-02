//
// Created by roland on 13.10.16.
//

#include <sstream>
#include <cstring>
#include <fstream>
#include "ClWrapper.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"

ClWrapper::ClWrapper(const char *kernel_path, int platform_nr) :
        kernel_path_(kernel_path) {

    //get all platforms (drivers)
    cl::Platform::get(&platforms_);

    if (platforms_.size() == 0) {
        throw ClException("No platforms found. Check OpenCL installation!");
    }

    // list all devices
    int n = 0;
    for (int i = 0; i < (int) platforms_.size(); ++i) {

        std::vector<cl::Device> temp_devices;

        platforms_[i].getDevices(CL_DEVICE_TYPE_ALL, &temp_devices);

        for (size_t j = 0; j < temp_devices.size(); ++j, ++n) {
            std::cout << "(" << n << ")" << get_device_description(temp_devices[j]) << std::endl;
        }

        all_devices_.insert(all_devices_.end(), temp_devices.begin(), temp_devices.end());
    }

    if (platform_nr == -1) {

        size_t choice;

        std::string input = "";

        while (true) {
            std::cout << "Please choose device: ";
            std::getline(std::cin, input);

            std::stringstream myStream(input);
            if (myStream >> choice && choice < all_devices_.size())
                break;

            std::cout << "Invalid device, please try again" << std::endl;
        }

        device = all_devices_[choice];
        devices_.push_back(all_devices_[choice]);

    } else {
        if (platform_nr <= (int) all_devices_.size()) {
            device = all_devices_[platform_nr];
            devices_.push_back(all_devices_[platform_nr]);
        } else {
            char error[255];
            sprintf(error, "Device %d is not available.\n", platform_nr);
            throw ClException(error);
        }
    }

    if (devices_.size() == 0) {
        throw ClException("No devices found. Check OpenCL installation!");
    }

    std::cout << "using: " << get_device_description(device) << std::endl << std::endl;

    context_ = cl::Context(device);

}

void ClWrapper::LoadKernelFile() {

    // Load the source code containing the kernel
    std::ifstream in(kernel_path_);

    if (in) {
        kernel_source_string_ = std::string((std::istreambuf_iterator<char>(in)),
                                            std::istreambuf_iterator<char>());

        sources_ = cl::Program::Sources(1,
                                        std::make_pair(kernel_source_string_.c_str(),
                                                       kernel_source_string_.length() + 1));
    } else {
        throw ClException("File not found: " + kernel_path_);
    }
}

void ClWrapper::Build(std::string kernelFunctionName) {

    // Create Command Queue
#ifdef PROFILING
    command_queue_ = cl::CommandQueue(context_, device, CL_QUEUE_PROFILING_ENABLE);
#else
    command_queue_ = cl::CommandQueue(context_, device);
#endif

    this->LoadKernelFile();


    // Create Kernel Program from the source
    program_ = cl::Program(context_, sources_);

    // Build Kernel Program
    try {
        program_.build(devices_, BUILD_OPTIONS);
    }
    catch (cl::Error &e) {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            for (cl::Device dev : devices_) {
                // Check the build status
                cl_build_status status = program_.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR)
                    continue;

                // Get the build log
                std::string name = dev.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                          << buildlog << std::endl;
            }
        } else {
            throw e;
        }
    }

    // Create OpenCL Kernel
    kernel = cl::Kernel(program_, kernelFunctionName.c_str());
}

void ClWrapper::setKernelArg(void *parameter, cl_uint arg_index, size_t size) {

    kernel.setArg(arg_index, size, parameter);

}

cl::Buffer ClWrapper::AddBuffer(cl_mem_flags flags, cl_uint arg_index, size_t buffer_size) {

    // Create Memory Buffers
    cl::Buffer buffer(context_, flags, buffer_size);

    // Set OpenCL Kernel Parameters
    kernel.setArg(arg_index, buffer);

    buffer_read_times_[arg_index] = 0;
    buffer_write_times_[arg_index] = 0;
    buffer_sizes_[arg_index] = buffer_size;

    return buffer;
}

void ClWrapper::Run(const cl::NDRange local_work_size, const cl::NDRange global_work_size) {

    std::vector<cl::Event> write_events = getEvents(buffer_write_events_);

    kernel_event_.push_back(cl::Event());
    int err = command_queue_.enqueueNDRangeKernel(kernel,
                                                  cl::NullRange,
                                                  global_work_size,
                                                  local_work_size,
                                                  &write_events,
                                                  &kernel_event_.back());
    CL_ERRCHECK(err)

#ifdef PROFILING
    kernel_event_.back().wait();
    command_queue_.finish();
    kernel_execution_time_ += getDuration(kernel_event_.back());
#endif

}

double ClWrapper::getTotalExecutionTime() {

    double total = kernel_execution_time_;

    for (cl_uint j = 0; j < buffer_write_times_.size(); j++) {
        total += buffer_write_times_[j];
        total += buffer_read_times_[j];
    }
    return total;
}


#define WIDTH 10

void ClWrapper::printProfilingInfo() {

    // print headline
    std::cout << std::setw(WIDTH) << "kernel";
    for (cl_uint j = 0; j < buffer_write_times_.size(); j++) {
        std::cout << std::setw(WIDTH) << "write " << j << ";"
                  << std::setw(WIDTH) << "read " << j << ";";
    }

    std::cout << std::endl;

    // print data
    std::cout << kernel_execution_time_ << "; ";
    for (cl_uint j = 0; j < buffer_write_times_.size(); j++) {
        std::cout << std::setw(WIDTH) << buffer_write_times_[j] << "; "
                  << std::setw(WIDTH) << buffer_read_times_[j] << "; ";
    }

    std::cout << std::endl;

}

double ClWrapper::getDuration(cl::Event event) {

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    return (end_time - start_time) * 1e-6;

}

std::string ClWrapper::get_device_description(const cl::Device device) {
    std::string name, vendor, type, version;

    name = device.getInfo<CL_DEVICE_NAME>();
    vendor = device.getInfo<CL_DEVICE_VENDOR>();

    type = device_type_string(device.getInfo<CL_DEVICE_TYPE>());
    version = device.getInfo<CL_DEVICE_VERSION>();

    return name + " | " + vendor + " | " + type + " | " + version;
}

std::vector<cl::Event> ClWrapper::getEvents(std::map<int, cl::Event> myMap) {

    std::vector<cl::Event> retVal;
    for (std::map<int, cl::Event>::iterator it = myMap.begin();
         it != myMap.end(); ++it) {
        retVal.push_back(it->second);
    }
    return retVal;
};

const char *ClWrapper::device_type_string(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_CPU:
            return "CPU";
        case CL_DEVICE_TYPE_GPU:
            return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR:
            return "ACC";
        default:
            return "UNKNOWN";
    }
}

const char *ClWrapper::get_error_string(cl_int error) {
    switch (error) {
        case -1:
            return "CL_DEVICE_NOT_FOUND";
        case -2:
            return "CL_DEVICE_NOT_AVAILABLE";
        case -3:
            return "CL_COMPILER_NOT_AVAILABLE";
        case -4:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5:
            return "CL_OUT_OF_RESOURCES";
        case -6:
            return "CL_OUT_OF_HOST_MEMORY";
        case -7:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8:
            return "CL_MEM_COPY_OVERLAP";
        case -9:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case -10:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11:
            return "CL_BUILD_PROGRAM_FAILURE";
        case -12:
            return "CL_MAP_FAILURE";
        case -13:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case -16:
            return "CL_LINKER_NOT_AVAILABLE";
        case -17:
            return "CL_LINK_PROGRAM_FAILURE";
        case -18:
            return "CL_DEVICE_PARTITION_FAILED";
        case -19:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
        case -30:
            return "CL_INVALID_VALUE";
        case -31:
            return "CL_INVALID_DEVICE_TYPE";
        case -32:
            return "CL_INVALID_PLATFORM";
        case -33:
            return "CL_INVALID_DEVICE";
        case -34:
            return "CL_INVALID_CONTEXT";
        case -35:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case -36:
            return "CL_INVALID_COMMAND_QUEUE";
        case -37:
            return "CL_INVALID_HOST_PTR";
        case -38:
            return "CL_INVALID_MEM_OBJECT";
        case -39:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40:
            return "CL_INVALID_IMAGE_SIZE";
        case -41:
            return "CL_INVALID_SAMPLER";
        case -42:
            return "CL_INVALID_BINARY";
        case -43:
            return "CL_INVALID_BUILD_OPTIONS";
        case -44:
            return "CL_INVALID_PROGRAM";
        case -45:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46:
            return "CL_INVALID_KERNEL_NAME";
        case -47:
            return "CL_INVALID_KERNEL_DEFINITION";
        case -48:
            return "CL_INVALID_KERNEL";
        case -49:
            return "CL_INVALID_ARG_INDEX";
        case -50:
            return "CL_INVALID_ARG_VALUE";
        case -51:
            return "CL_INVALID_ARG_SIZE";
        case -52:
            return "CL_INVALID_KERNEL_ARGS";
        case -53:
            return "CL_INVALID_WORK_DIMENSION";
        case -54:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case -55:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case -56:
            return "CL_INVALID_GLOBAL_OFFSET";
        case -57:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case -58:
            return "CL_INVALID_EVENT";
        case -59:
            return "CL_INVALID_OPERATION";
        case -60:
            return "CL_INVALID_GL_OBJECT";
        case -61:
            return "CL_INVALID_BUFFER_SIZE";
        case -62:
            return "CL_INVALID_MIP_LEVEL";
        case -63:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64:
            return "CL_INVALID_PROPERTY";
        case -65:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66:
            return "CL_INVALID_COMPILER_OPTIONS";
        case -67:
            return "CL_INVALID_LINKER_OPTIONS";
        case -68:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
        case -1000:
            return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001:
            return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002:
            return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003:
            return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004:
            return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005:
            return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default:
            return "Unknown OpenCL error";

    }
}


#pragma clang diagnostic pop
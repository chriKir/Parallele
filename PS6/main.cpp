//
// Created by roland on 11/23/16.
//
#include <iostream>
#include <iomanip>

#include "ClWrapper.h"
#include "stb_image.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"

#define DTYPE_COLOR_VALUE cl_uchar
#define DTYPE_SUM_VALUE cl_ulong
#define WORKGROUP_SIZE 128

const std::string filename = "test_min.png";

void printArray(DTYPE_COLOR_VALUE *vec, size_t N);

int main() {

    try {
        int width, height, components;
        DTYPE_COLOR_VALUE *image = stbi_load(filename.c_str(), &width, &height, &components, 0);

        if (!image) throw ClException("File not found: " + filename);

        ClWrapper cl("auto_levels.c", 0);
        cl.Build("mmav_reduction");

//        int WORKGROUP_SIZE = cl.device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() / 2;

        // round buffer size to next multiple of WORKGROUP_SIZE
        size_t pixel_count = (size_t) (height * width);
        if (pixel_count % WORKGROUP_SIZE != 0)
            pixel_count = (pixel_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;

        // size of wg_buffer
        size_t buffer_size = pixel_count * sizeof(DTYPE_COLOR_VALUE) * components;
        size_t wg_buffer_size = buffer_size / WORKGROUP_SIZE;
        size_t wg_sum_buffer_size = (sizeof(DTYPE_SUM_VALUE) - sizeof(DTYPE_COLOR_VALUE)) * wg_buffer_size + wg_buffer_size;

        std::cout << filename << ": " << width << "x" << height << " " << pixel_count << " Pixels, " << components
                  << " Components" << std::endl;
        std::cout << "Buffer size: " << buffer_size << ", Workgroup Size: " << WORKGROUP_SIZE << ", Wg Buffer sizes: "
                  << wg_buffer_size << ":" << wg_sum_buffer_size << std::endl;

        DTYPE_COLOR_VALUE *wg_min = (DTYPE_COLOR_VALUE *) malloc(wg_buffer_size);
        DTYPE_COLOR_VALUE *wg_max = (DTYPE_COLOR_VALUE *) malloc(wg_buffer_size);
        DTYPE_SUM_VALUE *wg_sum = (DTYPE_SUM_VALUE *) malloc(wg_sum_buffer_size);

        for (size_t i = 0; i < wg_buffer_size; i++) {
            wg_min[i] = 255;
            wg_max[i] = 0;
            wg_sum[i] = 0;
        }

        cl::Buffer b_input = cl.AddBuffer(CL_READ_ONLY_CACHE, 0, buffer_size);
        cl::Buffer b_wg_min = cl.AddBuffer(CL_READ_WRITE_CACHE, 1, wg_buffer_size);
        cl::Buffer b_wg_max = cl.AddBuffer(CL_READ_WRITE_CACHE, 2, wg_buffer_size);
        cl::Buffer b_wg_sum = cl.AddBuffer(CL_READ_WRITE_CACHE, 3, wg_sum_buffer_size);

        cl.kernel.setArg(4, sizeof(DTYPE_COLOR_VALUE) * WORKGROUP_SIZE, NULL);
        cl.kernel.setArg(5, sizeof(DTYPE_COLOR_VALUE) * WORKGROUP_SIZE, NULL);
        cl.kernel.setArg(6, sizeof(DTYPE_SUM_VALUE) * WORKGROUP_SIZE, NULL);

        cl_uint size = (cl_uint) (height * width);
        cl.kernel.setArg(7, sizeof(cl_uint), &size);

        cl.WriteBuffer(b_input, image, 0);
        cl.WriteBuffer(b_wg_min, wg_min, 1);
        cl.WriteBuffer(b_wg_max, wg_max, 2);
        cl.WriteBuffer(b_wg_sum, wg_sum, 3);

        for (size_t i = pixel_count; i > 1; i /= WORKGROUP_SIZE) {
            std::cout << "##" << i << std::endl;

            cl::NDRange global(i);
            cl::NDRange local(i < WORKGROUP_SIZE ? cl::NullRange : WORKGROUP_SIZE);

            cl.Run(local, global);

        }

        cl.ReadBuffer(b_wg_min, 1, wg_min);
        cl.ReadBuffer(b_wg_max, 2, wg_max);
        cl.ReadBuffer(b_wg_sum, 3, wg_sum);

        for (int c = 0; c < components; c++) {

            std::cout << "component " << c << ": " << +wg_min[c] << " / " << wg_sum[c] / (height * width) << " / " << +wg_max[0]
                      << std::endl;
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

void printArray(DTYPE_COLOR_VALUE *vec, size_t N) {
    size_t i = 0;
    printf("Array: { ");
    for (; i < N - 1; ++i) {
        printf("%c, ", vec[i]);
    }
    printf("%d}\n", vec[i]);
}

#pragma clang diagnostic pop

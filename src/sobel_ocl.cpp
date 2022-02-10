/* sobel_ocl.cpp - Sobel filter with OpenCL */
#include <float.h>
#include <assert.h>
#include <omp.h>
#include <math.h>

#include <vector>
#include <iostream>
#include <string>
#include <cstring>

#include "imgutils.h"
#include "oclutils.h"

cl_context clContext;
cl_device_id clDeviceId;
cl_mem clImageDataIn;
cl_mem clImageDataOut;
cl_mem clSobelMiddleResult;
cl_mem clMinValues;
cl_mem clMaxValues;
cl_command_queue clCommandQueue;
cl_program clProgram;
cl_kernel clSobelKernel;
cl_kernel clMinMaxKernel;
cl_kernel clMinMaxIterKernel;
cl_kernel clScalePixelsKernel;

unsigned int gImageBytes;
unsigned int gMinMaxKernelLocalSize;
unsigned int gMinMaxThreadNumPerBlock;
unsigned int gMinMaxBlockNum;

bool init_ocl_ctx(int chNum)
{
    LOG_INFO("Initialize OpenCL environment start ...");

    cl_platform_id platformID;
    cl_int retCode;
    char stringBuffer[1024];

    bool isNV = false;
    if (get_platform_id(platformID, isNV) != CL_SUCCESS) return false;

    clGetPlatformInfo(platformID, CL_PLATFORM_EXTENSIONS, sizeof(stringBuffer), &stringBuffer, NULL);
    LOG_INFO("Platform extensions: " << stringBuffer);

    cl_context_properties prop[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platformID),
        0
    };
    clContext = clCreateContextFromType(prop, CL_DEVICE_TYPE_GPU, NULL, NULL, &retCode);

    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Cannot create OpenCL context! Error code = " << retCode);
        return false;
    }

    clDeviceId = get_max_flops_device(clContext);
    show_opencl_device_info(clDeviceId);
    clGetDeviceInfo(clDeviceId, CL_DEVICE_EXTENSIONS, sizeof(stringBuffer), &stringBuffer, NULL);

    clCommandQueue = clCreateCommandQueue(clContext, clDeviceId, 0, &retCode);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Create command queue failed! Error code = " << retCode);
        return false;
    }

    cl_program program;
    if (load_cl_source_from_file(clContext, "sobel.cl", program)) {
        clProgram = program;
    } else {
        return false;
    }

    std::string buildOpts = "-cl-fast-relaxed-math";
    retCode = clBuildProgram(clProgram, 1, &(clDeviceId), buildOpts.c_str(), NULL, NULL);
    if (retCode != CL_SUCCESS) {
        char buildLog[10240];
        clGetProgramBuildInfo(clProgram, clDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        LOG_ERROR("Build OpenCL program failed: " << buildLog);
        return false;
    }
    LOG_INFO("OpenCL program build successful!");

    if (chNum == 4) {
        clSobelKernel = clCreateKernel(clProgram, "k_sobel4", &retCode);
    } else if (chNum == 3) {
        clSobelKernel = clCreateKernel(clProgram, "k_sobel3", &retCode);
    } else if (chNum == 1) {
        clSobelKernel = clCreateKernel(clProgram, "k_sobel1", &retCode);
    } else {
        return false;
    }
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Cannot create sobel kernel! Error code = " << retCode);
        return false;
    }

    clMinMaxKernel = clCreateKernel(clProgram, "k_min_max", &retCode);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Cannot create minmax kernel! Error code = " << retCode);
        return false;
    }

    clMinMaxIterKernel = clCreateKernel(clProgram, "k_min_max_iter", &retCode);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Cannot create minmax iteration kernel! Error code = " << retCode);
        return false;
    }

    if (chNum == 4) {
        clScalePixelsKernel = clCreateKernel(clProgram, "k_scale_pixels4", &retCode);
    } else if (chNum == 3) {
        clScalePixelsKernel = clCreateKernel(clProgram, "k_scale_pixels3", &retCode);
    } else if (chNum == 1) {
        clScalePixelsKernel = clCreateKernel(clProgram, "k_scale_pixels1", &retCode);
    } else {
        return false;
    }
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Cannot create scale pixels kernel! Error code = " << retCode);
        return false;
    }

    size_t localSize;
    retCode = clGetKernelWorkGroupInfo(clMinMaxKernel, clDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &localSize, NULL);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Get kernel work group size of minmax kernel failed! Error code = " << retCode);
        return false;
    } else {
        LOG_INFO("Minmax kernel work group size: " << localSize);
    }

    gMinMaxKernelLocalSize = (unsigned int)localSize;
    if (!is_pow2(gMinMaxKernelLocalSize)) {
        gMinMaxKernelLocalSize = next_pow2((gMinMaxKernelLocalSize + 2) >> 1);
        LOG_INFO("Reduce minmax kernel work group size to (power of 2): " << gMinMaxKernelLocalSize);
    }

    LOG_INFO("OpenCL context successfully initialized!");
    return true;
}

bool create_ocl_mems(int imgWidth, int imgHeight, int pitch, int chNum)
{
    cl_int retCode;
    unsigned int imgBytes = gImageBytes;
    clImageDataIn = clCreateBuffer(clContext,
        CL_MEM_READ_ONLY, imgBytes, 0, &retCode);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Create buffer for input image failed! Error code = " << retCode);
        return false;
    }

    clImageDataOut = clCreateBuffer(clContext,
        CL_MEM_WRITE_ONLY, imgBytes, 0, &retCode);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Create buffer for output image failed! Error code = " << retCode);
        return false;
    }

    unsigned int pixelNum = imgWidth*imgHeight;
    clSobelMiddleResult = clCreateBuffer(clContext,
        CL_MEM_WRITE_ONLY, pixelNum * sizeof(float), 0, &retCode);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Create buffer for sobel middle result failed! Error code = " << retCode);
        return false;
    }

    gMinMaxThreadNumPerBlock = (pixelNum < gMinMaxKernelLocalSize) ? next_pow2(pixelNum) : gMinMaxKernelLocalSize;
    gMinMaxBlockNum = (pixelNum + gMinMaxThreadNumPerBlock - 1) / gMinMaxThreadNumPerBlock;
    if (gMinMaxBlockNum == 0) {
        LOG_ERROR("Block number of minmax kernel is 0!");
        return false;
    }
    LOG_INFO("Minmax kernel thread number per block: " << gMinMaxThreadNumPerBlock);
    LOG_INFO("Minmax kernel block number: " << gMinMaxBlockNum);

    clMinValues = clCreateBuffer(clContext,
        CL_MEM_WRITE_ONLY, gMinMaxBlockNum * sizeof(float), 0, &retCode);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Create buffer for min results failed! Error code = " << retCode);
        return false;
    }

    clMaxValues = clCreateBuffer(clContext,
        CL_MEM_WRITE_ONLY, gMinMaxBlockNum * sizeof(float), 0, &retCode);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Create buffer for max results failed! Error code = " << retCode);
        return false;
    }

    return true;
}

void clear_ocl_ctx()
{
    if (clSobelKernel) { clReleaseKernel(clSobelKernel); clSobelKernel = 0; }
    if (clMinMaxKernel) { clReleaseKernel(clMinMaxKernel); clMinMaxKernel = 0; }
    if (clMinMaxIterKernel) { clReleaseKernel(clMinMaxIterKernel); clMinMaxIterKernel = 0; }
    if (clScalePixelsKernel) { clReleaseKernel(clScalePixelsKernel); clScalePixelsKernel = 0; }
    if (clProgram) { clReleaseProgram(clProgram); clProgram = 0; }
    if (clImageDataIn) { clReleaseMemObject(clImageDataIn); clImageDataIn = 0; }
    if (clImageDataOut) { clReleaseMemObject(clImageDataOut); clImageDataOut = 0; }
    if (clSobelMiddleResult) { clReleaseMemObject(clSobelMiddleResult); clSobelMiddleResult = 0; }
    if (clMinValues) { clReleaseMemObject(clMinValues); clMinValues = 0; }
    if (clMaxValues) { clReleaseMemObject(clMaxValues); clMaxValues = 0; }
    if (clCommandQueue) { clReleaseCommandQueue(clCommandQueue); clCommandQueue = 0; }
    if (clContext) { clReleaseContext(clContext); clContext = 0; }
}

void sobel_filtering(FIBITMAP *imgIn, FIBITMAP *imgOut)
{
    int width = FreeImage_GetWidth(imgIn);
    assert(width == FreeImage_GetWidth(imgOut));

    int height = FreeImage_GetHeight(imgIn);
    assert(height == FreeImage_GetHeight(imgOut));
    assert(FreeImage_GetBPP(imgIn) == FreeImage_GetBPP(imgOut));

    int widthBytes = FreeImage_GetLine(imgIn);
    int pitch = FreeImage_GetPitch(imgIn);
    int chNum = 0;
    switch (FreeImage_GetBPP(imgIn))
    {
    case 8: chNum = 1; break;
    case 24: chNum = 3; break;
    case 32: chNum = 4; break;
    default:
        std::cout << "Unsupported image format!" << std::endl;
        break;
    }

    if (chNum == 0) return;

    gImageBytes = height * pitch;

    if (!init_ocl_ctx(chNum)) return;
    if (!create_ocl_mems(width, height, pitch, chNum)) return;

    LOG_INFO("Filtering of input image start ...");
    double start_time = omp_get_wtime();

    // load source image data from host to device
    BYTE *dataIn = FreeImage_GetBits(imgIn);
    cl_int retCode;
    retCode = clEnqueueWriteBuffer(clCommandQueue, clImageDataIn,
        CL_TRUE, 0, gImageBytes, dataIn, 0, NULL, NULL);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Write source image to OpenCL memory object failed! Error code = " << retCode);
        return;
    }

    // Pass 1: do sobel filtering
    retCode = clSetKernelArg(clSobelKernel, 0, sizeof(cl_mem), &(clSobelMiddleResult));
    retCode |= clSetKernelArg(clSobelKernel, 1, sizeof(cl_mem), &(clImageDataIn));
    retCode |= clSetKernelArg(clSobelKernel, 2, sizeof(int), &(width));
    retCode |= clSetKernelArg(clSobelKernel, 3, sizeof(int), &(height));
    if (chNum != 4) retCode |= clSetKernelArg(clSobelKernel, 4, sizeof(int), &(pitch));
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Set sobel kernel arguments failed! Error code = " << retCode);
        return;
    }

    size_t globalDim[2] = { width, height };
    retCode = clEnqueueNDRangeKernel(clCommandQueue, clSobelKernel,
        2, NULL, globalDim, NULL, 0, NULL, NULL);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Enqueue sobel kernel failed! Error code = " << retCode);
        return;
    }

    // Pass 2: get min & max pixel value
    unsigned int pixelNum = width * height;
    retCode = clSetKernelArg(clMinMaxKernel, 0, sizeof(cl_mem), &(clMinValues));
    retCode |= clSetKernelArg(clMinMaxKernel, 1, sizeof(cl_mem), &(clMaxValues));
    retCode |= clSetKernelArg(clMinMaxKernel, 2, sizeof(cl_mem), &(clSobelMiddleResult));
    retCode |= clSetKernelArg(clMinMaxKernel, 3, sizeof(unsigned int), &(pixelNum));
    retCode |= clSetKernelArg(clMinMaxKernel, 4, gMinMaxThreadNumPerBlock * sizeof(float), NULL);
    retCode |= clSetKernelArg(clMinMaxKernel, 5, gMinMaxThreadNumPerBlock * sizeof(float), NULL);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Set minmax kernel arguments failed! Error code = " << retCode);
        return;
    }

    globalDim[0] = gMinMaxBlockNum * gMinMaxThreadNumPerBlock;
    size_t localDim[1] = { gMinMaxThreadNumPerBlock };
    retCode = clEnqueueNDRangeKernel(clCommandQueue, clMinMaxKernel,
        1, NULL, globalDim, localDim, 0, NULL, NULL);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Enqueue minmax kernel failed! Error code = " << retCode);
        return;
    }

    unsigned int minMaxIterBlockNum = gMinMaxBlockNum;
    while (minMaxIterBlockNum > 1) {
        retCode = clSetKernelArg(clMinMaxIterKernel, 0, sizeof(cl_mem), &(clMinValues));
        retCode |= clSetKernelArg(clMinMaxIterKernel, 1, sizeof(cl_mem), &(clMaxValues));
        retCode |= clSetKernelArg(clMinMaxIterKernel, 2, sizeof(unsigned int), &(minMaxIterBlockNum));
        retCode |= clSetKernelArg(clMinMaxIterKernel, 3, gMinMaxThreadNumPerBlock * sizeof(float), NULL);
        retCode |= clSetKernelArg(clMinMaxIterKernel, 4, gMinMaxThreadNumPerBlock * sizeof(float), NULL);
        if (retCode != CL_SUCCESS) {
            LOG_ERROR("Set minmaxiter kernel arguments failed! Error code = " << retCode);
            return;
        }
        minMaxIterBlockNum = (minMaxIterBlockNum + gMinMaxThreadNumPerBlock - 1) / gMinMaxThreadNumPerBlock;
        globalDim[0] = minMaxIterBlockNum * gMinMaxThreadNumPerBlock;
        retCode = clEnqueueNDRangeKernel(clCommandQueue, clMinMaxIterKernel,
            1, NULL, globalDim, localDim, 0, NULL, NULL);
        if (retCode != CL_SUCCESS) {
            LOG_ERROR("Enqueue minmaxiter kernel failed! Error code = " << retCode);
            return;
        }
    }

    // copy min/max values from device to host
    float minVal, maxVal;
    //std::vector<float> minValues(gMinMaxBlockNum, 0);
    //std::vector<float> maxValues(gMinMaxBlockNum, 0);
    retCode = clEnqueueReadBuffer(clCommandQueue, clMinValues, CL_TRUE, 0, 1 * sizeof(float), &minVal, 0, NULL, NULL);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Enqueue read min values failed! Error code = " << retCode);
        return;
    }
    retCode = clEnqueueReadBuffer(clCommandQueue, clMaxValues, CL_TRUE, 0, 1 * sizeof(float), &maxVal, 0, NULL, NULL);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Enqueue read min values failed! Error code = " << retCode);
        return;
    }

    //minVal = minValues[0];
    //maxVal = maxValues[0];
    //for (size_t i = 1; i < gMinMaxBlockNum; ++i) {
    //    if (minValues[i] < minVal) minVal = minValues[i];
    //    if (maxValues[i] > maxVal) maxVal = maxValues[i];
    //}

    LOG_INFO("the minimum value: " << minVal);
    LOG_INFO("the maximum value: " << maxVal);

    // Pass 3: scale pixel to [0, 255]
    retCode = clSetKernelArg(clScalePixelsKernel, 0, sizeof(cl_mem), &(clImageDataOut));
    retCode |= clSetKernelArg(clScalePixelsKernel, 1, sizeof(cl_mem), &(clSobelMiddleResult));
    retCode |= clSetKernelArg(clScalePixelsKernel, 2, sizeof(float), &(minVal));
    retCode |= clSetKernelArg(clScalePixelsKernel, 3, sizeof(float), &(maxVal));
    if (chNum == 4) {
        retCode |= clSetKernelArg(clScalePixelsKernel, 4, sizeof(unsigned int), &(pixelNum));
    } else {
        retCode |= clSetKernelArg(clScalePixelsKernel, 4, sizeof(int), &(width));
        retCode |= clSetKernelArg(clScalePixelsKernel, 5, sizeof(int), &(height));
        retCode |= clSetKernelArg(clScalePixelsKernel, 6, sizeof(int), &(pitch));
    }
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Set scale pixels kernel arguments failed! Error code = " << retCode);
        return;
    }

    if (chNum == 4) {
        globalDim[0] = pixelNum;
        retCode = clEnqueueNDRangeKernel(clCommandQueue, clScalePixelsKernel,
            1, NULL, globalDim, NULL, 0, NULL, NULL);
    } else {
        globalDim[0] = width;
        globalDim[1] = height;
        retCode = clEnqueueNDRangeKernel(clCommandQueue, clScalePixelsKernel,
            2, NULL, globalDim, NULL, 0, NULL, NULL);
    }
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Enqueue scale pixels kernel failed! Error code = " << retCode);
        return;
    }

    // store result
    BYTE *dataOut = FreeImage_GetBits(imgOut);
    retCode = clEnqueueReadBuffer(clCommandQueue, clImageDataOut,
        CL_TRUE, 0, gImageBytes, dataOut, 0, NULL, NULL);
    if (retCode != CL_SUCCESS) {
        LOG_ERROR("Read result image from OpenCL memory object failed! Error code = " << retCode);
        return;
    }

    clFinish(clCommandQueue);

    LOG_INFO("The total time for execution is " << omp_get_wtime() - start_time << "s");
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: sobel_ocl input_image_file" << std::endl;
        return 0;
    }

    FIBITMAP *imgIn = load_image(argv[1], 0);
    if (!imgIn) {
        return -1;
    }

    FIBITMAP *imgOut = FreeImage_Clone(imgIn);

    clContext = 0;
    clDeviceId = 0;
    clImageDataIn = 0;
    clImageDataOut = 0;
    clSobelMiddleResult = 0;
    clMinValues = 0;
    clMaxValues = 0;
    clCommandQueue = 0;
    clProgram = 0;
    clSobelKernel = 0;
    clMinMaxKernel = 0;
    clScalePixelsKernel = 0;

    gImageBytes = 0;
    gMinMaxKernelLocalSize = 0;
    gMinMaxBlockNum = 0;
    gMinMaxThreadNumPerBlock = 0;

    sobel_filtering(imgIn, imgOut);

    clear_ocl_ctx();

    std::string fnIn(argv[1]);
    std::size_t found = fnIn.find_last_of('.');
    std::string fnOut = fnIn.substr(0, found) + "_ocl_out" + fnIn.substr(found);

    int r = 0;
    if (!save_image(fnOut.c_str(), imgOut, 0)) {
        LOG_ERROR("Failed to save output image file " << fnOut);
        r = -2;
    }

    FreeImage_Unload(imgIn);
    FreeImage_Unload(imgOut);
    return r;
}

/* sobel_cuda.cu - Sobel filter with CUDA */
#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <assert.h>
#include <cuda_runtime.h>

#include "imgutils.h"
#include "sobel.h"

int show_cuda_device_info(int devIdx)
{
    cudaDeviceProp devProp;
    if (cudaGetDeviceProperties(&devProp, devIdx) != cudaError::cudaSuccess) {
        std::cout << "ERROR: Failed to retrieve properties of CUDA device " << devIdx << "!" << std::endl;
        return -1;
    }

    std::cout << "CUDA device " << devIdx << ": " << devProp.name << std::endl;
    std::cout << "  multi processor count: " << devProp.multiProcessorCount << std::endl;
    std::cout << "  shared memory per block: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "  max bolcks per multi processor: " << devProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "  max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "  max threads per multi processor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  max warps per multi processor: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

    return devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor;
}

int choose_cuda_device(cudaDeviceProp& devProp)
{
    int numDev = 0;
    if (cudaGetDeviceCount(&numDev) != cudaError::cudaSuccess) {
        std::cout << "ERROR: Cannot get CUDA device count!" << std::endl;
        return -1;
    }

    if (numDev == 0) {
        std::cout << "ERROR: No CUDA device found!" << std::endl;
        return -2;
    }

    std::cout << numDev << " CUDA devices found!" << std::endl;
    int maxThreads = 0;
    int chosedDevIdx = 0;
    for (int i = 0; i < numDev; ++i) {
        int numThreads = show_cuda_device_info(i);
        if (numThreads > maxThreads) {
            chosedDevIdx = i;
            maxThreads = numThreads;
        }
    }

    cudaGetDeviceProperties(&devProp, chosedDevIdx);
    return chosedDevIdx;
}

void sobel_filtering(FIBITMAP* imgIn, FIBITMAP* imgOut)
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

    cudaDeviceProp devProp;
    int devToUse = choose_cuda_device(devProp);
    if (devToUse < 0) {
        return;
    }

    if (cudaSetDevice(devToUse) != cudaError::cudaSuccess) {
        std::cout << "ERROR: Failed to set CUDA device " << devToUse << "!" << std::endl;
        return;
    }
    std::cout << "Use CUDA device " << devToUse << std::endl;

    
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: sobel_cuda input_image_file" << std::endl;
        return 0;
    }

    FIBITMAP* imgIn = load_image(argv[1], 0);
    if (!imgIn) {
        return -1;
    }

    FIBITMAP* imgOut = FreeImage_Clone(imgIn);

    
    
}
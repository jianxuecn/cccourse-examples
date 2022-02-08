/* sobel_cuda.cu - Sobel filter with CUDA */
#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <assert.h>
#include <cuda_runtime.h>

#include "imgutils.h"
#include "sobel.h"
#include "cudautils.h"

int gDeviceIndex = -1;
cudaDeviceProp gDeviceProp;

// Implementation below is equivalent to linear 2D convolutions for H and V compoonents with:
//      Convo Coefs for Horizontal component
//  {  1,  0, -1,
//     2,  0, -2,
//     1,  0, -1 }
//      Convo Coefs for Vertical component
//  { -1, -2, -1,
//     0,  0,  0,
//     1,  2,  1  };
//*****************************************************************************
__global__ void k_sobel4(
    float* __restrict__ dataOut,
    const uchar4* __restrict__ dataIn,
    const int imageWidth,
    const int imageHeight
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imageWidth || y >= imageHeight) return;

    float temp = 0.0f;
    float hSum[3] = { 0.0f, 0.0f, 0.0f };
    float vSum[3] = { 0.0f, 0.0f, 0.0f };

    int pixOffset = (y - 1) * imageWidth + x - 1;

    // NW
    if (x > 0 && y > 0) {
        hSum[0] += (float)dataIn[pixOffset].x;    // horizontal gradient of Red
        hSum[1] += (float)dataIn[pixOffset].y;    // horizontal gradient of Green
        hSum[2] += (float)dataIn[pixOffset].z;    // horizontal gradient of Blue
        vSum[0] -= (float)dataIn[pixOffset].x;    // vertical gradient of Red
        vSum[1] -= (float)dataIn[pixOffset].y;    // vertical gradient of Green
        vSum[2] -= (float)dataIn[pixOffset].z;    // vertical gradient of Blue   
    }
    pixOffset++;

    // N
    if (y > 0) {
        vSum[0] -= (float)(dataIn[pixOffset].x << 1);  // vertical gradient of Red
        vSum[1] -= (float)(dataIn[pixOffset].y << 1);  // vertical gradient of Green
        vSum[2] -= (float)(dataIn[pixOffset].z << 1);  // vertical gradient of Blue
    }
    pixOffset++;

    // NE
    if (y > 0 && x < imageWidth - 1) {
        hSum[0] -= (float)dataIn[pixOffset].x;    // horizontal gradient of Red
        hSum[1] -= (float)dataIn[pixOffset].y;    // horizontal gradient of Green
        hSum[2] -= (float)dataIn[pixOffset].z;    // horizontal gradient of Blue
        vSum[0] -= (float)dataIn[pixOffset].x;    // vertical gradient of Red
        vSum[1] -= (float)dataIn[pixOffset].y;    // vertical gradient of Green
        vSum[2] -= (float)dataIn[pixOffset].z;    // vertical gradient of Blue
    }
    pixOffset += imageWidth - 2;

    // W
    if (x > 0) {
        hSum[0] += (float)(dataIn[pixOffset].x << 1);  // vertical gradient of Red
        hSum[1] += (float)(dataIn[pixOffset].y << 1);  // vertical gradient of Green
        hSum[2] += (float)(dataIn[pixOffset].z << 1);  // vertical gradient of Blue
    }
    pixOffset++;

    // C
    pixOffset++;

    // E
    if (x < imageWidth - 1) {
        hSum[0] -= (float)(dataIn[pixOffset].x << 1);  // vertical gradient of Red
        hSum[1] -= (float)(dataIn[pixOffset].y << 1);  // vertical gradient of Green
        hSum[2] -= (float)(dataIn[pixOffset].z << 1);  // vertical gradient of Blue
    }
    pixOffset += imageWidth - 2;

    // SW
    if (x > 0 && y < imageHeight - 1) {
        hSum[0] += (float)dataIn[pixOffset].x;    // horizontal gradient of Red
        hSum[1] += (float)dataIn[pixOffset].y;    // horizontal gradient of Green
        hSum[2] += (float)dataIn[pixOffset].z;    // horizontal gradient of Blue
        vSum[0] += (float)dataIn[pixOffset].x;    // vertical gradient of Red
        vSum[1] += (float)dataIn[pixOffset].y;    // vertical gradient of Green
        vSum[2] += (float)dataIn[pixOffset].z;    // vertical gradient of Blue
    }
    pixOffset++;

    // S
    if (y < imageHeight - 1) {
        vSum[0] += (float)(dataIn[pixOffset].x << 1);  // vertical gradient of Red
        vSum[1] += (float)(dataIn[pixOffset].y << 1);  // vertical gradient of Green
        vSum[2] += (float)(dataIn[pixOffset].z << 1);  // vertical gradient of Blue
    }
    pixOffset++;

    // SE
    if (x < imageWidth - 1 && y < imageHeight - 1) {
        hSum[0] -= (float)dataIn[pixOffset].x;    // horizontal gradient of Red
        hSum[1] -= (float)dataIn[pixOffset].y;    // horizontal gradient of Green
        hSum[2] -= (float)dataIn[pixOffset].z;    // horizontal gradient of Blue
        vSum[0] += (float)dataIn[pixOffset].x;    // vertical gradient of Red
        vSum[1] += (float)dataIn[pixOffset].y;    // vertical gradient of Green
        vSum[2] += (float)dataIn[pixOffset].z;    // vertical gradient of Blue
    }

    // Weighted combination of Root-Sum-Square per-color-band H & V gradients for each of RGB
    temp = sqrt((hSum[0] * hSum[0]) + (vSum[0] * vSum[0])) / 3.0f;
    temp += sqrt((hSum[1] * hSum[1]) + (vSum[1] * vSum[1])) / 3.0f;
    temp += sqrt((hSum[2] * hSum[2]) + (vSum[2] * vSum[2])) / 3.0f;

    int outIdx = y * imageWidth + x;
    dataOut[outIdx] = temp;
}

__global__ void k_sobel3(
    float* __restrict__ dataOut,
    const unsigned char* __restrict__ dataIn,
    const int imageWidth,
    const int imageHeight,
    const int pitch
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imageWidth || y >= imageHeight) return;

    float temp = 0.0f;
    float hSum[3] = { 0.0f, 0.0f, 0.0f };
    float vSum[3] = { 0.0f, 0.0f, 0.0f };

    int pixOffset = (y - 1) * pitch + (x - 1) * 3;

    // NW
    if (x > 0 && y > 0) {
        hSum[0] += (float)dataIn[pixOffset + 0];    // horizontal gradient of Red
        hSum[1] += (float)dataIn[pixOffset + 1];    // horizontal gradient of Green
        hSum[2] += (float)dataIn[pixOffset + 2];    // horizontal gradient of Blue
        vSum[0] -= (float)dataIn[pixOffset + 0];    // vertical gradient of Red
        vSum[1] -= (float)dataIn[pixOffset + 1];    // vertical gradient of Green
        vSum[2] -= (float)dataIn[pixOffset + 2];    // vertical gradient of Blue   
    }
    pixOffset += 3;

    // N
    if (y > 0) {
        vSum[0] -= (float)(dataIn[pixOffset + 0] << 1);  // vertical gradient of Red
        vSum[1] -= (float)(dataIn[pixOffset + 1] << 1);  // vertical gradient of Green
        vSum[2] -= (float)(dataIn[pixOffset + 2] << 1);  // vertical gradient of Blue
    }
    pixOffset += 3;

    // NE
    if (y > 0 && x < imageWidth - 1) {
        hSum[0] -= (float)dataIn[pixOffset + 0];    // horizontal gradient of Red
        hSum[1] -= (float)dataIn[pixOffset + 1];    // horizontal gradient of Green
        hSum[2] -= (float)dataIn[pixOffset + 2];    // horizontal gradient of Blue
        vSum[0] -= (float)dataIn[pixOffset + 0];    // vertical gradient of Red
        vSum[1] -= (float)dataIn[pixOffset + 1];    // vertical gradient of Green
        vSum[2] -= (float)dataIn[pixOffset + 2];    // vertical gradient of Blue
    }
    pixOffset += pitch - 6;

    // W
    if (x > 0) {
        hSum[0] += (float)(dataIn[pixOffset + 0] << 1);  // vertical gradient of Red
        hSum[1] += (float)(dataIn[pixOffset + 1] << 1);  // vertical gradient of Green
        hSum[2] += (float)(dataIn[pixOffset + 2] << 1);  // vertical gradient of Blue
    }
    pixOffset += 3;

    // C
    pixOffset += 3;

    // E
    if (x < imageWidth - 1) {
        hSum[0] -= (float)(dataIn[pixOffset + 0] << 1);  // vertical gradient of Red
        hSum[1] -= (float)(dataIn[pixOffset + 1] << 1);  // vertical gradient of Green
        hSum[2] -= (float)(dataIn[pixOffset + 2] << 1);  // vertical gradient of Blue
    }
    pixOffset += pitch - 6;

    // SW
    if (x > 0 && y < imageHeight - 1) {
        hSum[0] += (float)dataIn[pixOffset + 0];    // horizontal gradient of Red
        hSum[1] += (float)dataIn[pixOffset + 1];    // horizontal gradient of Green
        hSum[2] += (float)dataIn[pixOffset + 2];    // horizontal gradient of Blue
        vSum[0] += (float)dataIn[pixOffset + 0];    // vertical gradient of Red
        vSum[1] += (float)dataIn[pixOffset + 1];    // vertical gradient of Green
        vSum[2] += (float)dataIn[pixOffset + 2];    // vertical gradient of Blue
    }
    pixOffset += 3;

    // S
    if (y < imageHeight - 1) {
        vSum[0] += (float)(dataIn[pixOffset + 0] << 1);  // vertical gradient of Red
        vSum[1] += (float)(dataIn[pixOffset + 1] << 1);  // vertical gradient of Green
        vSum[2] += (float)(dataIn[pixOffset + 2] << 1);  // vertical gradient of Blue
    }
    pixOffset += 3;

    // SE
    if (x < imageWidth - 1 && y < imageHeight - 1) {
        hSum[0] -= (float)dataIn[pixOffset + 0];    // horizontal gradient of Red
        hSum[1] -= (float)dataIn[pixOffset + 1];    // horizontal gradient of Green
        hSum[2] -= (float)dataIn[pixOffset + 2];    // horizontal gradient of Blue
        vSum[0] += (float)dataIn[pixOffset + 0];    // vertical gradient of Red
        vSum[1] += (float)dataIn[pixOffset + 1];    // vertical gradient of Green
        vSum[2] += (float)dataIn[pixOffset + 2];    // vertical gradient of Blue
    }

    // Weighted combination of Root-Sum-Square per-color-band H & V gradients for each of RGB
    temp = sqrt((hSum[0] * hSum[0]) + (vSum[0] * vSum[0])) / 3.0f;
    temp += sqrt((hSum[1] * hSum[1]) + (vSum[1] * vSum[1])) / 3.0f;
    temp += sqrt((hSum[2] * hSum[2]) + (vSum[2] * vSum[2])) / 3.0f;

    int outIdx = y * imageWidth + x;
    dataOut[outIdx] = temp;
}

__global__ void k_sobel1(
    float* __restrict__ dataOut,
    const unsigned char* __restrict__ dataIn,
    const int imageWidth,
    const int imageHeight,
    const int pitch
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= imageWidth || y >= imageHeight) return;

    float temp = 0.0f;
    float hSum = 0.0f;
    float vSum = 0.0f;

    int pixOffset = (y - 1) * pitch + x - 1;

    // NW
    if (x > 0 && y > 0) {
        hSum += (float)dataIn[pixOffset];    // horizontal gradient of Red
        vSum -= (float)dataIn[pixOffset];    // vertical gradient of Red
    }
    pixOffset++;

    // N
    if (y > 0) {
        vSum -= (float)(dataIn[pixOffset] << 1);  // vertical gradient of Red
    }
    pixOffset++;

    // NE
    if (y > 0 && x < imageWidth - 1) {
        hSum -= (float)dataIn[pixOffset];    // horizontal gradient of Red
        vSum -= (float)dataIn[pixOffset];    // vertical gradient of Red
    }
    pixOffset += pitch - 2;

    // W
    if (x > 0) {
        hSum += (float)(dataIn[pixOffset] << 1);  // vertical gradient of Red
    }
    pixOffset++;

    // C
    pixOffset++;

    // E
    if (x < imageWidth - 1) {
        hSum -= (float)(dataIn[pixOffset] << 1);  // vertical gradient of Red
    }
    pixOffset += pitch - 2;

    // SW
    if (x > 0 && y < imageHeight - 1) {
        hSum += (float)dataIn[pixOffset];    // horizontal gradient of Red
        vSum += (float)dataIn[pixOffset];    // vertical gradient of Red
    }
    pixOffset++;

    // S
    if (y < imageHeight - 1) {
        vSum += (float)(dataIn[pixOffset] << 1);  // vertical gradient of Red
    }
    pixOffset++;

    // SE
    if (x < imageWidth - 1 && y < imageHeight - 1) {
        hSum -= (float)dataIn[pixOffset];    // horizontal gradient of Red
        vSum += (float)dataIn[pixOffset];    // vertical gradient of Red
    }

    // Weighted combination of Root-Sum-Square H & V gradients
    temp = sqrt((hSum * hSum) + (vSum * vSum));

    int outIdx = y * imageWidth + x;
    dataOut[outIdx] = temp;
}

extern __shared__ float minMaxShared[];
__global__ void k_min_max(
    float* __restrict__ minOut,
    float* __restrict__ maxOut,
    const float* __restrict__ dataIn,
    const unsigned int n
) {
    float* minShared = (float*)minMaxShared;
    float* maxShared = (float*)&minShared[blockDim.x];
    //float minShared[1024];
    //float maxShared[1024];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    minShared[tid] = (i < n) ? dataIn[i] : FLT_MAX;
    maxShared[tid] = (i < n) ? dataIn[i] : FLT_MIN;
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (minShared[tid + s] < minShared[tid]) minShared[tid] = minShared[tid + s];
            if (maxShared[tid + s] > maxShared[tid]) maxShared[tid] = maxShared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned int outIdx = blockIdx.x;
        minOut[outIdx] = minShared[0];
        maxOut[outIdx] = maxShared[0];
    }
}

extern __shared__ float minMaxSharedIter[];
__global__ void k_min_max_iter(float* __restrict__ minInOut, float* __restrict__ maxInOut, const unsigned int n)
{
    float* minShared = (float*)minMaxSharedIter;
    float* maxShared = (float*)&minShared[blockDim.x];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    minShared[tid] = (i < n) ? minInOut[i] : FLT_MAX;
    maxShared[tid] = (i < n) ? maxInOut[i] : FLT_MIN;
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (minShared[tid + s] < minShared[tid]) minShared[tid] = minShared[tid + s];
            if (maxShared[tid + s] > maxShared[tid]) maxShared[tid] = maxShared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned int outIdx = blockIdx.x;
        minInOut[outIdx] = minShared[0];
        maxInOut[outIdx] = maxShared[0];
    }
}

__global__ void k_scale_pixels4(
    uchar4* __restrict__ dataOut,
    const float* __restrict__ dataIn,
    const float dataInMin,
    const float dataInMax,
    const unsigned int n
) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        float temp = dataIn[i];
        temp = 255.0f * (temp - dataInMin) / (dataInMax - dataInMin);
        dataOut[i] = make_uchar4(temp, temp, temp, 255);
    }
}

__global__ void k_scale_pixels3(
    unsigned char* __restrict__ dataOut,
    const float* __restrict__ dataIn,
    const float dataInMin,
    const float dataInMax,
    const int imageWidth,
    const int imageHeight,
    const int pitch
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageWidth && y < imageHeight) {
        int posIn = y * imageWidth + x;
        int posOut = y * pitch + x * 3;
        float temp = dataIn[posIn];
        temp = 255.0f * (temp - dataInMin) / (dataInMax - dataInMin);
        unsigned char outVal = (unsigned char)temp;
        dataOut[posOut + 0] = outVal;
        dataOut[posOut + 1] = outVal;
        dataOut[posOut + 2] = outVal;
    }
}

__global__ void k_scale_pixels1(
    unsigned char* __restrict__ dataOut,
    const float* __restrict__ dataIn,
    const float dataInMin,
    const float dataInMax,
    const int imageWidth,
    const int imageHeight,
    const int pitch
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < imageWidth && y < imageHeight) {
        int posIn = y * imageWidth + x;
        int posOut = y * pitch + x;
        float temp = dataIn[posIn];
        temp = 255.0f * (temp - dataInMin) / (dataInMax - dataInMin);
        dataOut[posOut] = (unsigned char)temp;
    }
}

void sobel_filtering(FIBITMAP* imgIn, FIBITMAP* imgOut)
{
    int width = FreeImage_GetWidth(imgIn);
    assert(width == FreeImage_GetWidth(imgOut));

    int height = FreeImage_GetHeight(imgIn);
    assert(height == FreeImage_GetHeight(imgOut));
    assert(FreeImage_GetBPP(imgIn) == FreeImage_GetBPP(imgOut));

    //int widthBytes = FreeImage_GetLine(imgIn);
    int pitch = FreeImage_GetPitch(imgIn);
    int chNum = 0;
    switch (FreeImage_GetBPP(imgIn))
    {
    case 8: chNum = 1; break;
    case 24: chNum = 3; break;
    case 32: chNum = 4; break;
    default:
        LOG_INFO("Unsupported image format!");
        break;
    }

    if (chNum == 0) return;

    unsigned int totalPixelNum = width * height;
    size_t imageBytes = pitch * height;

    cudaError_t cudaRetCode;
    unsigned char* devImageDataIn = nullptr;
    unsigned char* devImageDataOut = nullptr;
    float* devMiddleResult = nullptr;
    float* devMinValues = nullptr;
    float* devMaxValues = nullptr;

    cudaRetCode = cudaMalloc(&devImageDataIn, imageBytes);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for input image on device!");
    
    cudaRetCode = cudaMalloc(&devImageDataOut, imageBytes);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for output image on device!");

    cudaRetCode = cudaMalloc(&devMiddleResult, totalPixelNum * sizeof(float));
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for middle result on device!");

    unsigned int minMaxBlockThreadNum = gDeviceProp.maxThreadsPerBlock;
    unsigned int minMaxBlockNum = (totalPixelNum + minMaxBlockThreadNum - 1) / minMaxBlockThreadNum;
    LOG_DEBUG("minMaxBlockThreadNum = " << minMaxBlockThreadNum);
    cudaRetCode = cudaMalloc(&devMinValues, minMaxBlockNum * sizeof(float));
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for min reduction on device!");
    cudaRetCode = cudaMalloc(&devMaxValues, minMaxBlockNum * sizeof(float));
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for max reduction on device!");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // load source image data from host to device
    BYTE* dataIn = FreeImage_GetBits(imgIn);
    cudaRetCode = cudaMemcpy(devImageDataIn, dataIn, imageBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot load source image data to device!");

    // Pass 1: do sobel filtering
    dim3 sobelBlockDim(32, gDeviceProp.maxThreadsPerBlock / 32);
    dim3 sobelGridDim((width + sobelBlockDim.x - 1) / sobelBlockDim.x, (height + sobelBlockDim.y - 1) / sobelBlockDim.y);
    if (chNum == 4) {
        k_sobel4<<<sobelGridDim, sobelBlockDim>>>(devMiddleResult, (uchar4 *)devImageDataIn, width, height);
    } else if (chNum == 3) {
        k_sobel3<<<sobelGridDim, sobelBlockDim>>>(devMiddleResult, devImageDataIn, width, height, pitch);
    } else if (chNum == 1) {
        k_sobel1<<<sobelGridDim, sobelBlockDim>>>(devMiddleResult, devImageDataIn, width, height, pitch);
    }

    // Pass 2: get min & max pixel value
    k_min_max<<<minMaxBlockNum, minMaxBlockThreadNum, minMaxBlockThreadNum*2*sizeof(float)>>>(devMinValues, devMaxValues, devMiddleResult, totalPixelNum);
    unsigned int valNum = minMaxBlockNum;
    while (minMaxBlockNum > 1) {
        valNum = minMaxBlockNum;
        minMaxBlockNum = (minMaxBlockNum + minMaxBlockThreadNum - 1) / minMaxBlockThreadNum;
        k_min_max_iter<<<minMaxBlockNum, minMaxBlockThreadNum, minMaxBlockThreadNum*2*sizeof(float)>>>(devMinValues, devMaxValues, valNum);
    }
    //cudaDeviceSynchronize();
    //CUDA_DEBUG("CUDA error occurred!");
    float minVal, maxVal;
    cudaRetCode = cudaMemcpy(&minVal, devMinValues, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot copy min value to host!");
    cudaRetCode = cudaMemcpy(&maxVal, devMaxValues, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot copy max value to host!");

    // Pass 3: scale pixel to [0, 255]
    if (chNum == 4) {
        unsigned int scaleBlockThreadNum = gDeviceProp.maxThreadsPerBlock;
        unsigned int scaleBlockNum = (totalPixelNum + minMaxBlockThreadNum - 1) / minMaxBlockThreadNum;
        k_scale_pixels4<<<scaleBlockNum, scaleBlockThreadNum>>>((uchar4*)devImageDataOut, devMiddleResult, minVal, maxVal, totalPixelNum);
    } else {
        dim3 scaleBlockDim(32, gDeviceProp.maxThreadsPerBlock / 32);
        dim3 scaleGridDim((width + sobelBlockDim.x - 1) / sobelBlockDim.x, (height + sobelBlockDim.y - 1) / sobelBlockDim.y);
        if (chNum == 3) {
            k_scale_pixels3<<<scaleGridDim, scaleBlockDim>>>(devImageDataOut, devMiddleResult, minVal, maxVal, width, height, pitch);
        } else if (chNum == 1) {
            k_scale_pixels1<<<scaleGridDim, scaleBlockDim>>>(devImageDataOut, devMiddleResult, minVal, maxVal, width, height, pitch);
        }
    }

    // store result
    BYTE* dataOut = FreeImage_GetBits(imgOut);
    cudaRetCode = cudaMemcpy(dataOut, devImageDataOut, imageBytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot store result image data to host!");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //cudaDeviceSynchronize();

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    LOG_INFO("the value of minimum: " << minVal);
    LOG_INFO("the value of maximum: " << maxVal);
    LOG_INFO("The total time for execution is: " << elapsedTime / 1000.0 << "s");

    cudaFree(devImageDataIn);
    cudaFree(devImageDataOut);
    cudaFree(devMiddleResult);
    cudaFree(devMinValues);
    cudaFree(devMaxValues);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[])
{
    cudaError_t cudaRetCode;

    gDeviceIndex = choose_cuda_device(gDeviceProp);
    if (gDeviceIndex < 0) {
        return -1;
    }

    cudaRetCode = cudaSetDevice(gDeviceIndex);
    CUDA_CHECK_RETURN_N(cudaRetCode, -2, "Failed to set CUDA device " << gDeviceIndex << "!");

    LOG_INFO("Use CUDA device " << gDeviceIndex);

    if (argc < 2) {
        std::cout << "Usage: sobel_cuda input_image_file" << std::endl;
        return 0;
    }

    FIBITMAP* imgIn = load_image(argv[1], 0);
    if (!imgIn) {
        return -3;
    }

    FIBITMAP* imgOut = FreeImage_Clone(imgIn);

    sobel_filtering(imgIn, imgOut);

    cudaDeviceReset();

    std::string fnIn(argv[1]);
    std::size_t found = fnIn.find_last_of('.');
    std::string fnOut = fnIn.substr(0, found) + "_cuda_out" + fnIn.substr(found);

    int r = 0;
    if (!save_image(fnOut.c_str(), imgOut, 0)) {
        LOG_ERROR("Failed to save output image file " << fnOut);
        r = -4;
    }

    FreeImage_Unload(imgIn);
    FreeImage_Unload(imgOut);
    return r;    
}
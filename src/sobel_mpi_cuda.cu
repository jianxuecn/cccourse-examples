/* sobel_cuda.cu - Sobel filter with CUDA */
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <assert.h>
#include <float.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "imgutils.h"
#include "sobel.h"
#include "cudautils.h"

#define MPI_ERROR_CHECK(retCode) do { if (retCode != MPI_SUCCESS) { std::cout << "| MPI ERROR | in " __FILE__ ", line " << __LINE__ << ": "<< retCode << std::endl; exit(-1); } } while (0)

int gDeviceIndex = -1;
cudaDeviceProp gDeviceProp;

int const MASTER = 0;
char gProcessorName[MPI_MAX_PROCESSOR_NAME];

int gImageWidth;
int gImageHeight;
int gImagePitch;
int gImageChNum;

FIBITMAP* gImageIn;
FIBITMAP* gImageOut;

// Implementation below is equivalent to linear 2D convolutions for H and V components with:
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
    const int imageWidthOut,
    const int imageHeightOut,
    const int marginX,
    const int marginY
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int imageWidthIn = imageWidthOut + (marginX * 2);
    if (x < imageWidthOut && y < imageHeightOut) {
        int posIn = (y + marginY) * imageWidthIn + x + marginX;
        int posOut = y * imageWidthOut + x;
        float temp = dataIn[posIn];
        temp = 255.0f * (temp - dataInMin) / (dataInMax - dataInMin);
        dataOut[posOut] = make_uchar4(temp, temp, temp, 255);
    }
}

__global__ void k_scale_pixels3(
    unsigned char* __restrict__ dataOut,
    const float* __restrict__ dataIn,
    const float dataInMin,
    const float dataInMax,
    const int imageWidthOut,
    const int imageHeightOut,
    const int marginX,
    const int marginY
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int imageWidthIn = imageWidthOut + (marginX * 2);
    if (x < imageWidthOut && y < imageHeightOut) {
        int posIn = (y + marginY) * imageWidthIn + x + marginX;
        int posOut = (y * imageWidthOut + x) * 3;
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
    const int imageWidthOut,
    const int imageHeightOut,
    const int marginX,
    const int marginY
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int imageWidthIn = imageWidthOut + (marginX * 2);
    if (x < imageWidthOut && y < imageHeightOut) {
        int posIn = (y + marginY) * imageWidthIn + x + marginX;
        int posOut = y * imageWidthOut + x;
        float temp = dataIn[posIn];
        temp = 255.0f * (temp - dataInMin) / (dataInMax - dataInMin);
        dataOut[posOut] = (unsigned char)temp;
    }
}

void sobel_filtering_gpu(int rank, int np,
    int imageWidthIn, int imageHeightIn, std::vector<BYTE> const& imageDataIn,
    int imageWidthOut, int imageHeightOut, std::vector<BYTE>& imageDataOut)
{
    LOG_INFO("Filtering of input image on Process " << rank << " of " << gProcessorName << " start ...");

    unsigned int const totalPixelNum = imageWidthIn * imageHeightIn;
    int const pitchIn = imageWidthIn * gImageChNum;
    size_t const imageInBytes = totalPixelNum * gImageChNum;
    size_t const imageOutBytes = imageWidthOut * imageHeightOut * gImageChNum;

    cudaError_t cudaRetCode;
    unsigned char* devImageDataIn = nullptr;
    unsigned char* devImageDataOut = nullptr;
    float* devMiddleResult = nullptr;
    float* devMinValues = nullptr;
    float* devMaxValues = nullptr;

    cudaRetCode = cudaMalloc(&devImageDataIn, imageInBytes);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for input image on device!");

    cudaRetCode = cudaMalloc(&devImageDataOut, imageOutBytes);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for output image on device!");

    cudaRetCode = cudaMalloc(&devMiddleResult, totalPixelNum * sizeof(float));
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for middle result on device!");

    unsigned int minMaxBlockThreadNum = gDeviceProp.maxThreadsPerBlock;
    if (!is_pow2(minMaxBlockThreadNum)) {
        minMaxBlockThreadNum = next_pow2((minMaxBlockThreadNum + 2) >> 1);
        LOG_INFO("Reduce minmax kernel block size to (power of 2): " << minMaxBlockThreadNum);
    }

    unsigned int minMaxBlockNum = (totalPixelNum + minMaxBlockThreadNum - 1) / minMaxBlockThreadNum;
    cudaRetCode = cudaMalloc(&devMinValues, minMaxBlockNum * sizeof(float));
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for min reduction on device!");
    cudaRetCode = cudaMalloc(&devMaxValues, minMaxBlockNum * sizeof(float));
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot allocate memory for max reduction on device!");

    // load source image data from host to device
    cudaRetCode = cudaMemcpy(devImageDataIn, imageDataIn.data(), imageInBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot load source image data to device!");

    // Pass 1: do sobel filtering
    dim3 sobelBlockDim(32, gDeviceProp.maxThreadsPerBlock / 32);
    dim3 sobelGridDim((imageWidthIn + sobelBlockDim.x - 1) / sobelBlockDim.x, (imageHeightIn + sobelBlockDim.y - 1) / sobelBlockDim.y);
    if (gImageChNum == 4) {
        k_sobel4<<<sobelGridDim, sobelBlockDim>>>(devMiddleResult, (uchar4*)devImageDataIn, imageWidthIn, imageHeightIn);
    } else if (gImageChNum == 3) {
        k_sobel3<<<sobelGridDim, sobelBlockDim>>>(devMiddleResult, devImageDataIn, imageWidthIn, imageHeightIn, pitchIn);
    } else if (gImageChNum == 1) {
        k_sobel1<<<sobelGridDim, sobelBlockDim>>>(devMiddleResult, devImageDataIn, imageWidthIn, imageHeightIn, pitchIn);
    }

    // Pass 2: get min & max pixel value
    k_min_max<<<minMaxBlockNum, minMaxBlockThreadNum, minMaxBlockThreadNum * 2 * sizeof(float)>>>(devMinValues, devMaxValues, devMiddleResult, totalPixelNum);
    unsigned int minMaxIterBlockNum = minMaxBlockNum;
    while (minMaxIterBlockNum > 1) {
        unsigned int valNum = minMaxIterBlockNum;
        minMaxIterBlockNum = (minMaxIterBlockNum + minMaxBlockThreadNum - 1) / minMaxBlockThreadNum;
        k_min_max_iter<<<minMaxIterBlockNum, minMaxBlockThreadNum, minMaxBlockThreadNum * 2 * sizeof(float)>>>(devMinValues, devMaxValues, valNum);
    }
    float minVal, maxVal;
    cudaRetCode = cudaMemcpy(&minVal, devMinValues, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot copy min value to host!");
    cudaRetCode = cudaMemcpy(&maxVal, devMaxValues, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(cudaRetCode, "Cannot copy max value to host!");

    LOG_INFO("Min/Max on Process " << rank << " of " << gProcessorName << ": " << minVal << "/" << maxVal);

    MPI_Status status;

    if (rank != MASTER) {
        // send min & max to MASTER
        LOG_INFO("Send min & max to MASTER ...");
        MPI_Send(&minVal, 1, MPI_FLOAT, MASTER, rank + 100, MPI_COMM_WORLD);
        MPI_Send(&maxVal, 1, MPI_FLOAT, MASTER, rank + 110, MPI_COMM_WORLD);

        // receive global min & max from MASTER
        LOG_INFO("Receive global min & max from MASTER ...");
        MPI_Recv(&minVal, 1, MPI_FLOAT, MASTER, rank + 200, MPI_COMM_WORLD, &status);
        MPI_Recv(&maxVal, 1, MPI_FLOAT, MASTER, rank + 210, MPI_COMM_WORLD, &status);
    }

    if (rank == MASTER) {
        // collect the min & max values from other nodes and calculate the global min & max
        double tempMin, tempMax;
        LOG_INFO("Receive min & max from other nodes ...");
        for (int i = 1; i < np; ++i) {
            MPI_Recv(&tempMin, 1, MPI_FLOAT, i, i + 100, MPI_COMM_WORLD, &status);
            MPI_Recv(&tempMax, 1, MPI_FLOAT, i, i + 110, MPI_COMM_WORLD, &status);
            if (tempMin < minVal) minVal = tempMin;
            if (tempMax > maxVal) maxVal = tempMax;
        }

        LOG_INFO("Min/Max Global: " << minVal << "/" << maxVal);

        // send global min & max to other nodes
        LOG_INFO("Send global min & max to other nodes ...");
        for (int i = 1; i < np; ++i) {
            MPI_Send(&minVal, 1, MPI_FLOAT, i, i + 200, MPI_COMM_WORLD);
            MPI_Send(&maxVal, 1, MPI_FLOAT, i, i + 210, MPI_COMM_WORLD);
        }
    }

    if (maxVal - minVal > 0) {
        // Pass 3: scale pixel to [0, 255]
        if (gImageChNum == 4) {
            unsigned int scaleBlockThreadNum = gDeviceProp.maxThreadsPerBlock;
            unsigned int scaleBlockNum = (totalPixelNum + minMaxBlockThreadNum - 1) / minMaxBlockThreadNum;
            k_scale_pixels4<<<scaleBlockNum, scaleBlockThreadNum>>>((uchar4*)devImageDataOut, devMiddleResult, minVal, maxVal, imageWidthOut, imageHeightOut, 1, 1);
        } else {
            dim3 scaleBlockDim(32, gDeviceProp.maxThreadsPerBlock / 32);
            dim3 scaleGridDim((imageWidthOut + sobelBlockDim.x - 1) / sobelBlockDim.x, (imageHeightOut + sobelBlockDim.y - 1) / sobelBlockDim.y);
            if (gImageChNum == 3) {
                k_scale_pixels3<<<scaleGridDim, scaleBlockDim>>>(devImageDataOut, devMiddleResult, minVal, maxVal, imageWidthOut, imageHeightOut, 1, 1);
            } else if (gImageChNum == 1) {
                k_scale_pixels1<<<scaleGridDim, scaleBlockDim>>>(devImageDataOut, devMiddleResult, minVal, maxVal, imageWidthOut, imageHeightOut, 1, 1);
            }
        }
        cudaRetCode = cudaMemcpy(imageDataOut.data(), devImageDataOut, imageOutBytes, cudaMemcpyDeviceToHost);
        CUDA_CHECK_RETURN(cudaRetCode, "Cannot store result image data to host!");
    } else {
        LOG_INFO("Nothing exists on Process " << rank << " of " << gProcessorName << "!");
    }

    cudaDeviceSynchronize();

    cudaFree(devImageDataIn);
    cudaFree(devImageDataOut);
    cudaFree(devMiddleResult);
    cudaFree(devMinValues);
    cudaFree(devMaxValues);

    LOG_INFO("Filtering of input image on Process " << rank << " of " << gProcessorName << " finish.");
}

void sobel_filtering_cpu(int rank, int np,
    int imageWidthIn, int imageHeightIn, std::vector<BYTE> const& imageDataIn,
    int imageWidthOut, int imageHeightOut, std::vector<BYTE>& imageDataOut)
{
    LOG_INFO("Filtering of input image on Process " << rank << " of " << gProcessorName << " start ...");

    float pixel_value_min, pixel_value_max;
    pixel_value_min = FLT_MAX;
    pixel_value_max = FLT_MIN;

    int const pitchIn = imageWidthIn * gImageChNum;
    for (int y = 1; y < imageHeightIn - 1; ++y) {
        for (int x = 1; x < imageWidthIn - 1; ++x) {
            float pixel_value_x = 0.0;
            float pixel_value_y = 0.0;
            float pixel_value = 0.0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    pixel_value_x += sobel_weight_x[j + 1][i + 1] * to_gray(imageDataIn.data(), x + i, y + j, pitchIn, gImageChNum);
                    pixel_value_y += sobel_weight_y[j + 1][i + 1] * to_gray(imageDataIn.data(), x + i, y + j, pitchIn, gImageChNum);
                }
            }
            pixel_value = sqrt((pixel_value_x * pixel_value_x) + (pixel_value_y * pixel_value_y));
            if (pixel_value < pixel_value_min) pixel_value_min = pixel_value;
            if (pixel_value > pixel_value_max) pixel_value_max = pixel_value;
        }
    }

    LOG_INFO("Min/Max on Process " << rank << " of " << gProcessorName << ": " << pixel_value_min << "/" << pixel_value_max);

    MPI_Status status;

    if (rank != MASTER) {
        // send min & max to MASTER
        LOG_INFO("Send min & max to MASTER ...");
        MPI_Send(&pixel_value_min, 1, MPI_FLOAT, MASTER, rank + 100, MPI_COMM_WORLD);
        MPI_Send(&pixel_value_max, 1, MPI_FLOAT, MASTER, rank + 110, MPI_COMM_WORLD);

        // receive global min & max from MASTER
        LOG_INFO("Receive global min & max from MASTER ...");
        MPI_Recv(&pixel_value_min, 1, MPI_FLOAT, MASTER, rank + 200, MPI_COMM_WORLD, &status);
        MPI_Recv(&pixel_value_max, 1, MPI_FLOAT, MASTER, rank + 210, MPI_COMM_WORLD, &status);
    }

    if (rank == MASTER) {
        // collect the min & max values from other nodes and calculate the global min & max
        float tempMin, tempMax;
        LOG_INFO("Receive min & max from other nodes ...");
        for (int i = 1; i < np; ++i) {
            MPI_Recv(&tempMin, 1, MPI_FLOAT, i, i + 100, MPI_COMM_WORLD, &status);
            MPI_Recv(&tempMax, 1, MPI_FLOAT, i, i + 110, MPI_COMM_WORLD, &status);
            if (tempMin < pixel_value_min) pixel_value_min = tempMin;
            if (tempMax > pixel_value_max) pixel_value_max = tempMax;
        }

        LOG_INFO("Min/Max Global: " << pixel_value_min << "/" << pixel_value_max);

        // send global min & max to other nodes
        LOG_INFO("Send global min & max to other nodes ...");
        for (int i = 1; i < np; ++i) {
            MPI_Send(&pixel_value_min, 1, MPI_FLOAT, i, i + 200, MPI_COMM_WORLD);
            MPI_Send(&pixel_value_max, 1, MPI_FLOAT, i, i + 210, MPI_COMM_WORLD);
        }
    }

    if (pixel_value_max - pixel_value_min == 0) {
        LOG_INFO("Nothing exists on Process " << rank << " of " << gProcessorName << "!");
        return;
    }

    /* Generation of imgOut after linear transformation */
    int const pitchOut = imageWidthOut * gImageChNum;
    for (int y = 1; y < imageHeightIn - 1; ++y) {
        for (int x = 1; x < imageWidthIn - 1; x++) {
            float pixel_value_x = 0.0;
            float pixel_value_y = 0.0;
            float pixel_value = 0.0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    pixel_value_x += sobel_weight_x[j + 1][i + 1] * to_gray(imageDataIn.data(), x + i, y + j, pitchIn, gImageChNum);
                    pixel_value_y += sobel_weight_y[j + 1][i + 1] * to_gray(imageDataIn.data(), x + i, y + j, pitchIn, gImageChNum);
                }
            }
            pixel_value = sqrt((pixel_value_x * pixel_value_x) + (pixel_value_y * pixel_value_y));
            pixel_value = 255.0 * (pixel_value - pixel_value_min) / (pixel_value_max - pixel_value_min);
            set_pixel(imageDataOut.data(), (BYTE)pixel_value, x - 1, y - 1, pitchOut, gImageChNum);
        }
    }

    LOG_INFO("Filtering of input image on Process " << rank << " of " << gProcessorName << " finish.");
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: sobel_mpi_cuda input_image_file" << std::endl;
        return 0;
    }

    gImageIn = nullptr;
    gImageOut = nullptr;
    gImageWidth = 0;
    gImageHeight = 0;
    gImagePitch = 0;
    gImageChNum = 0;

    double start_time, end_time;
    int retCode;
    retCode = MPI_Init(&argc, &argv);
    MPI_ERROR_CHECK(retCode);

    int rank, np, nameLen;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(gProcessorName, &nameLen);

    if (rank == MASTER) {

        gImageIn = load_image(argv[1], 0);
        if (!gImageIn) {
            LOG_ERROR("Cannot load image " << argv[1]);
            MPI_Finalize();
            return 1;
        }

        gImageOut = FreeImage_Clone(gImageIn);

        gImageWidth = FreeImage_GetWidth(gImageIn);
        gImageHeight = FreeImage_GetHeight(gImageIn);
        gImagePitch = FreeImage_GetPitch(gImageIn);
        gImageChNum = 0;
        switch (FreeImage_GetBPP(gImageIn))
        {
        case 8: gImageChNum = 1; break;
        case 24: gImageChNum = 3; break;
        case 32: gImageChNum = 4; break;
        default:
            break;
        }

        start_time = MPI_Wtime();
    }

    cudaError_t cudaRetCode;
    bool cudaEnabled = false;
    gDeviceIndex = choose_cuda_device(gDeviceProp);
    if (gDeviceIndex < 0) {
        LOG_INFO("No cuda device found!");
    } else {
        cudaRetCode = cudaSetDevice(gDeviceIndex);
        if (cudaRetCode == cudaError::cudaSuccess) {
            cudaEnabled = true;
            LOG_INFO("Use CUDA device " << gDeviceIndex);
        } else {
            LOG_CUDA_ERROR(cudaRetCode, "Failed to set CUDA device " << gDeviceIndex << "!");
        }
    }

    // Broadcast the image size to all the nodes
    MPI_Bcast(&gImageChNum, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&gImageWidth, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&gImageHeight, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&gImagePitch, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // MPI_Barrier for Synchronization
    MPI_Barrier(MPI_COMM_WORLD);

    if (gImageChNum == 0) {
        LOG_ERROR("Unsupported image format!");
        MPI_Finalize();
        if (rank == MASTER) {
            FreeImage_Unload(gImageIn);
            FreeImage_Unload(gImageOut);
        }
        return 2;
    }

    int imageDim[2] = { gImageHeight, gImagePitch };
    int scatterTypeDim[2] = { 1, gImageWidth * gImageChNum };
    int scatterTypeOff[2] = { 0, 0 };
    MPI_Datatype scatterType, scatterSubarrayType;
    MPI_Type_create_subarray(2, imageDim, scatterTypeDim, scatterTypeOff, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &scatterType);
    MPI_Type_create_resized(scatterType, 0, gImagePitch, &scatterSubarrayType);
    MPI_Type_commit(&scatterSubarrayType);

    int gatherTypeDim[2] = { 1, (gImageWidth - 2) * gImageChNum };
    int gatherTypeOff[2] = { 0, gImageChNum };
    MPI_Datatype gatherType, gatherSubarrayType;
    MPI_Type_create_subarray(2, imageDim, gatherTypeDim, gatherTypeOff, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &gatherType);
    MPI_Type_create_resized(gatherType, 0, gImagePitch, &gatherSubarrayType);
    MPI_Type_commit(&gatherSubarrayType);

    int gatherImageWidth = gImageWidth - 2;
    int gatherImageHeight = (gImageHeight - 2) / np;
    int gatherImageRestLines = (gImageHeight - 2) - gatherImageHeight * np;

    int scatterImageWidth = gImageWidth;
    int scatterImageHeight = gatherImageHeight + 2;

    // allocate memory of image chunk for each node
    std::vector<BYTE> localImageIn(scatterImageWidth * (scatterImageHeight + 1) * gImageChNum, 0);
    std::vector<BYTE> localImageOut(gatherImageWidth * (gatherImageHeight + 1) * gImageChNum, 0);

    std::vector<int>sendCounts(np, 0);
    std::vector<int>sendDispls(np, 0);
    std::vector<int>recvCounts(np, 0);
    std::vector<int>recvDispls(np, 0);

    for (int i = 0, li = 1; i < np; ++i) {
        recvDispls[i] = li;
        recvCounts[i] = gatherImageHeight;
        if (i > 0 && gatherImageRestLines > 0) {
            recvCounts[i]++;
            gatherImageRestLines--;
        }
        sendDispls[i] = li - 1;
        sendCounts[i] = recvCounts[i] + 2;

        li += recvCounts[i];
    }

    MPI_Scatterv(FreeImage_GetBits(gImageIn), sendCounts.data(), sendDispls.data(), scatterSubarrayType,
        localImageIn.data(), scatterImageWidth * sendCounts[rank] * gImageChNum, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    if (cudaEnabled) {
        sobel_filtering_gpu(rank, np, scatterImageWidth, sendCounts[rank], localImageIn, gatherImageWidth, recvCounts[rank], localImageOut);
    } else {
        sobel_filtering_cpu(rank, np, scatterImageWidth, sendCounts[rank], localImageIn, gatherImageWidth, recvCounts[rank], localImageOut);
    }

    MPI_Gatherv(localImageOut.data(), gatherImageWidth * recvCounts[rank] * gImageChNum, MPI_UNSIGNED_CHAR,
        FreeImage_GetBits(gImageOut), recvCounts.data(), recvDispls.data(), gatherSubarrayType,
        0, MPI_COMM_WORLD);

    if (rank == MASTER) {
        end_time = MPI_Wtime();
    }

    MPI_Type_free(&scatterSubarrayType);
    MPI_Type_free(&gatherSubarrayType);

    cudaDeviceReset();

    int r = 0;
    if (rank == MASTER) {
        double execTime = end_time - start_time;
        LOG_INFO("The total time for execution is " << execTime);

        // output execution info (number of processors & execution time)
        std::string logFileName = argv[0];
        logFileName += "_np_t.log";
        std::ofstream logFile(logFileName.c_str(), std::ios::out | std::ios::app);
        logFile << np << "," << execTime << std::endl;
        logFile.close();

        std::string fnIn(argv[1]);
        std::size_t found = fnIn.find_last_of('.');
        std::string fnOut = fnIn.substr(0, found) + "_mpi_cuda_out" + fnIn.substr(found);

        if (!save_image(fnOut.c_str(), gImageOut, 0)) {
            LOG_ERROR("Failed to save output image file " << fnOut);
            r = -2;
        }

        FreeImage_Unload(gImageIn);
        FreeImage_Unload(gImageOut);
    }

    MPI_Finalize();
    return r;
}
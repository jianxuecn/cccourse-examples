/**
 * -------------------------------------------------------------------------------
 * This source file is part of cccourse-examples, one of the examples for
 * the Cloud Computing Course and the Computer Graphics Course at the School
 * of Engineering Science (SES), University of Chinese Academy of Sciences (UCAS).
 * Copyright (C) 2022 Xue Jian (xuejian@ucas.ac.cn)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------------
 */
#ifndef __cudautils_h
#define __cudautils_h

#include <cuda_runtime.h>

#include "logutils.h"

#define LOG_CUDA_ERROR(cudaCode, errInfo) LOG_ERROR(errInfo << " ErrorCode: " << cudaCode << ", Description: " << cudaGetErrorString(cudaCode))

#define CUDA_CHECK_RETURN_N(cudaCode, retCode, errInfo) \
  do { \
    if (cudaCode != cudaError::cudaSuccess) { \
      LOG_CUDA_ERROR(cudaCode, errInfo); \
      return retCode; \
    } \
  } while (0)

#define CUDA_CHECK_RETURN(cudaCode, errInfo) \
  do { \
    if (cudaCode != cudaError::cudaSuccess) { \
      LOG_CUDA_ERROR(cudaCode, errInfo); \
      return; \
    } \
  } while (0)

#define CUDA_DEBUG(errInfo) \
  do { \
    cudaError_t _x_cudaErrDbg = cudaGetLastError(); \
    if (_x_cudaErrDbg != cudaError::cudaSuccess) { \
      LOG_CUDA_ERROR(_x_cudaErrDbg, errInfo); \
    } \
  } while (0)

#define CUDA_DEBUG_RETURN_N(retCode, errInfo) \
  do { \
    cudaError_t _x_cudaErrDbg = cudaGetLastError(); \
    if (_x_cudaErrDbg != cudaError::cudaSuccess) { \
      LOG_CUDA_ERROR(_x_cudaErrDbg, errInfo); \
      return retCode; \
    } \
  } while (0)

#define CUDA_DEBUG_RETURN(errInfo) \
  do { \
    cudaError_t _x_cudaErrDbg = cudaGetLastError(); \
    if (_x_cudaErrDbg != cudaError::cudaSuccess) { \
      LOG_CUDA_ERROR(_x_cudaErrDbg, errInfo); \
      return; \
    } \
  } while (0)

inline int show_cuda_device_info(int devIdx)
{
    cudaDeviceProp devProp;
    cudaError_t cudaRetCode;
    cudaRetCode = cudaGetDeviceProperties(&devProp, devIdx);
    CUDA_CHECK_RETURN_N(cudaRetCode, -1, "Failed to retrieve properties of CUDA device " << devIdx << "!");

    LOG_INFO("CUDA device " << devIdx << ": " << devProp.name);
    LOG_INFO("  multi processor count: " << devProp.multiProcessorCount);
    LOG_INFO("  shared memory per block: " << devProp.sharedMemPerBlock / 1024.0 << " KB");
    //LOG_INFO("  max bolcks per multi processor: " << devProp.maxBlocksPerMultiProcessor); // not supported befor CUDA 11.0
    LOG_INFO("  max threads per block: " << devProp.maxThreadsPerBlock);
    LOG_INFO("  max threads per multi processor: " << devProp.maxThreadsPerMultiProcessor);
    LOG_INFO("  max warps per multi processor: " << devProp.maxThreadsPerMultiProcessor / 32);

    return devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor;
}

inline int choose_cuda_device(cudaDeviceProp& devProp)
{
    int numDev = 0;
    cudaError_t cudaRetCode;

    cudaRetCode = cudaGetDeviceCount(&numDev);
    CUDA_CHECK_RETURN_N(cudaRetCode, -1, "Cannot get CUDA device count!");

    if (numDev == 0) {
        LOG_ERROR("No CUDA device found!");
        return -2;
    }

    LOG_INFO(numDev << " CUDA devices found!");
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

#endif

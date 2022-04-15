# cccourse-examples

## Introduction

This project provide a set of Sobel edge detection examples for the *Cloud Computing Course* and the *Computer Graphics Course* at the School of Engineering Science (SES), University of Chinese Academy of Sciences (UCAS).

This project demonstrates how to write distributed parallel programs for fast image processing in a cluster environment with many CPU and GPU nodes. Following main topics have been included and implemented:

1. Perform CPU parallel acceleration of Sobel edge detection algorithm by OpenMP on a single node of a cluster.
2. Perform GPU parallel acceleration of Sobel edge detection algorithm by OpenCL and CUDA on a single node of a cluster.
3. Perform distributed parallel acceleration of Sobel edge detection algorithm by MPI on a cluster.
4. Perform mix-mode distributed parallel acceleration of Sobel edge detection algorithm by MPI+OpenMP and MPI+CUDA on a cluster. 

## Dependencies

1. **OpenMP**: Open Multi-Processing, an API specification for parallel programming on shared-memory platforms [https://www.openmp.org/](https://www.openmp.org/). It is widely supported by most modern compilers of C, C++ and Fortran.
2. **OpenCL**: Open Computing Language, an open standard for cross-platform, parallel programming of diverse accelerators ([https://www.khronos.org/opencl/](https://www.khronos.org/opencl/)). The development files for OpenCL are probably provided by the hardware vendor in some kind of development kit, e.g. CUDA from NVIDIA.
3. **CUDA Toolkit**: A development environment (including framework, language, supporting math libraries and plenty of  debugging tools) for creating high performance GPU-accelerated applications ([https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)).
4. **MPI**: Message Passing Interface,  a standardized and portable message-passing interface designed to function on distributed parallel computing architectures. The implementation of [Open MPI](https://www.open-mpi.org/) is suggested to be used in this project.
5. **FreeImage**: an Open Source library for accessing popular image formats like JPEG, PNG, BMP, TIFF, etc. [https://freeimage.sourceforge.io/](https://freeimage.sourceforge.io/)

## Instructions for Use

### Prepare develop tools

1. Install Git (optional) to retrieve the source code from GitHub.
2. Install appropriate C++ compiler (e.g. Clang on macOS, gcc on Linux and MSVC on Windows).
3. Install [CMake](https://cmake.org/) for generating native makefiles and workspaces according to the compiler actually used.
4. Install development kit for OpenCL according to the type of GPU you want to try the parallel programs on (e.g. CUDA for NVIDIA GPU).
5. Install CUDA if your experimental environment includes NVIDIA GPUs.
6. Install appropriate MPI development libraries according to the platforms on which you perform experiments (e.g. Open MPI on Linux, Microsoft MPI on Windows).
7. Install or prepare the FreeImage development libraries.

### Setup local workspace

1. Download the source code of this project.

2. Use CMake to generate appropriate native makefiles and workspaces according to the IDE you use.

3. Build the project and get the executable binaries.

### Perform experiments

The project contains following different implementations of Sobel edge detection algorithm, which can be run on single node or multiple nodes with CPU and GPU.

1. **sobel**: serial version
2. **sobel_omp**: parallel version for CPU implemented by OpenMP
3. **sobel_ocl**: parallel version for GPU implemented by OpenCL
4. **sobel_cuda**: parallel version for GPU implemented by CUDA
5. **sobel_mpi**: distributed parallel version for running on multi-nodes implemented by  MPI
6. **sobel_mpi_omp**: mix-mode distributed parallel version for running on multi-nodes implemented by MPI and OpenMP
7. **sobel_mpi_cuda**: mix-mode distributed parallel version for running on multi-nodes with CPU and GPU processors implemented by MPI and CUDA

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
/* sobel_mpi_omp.cpp - Sobel filter with MPI and OpenMP */
#include <float.h>
#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#include <vector>
#include <iostream>
#include <string>
#include <cstring>

#include "imgutils.h"
#include "sobel.h"

#define MPI_ERROR_CHECK(retCode) do { if (retCode != MPI_SUCCESS) { std::cout << "| MPI ERROR | in " __FILE__ ", line " << __LINE__ << ": "<< retCode << std::endl; exit(-1); } } while (0)

int const MASTER = 0;
int const num_threads = 8;

char gProcessorName[MPI_MAX_PROCESSOR_NAME];

int gImageWidth;
int gImageHeight;
int gImagePitch;
int gImageChNum;

FIBITMAP *gImageIn;
FIBITMAP *gImageOut;

void sobel_filtering(int rank, int np,
    int imageWidthIn, int imageHeightIn, std::vector<BYTE> const &imageDataIn,
    int imageWidthOut, int imageHeightOut, std::vector<BYTE> &imageDataOut)
{
    std::cout << "Filtering of input image on Process " << rank << " of " << gProcessorName << " start ..." << std::endl;

    omp_set_dynamic(false);
    omp_set_num_threads(num_threads);

    double pixel_value_min, pixel_value_max;
    pixel_value_min = DBL_MAX;
    pixel_value_max = DBL_MIN;

    int const pitchIn = imageWidthIn * gImageChNum;
    int actual_threads_num = 0;
#if defined(_MSC_VER)
#pragma omp parallel for
#else
#pragma omp parallel for collapse(2)
#endif
    for (int y = 1; y < imageHeightIn - 1; ++y) {
        for (int x = 1; x < imageWidthIn - 1; ++x) {
            double pixel_value_x = 0.0;
            double pixel_value_y = 0.0;
            double pixel_value = 0.0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    pixel_value_x += sobel_weight_x[j + 1][i + 1] * to_gray(imageDataIn.data(), x + i, y + j, pitchIn, gImageChNum);
                    pixel_value_y += sobel_weight_y[j + 1][i + 1] * to_gray(imageDataIn.data(), x + i, y + j, pitchIn, gImageChNum);
                }
            }
            pixel_value = sqrt((pixel_value_x * pixel_value_x) + (pixel_value_y * pixel_value_y));
            actual_threads_num = omp_get_num_threads();
#pragma omp critical
            {
                if (pixel_value < pixel_value_min) pixel_value_min = pixel_value;
                if (pixel_value > pixel_value_max) pixel_value_max = pixel_value;
            }
        }
    }

    std::cout << "Min/Max on Process " << rank << " of " << gProcessorName << ": " << pixel_value_min << "/" << pixel_value_max << std::endl;
    std::cout << "Actual threads number on Process " << rank << " of " << gProcessorName << ": " << actual_threads_num << std::endl;
    MPI_Status status;

    if (rank != MASTER) {
        // send min & max to MASTER
        std::cout << "Send min & max to MASTER ..." << std::endl;
        MPI_Send(&pixel_value_min, 1, MPI_DOUBLE, MASTER, rank + 100, MPI_COMM_WORLD);
        MPI_Send(&pixel_value_max, 1, MPI_DOUBLE, MASTER, rank + 110, MPI_COMM_WORLD);

        // receive global min & max from MASTER
        std::cout << "Receive global min & max from MASTER ..." << std::endl;
        MPI_Recv(&pixel_value_min, 1, MPI_DOUBLE, MASTER, rank + 200, MPI_COMM_WORLD, &status);
        MPI_Recv(&pixel_value_max, 1, MPI_DOUBLE, MASTER, rank + 210, MPI_COMM_WORLD, &status);
    }

    if (rank == MASTER) {
        // collect the min & max values from other nodes and calculate the global min & max
        double tempMin, tempMax;
        std::cout << "Receive min & max from other nodes ..." << std::endl;
        for (int i = 1; i < np; ++i) {
            MPI_Recv(&tempMin, 1, MPI_DOUBLE, i, i + 100, MPI_COMM_WORLD, &status);
            MPI_Recv(&tempMax, 1, MPI_DOUBLE, i, i + 110, MPI_COMM_WORLD, &status);
            if (tempMin < pixel_value_min) pixel_value_min = tempMin;
            if (tempMax > pixel_value_max) pixel_value_max = tempMax;
        }

        std::cout << "the global minimum value: " << pixel_value_min << std::endl;
        std::cout << "the global maximum value: " << pixel_value_max << std::endl;

        // send global min & max to other nodes
        std::cout << "Send global min & max to other nodes ..." << std::endl;
        for (int i = 1; i < np; ++i) {
            MPI_Send(&pixel_value_min, 1, MPI_DOUBLE, i, i + 200, MPI_COMM_WORLD);
            MPI_Send(&pixel_value_max, 1, MPI_DOUBLE, i, i + 210, MPI_COMM_WORLD);
        }
    }


    if (pixel_value_max - pixel_value_min == 0) {
        std::cout << "Nothing exists on Process " << rank << " of " << gProcessorName << "!" << std::endl;
        return;
    }

    /* Generation of imgOut after linear transformation */
    int const pitchOut = imageWidthOut * gImageChNum;
#if defined(_MSC_VER)
#pragma omp parallel for
#else
#pragma omp parallel for collapse(2)
#endif
    for (int y = 1; y < imageHeightIn - 1; ++y) {
        for (int x = 1; x < imageWidthIn - 1; x++) {
            double pixel_value_x = 0.0;
            double pixel_value_y = 0.0;
            double pixel_value = 0.0;
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

    std::cout << "Filtering of input image on Process " << rank << " of " << gProcessorName << " finish." << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: sobel_mpi_omp input_image_file" << std::endl;
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
            std::cout << "Cannot load image " << argv[1] << std::endl;
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

    // Broadcast the image size to all the nodes
    MPI_Bcast(&gImageChNum, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&gImageWidth, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&gImageHeight, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&gImagePitch, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // MPI_Barrier for Synchronization
    MPI_Barrier(MPI_COMM_WORLD);

    if (gImageChNum == 0) {
        std::cout << "Unsupported image format!" << std::endl;
        MPI_Finalize();
        if (rank == MASTER) {
            FreeImage_Unload(gImageIn);
            FreeImage_Unload(gImageOut);
        }
        return 2;
    }

    int imageDim[2] = { gImageHeight, gImagePitch };
    int scatterTypeDim[2] = { 1, gImageWidth*gImageChNum };
    int scatterTypeOff[2] = { 0, 0 };
    MPI_Datatype scatterType, scatterSubarrayType;
    MPI_Type_create_subarray(2, imageDim, scatterTypeDim, scatterTypeOff, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &scatterType);
    MPI_Type_create_resized(scatterType, 0, gImagePitch, &scatterSubarrayType);
    MPI_Type_commit(&scatterSubarrayType);

    int gatherTypeDim[2] = { 1, (gImageWidth - 2)*gImageChNum };
    int gatherTypeOff[2] = { 0, gImageChNum };
    MPI_Datatype gatherType, gatherSubarrayType;
    MPI_Type_create_subarray(2, imageDim, gatherTypeDim, gatherTypeOff, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &gatherType);
    MPI_Type_create_resized(gatherType, 0, gImagePitch, &gatherSubarrayType);
    MPI_Type_commit(&gatherSubarrayType);

    int gatherImageWidth = gImageWidth - 2;
    int gatherImageHeight = (gImageHeight - 2) / np;
    int gatherImageRestLines = (gImageHeight - 2) - gatherImageHeight*np;

    int scatterImageWidth = gImageWidth;
    int scatterImageHeight = gatherImageHeight + 2;

    // allocate memory of image chunk for each node
    std::vector<BYTE> localImageIn(scatterImageWidth*(scatterImageHeight + 1)*gImageChNum, 0);
    std::vector<BYTE> localImageOut(gatherImageWidth*(gatherImageHeight + 1)*gImageChNum, 0);

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
        localImageIn.data(), scatterImageWidth*sendCounts[rank] * gImageChNum, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    sobel_filtering(rank, np, scatterImageWidth, sendCounts[rank], localImageIn, gatherImageWidth, recvCounts[rank], localImageOut);

    MPI_Gatherv(localImageOut.data(), gatherImageWidth*recvCounts[rank] * gImageChNum, MPI_UNSIGNED_CHAR,
        FreeImage_GetBits(gImageOut), recvCounts.data(), recvDispls.data(), gatherSubarrayType,
        0, MPI_COMM_WORLD);

    if (rank == MASTER) {
        end_time = MPI_Wtime();
    }

    MPI_Type_free(&scatterSubarrayType);
    MPI_Type_free(&gatherSubarrayType);

    int r = 0;
    if (rank == MASTER) {
        std::cout << "The total time for execution is " << end_time - start_time << "s" << std::endl;

        std::string fnIn(argv[1]);
        std::size_t found = fnIn.find_last_of('.');
        std::string fnOut = fnIn.substr(0, found) + "_mpi_omp_out" + fnIn.substr(found);

        if (!save_image(fnOut.c_str(), gImageOut, 0)) {
            std::cout << "Failed to save output image file " << fnOut << std::endl;
            r = -2;
        }

        FreeImage_Unload(gImageIn);
        FreeImage_Unload(gImageOut);
    }

    MPI_Finalize();
    return r;
}

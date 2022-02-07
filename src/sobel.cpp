/* sobel.cpp - Sobel filter sequential program */
#include <float.h>
#include <assert.h>
#include <omp.h>
#include <math.h>

#include <vector>
#include <iostream>
#include <string>
#include <cstring>

#include "imgutils.h"
#include "sobel.h"

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

    std::cout << "Filtering of input image start ..." << std::endl;
    double start_time = omp_get_wtime();

    double pixel_value_min, pixel_value_max;
    pixel_value_min = DBL_MAX;
    pixel_value_max = -DBL_MAX;
    BYTE *dataIn = FreeImage_GetBits(imgIn);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            double pixel_value_x = 0.0;
            double pixel_value_y = 0.0;
            double pixel_value = 0.0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    pixel_value_x += sobel_weight_x[j + 1][i + 1] * to_gray(dataIn, x + i, y + j, pitch, chNum);
                    pixel_value_y += sobel_weight_y[j + 1][i + 1] * to_gray(dataIn, x + i, y + j, pitch, chNum);
                }
            }
            pixel_value = sqrt((pixel_value_x * pixel_value_x) + (pixel_value_y * pixel_value_y));
            if (pixel_value < pixel_value_min) pixel_value_min = pixel_value;
            if (pixel_value > pixel_value_max) pixel_value_max = pixel_value;
        }
    }
    std::cout << "the value of minimum " << pixel_value_min << std::endl;
    std::cout << "the value of maximum " << pixel_value_max << std::endl;
    if ((int)(pixel_value_max - pixel_value_min) == 0) {
        std::cout << "Nothing exists!!!" << std::endl;
        return;
    }

    /* Initialization of imgOut */
    BYTE *dataOut = FreeImage_GetBits(imgOut);
    //for (int y = 0; y < height; ++y) {
    //	for (int x = 0; x < pitch; ++x) {
    //		dataOut[y*pitch+x] = 0;
    //	}
    //}
    memset(dataOut, 0, height*pitch);

    /* Generation of imgOut after linear transformation */
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; x++) {
            double pixel_value_x = 0.0;
            double pixel_value_y = 0.0;
            double pixel_value = 0.0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    pixel_value_x += sobel_weight_x[j + 1][i + 1] * to_gray(dataIn, x + i, y + j, pitch, chNum);
                    pixel_value_y += sobel_weight_y[j + 1][i + 1] * to_gray(dataIn, x + i, y + j, pitch, chNum);
                }
            }
            pixel_value = sqrt((pixel_value_x * pixel_value_x) + (pixel_value_y * pixel_value_y));
            pixel_value = 255.0 * (pixel_value - pixel_value_min) / (pixel_value_max - pixel_value_min);
            set_pixel(dataOut, (BYTE)pixel_value, x, y, pitch, chNum);
        }
    }

    std::cout << "The total time for execution is " << omp_get_wtime() - start_time << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: sobel input_image_file" << std::endl;
        return 0;
    }

    FIBITMAP *imgIn = load_image(argv[1], 0);
    if (!imgIn) {
        return -1;
    }

    FIBITMAP *imgOut = FreeImage_Clone(imgIn);

    sobel_filtering(imgIn, imgOut);

    std::string fnIn(argv[1]);
    std::size_t found = fnIn.find_last_of('.');
    std::string fnOut = fnIn.substr(0, found) + "_out" + fnIn.substr(found);

    int r = 0;
    if (!save_image(fnOut.c_str(), imgOut, 0)) {
        std::cout << "Failed to save output image file " << fnOut << std::endl;
        r = -2;
    }

    FreeImage_Unload(imgIn);
    FreeImage_Unload(imgOut);
    return r;
}

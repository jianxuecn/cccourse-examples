/*batchresize.cpp - generate resized images in batch for parallel program test*/
#include <float.h>
#include <assert.h>
#include <omp.h>
#include <math.h>

#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>

#include "imgutils.h"

inline int str2int(const char *str)
{
    std::istringstream numStr(str);
    int r = 0;
    numStr >> r;
    return r;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cout << "Usage: batchresize input_image_file num_of_images_to_generate [num_of_downsampling]" << std::endl;
        return 0;
    }

    FIBITMAP *imgIn = load_image(argv[1], 0);
    if (!imgIn) {
        std::cout << "Cannot load image from file " << argv[1] << "." << std::endl;
        return -1;
    }

    int imgNum = str2int(argv[2]);
    if (imgNum < 1) {
        std::cout << "Nothing to generate cause num_of_images_to_generate is " << imgNum << "." << std::endl;
        return -2;
    }

    int dwnNum = (argc < 4) ? 0 : str2int(argv[3]);
    if (dwnNum > imgNum) {
        std::cout << "num_of_downsampling is greater than num_of_images!" << std::endl;
        return -3;
    }

    int srcWidth = FreeImage_GetWidth(imgIn);
    int srcHeight = FreeImage_GetHeight(imgIn);

    int r = 0;
    int total = 0;
    for (int i = 0; i < imgNum; ++i) {
        double scaleRatio = (i < dwnNum) ? 1.0 / sqrt(dwnNum + 1 - i) : sqrt(i - dwnNum + 1);
        int dstWidth = static_cast<int>(round(srcWidth*scaleRatio));
        int dstHeight = static_cast<int>(round(srcHeight*scaleRatio));

        FIBITMAP *imgOut = FreeImage_Rescale(imgIn, dstWidth, dstHeight);
        if (!imgOut) {
            std::cout << "Cannot generate image No." << imgNum << " with scale ratio = " << scaleRatio << "." << std::endl;
            r = -4;
            break;
        }

        std::ostringstream iStr;
        iStr << "_" << i + 1;
        std::string fnIn(argv[1]);
        std::size_t found = fnIn.find_last_of('.');
        std::string fnOut = fnIn.substr(0, found) + iStr.str() + fnIn.substr(found);

        int r = 0;
        if (!save_image(fnOut.c_str(), imgOut, 0)) {
            std::cout << "Failed to save output image file " << fnOut << std::endl;
            r = -5;
            break;
        }

        std::cout << "Image " << fnOut << " saved." << std::endl;
        total++;
    }

    std::cout << total << " images generated." << std::endl;

    FreeImage_Unload(imgIn);
    return r;
}

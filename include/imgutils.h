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
#ifndef __imgutils_h
#define __imgutils_h

#include <FreeImage.h>

inline bool is_pow2(unsigned int x)
{
    return ((x & (x - 1)) == 0);
}

inline unsigned int next_pow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

inline FIBITMAP * load_image(char const *filename, int flags)
{
    FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename, 0);

    if (fif == FIF_UNKNOWN) fif = FreeImage_GetFIFFromFilename(filename);
    if (fif == FIF_UNKNOWN) return nullptr;

    FIBITMAP *dib = FreeImage_Load(fif, filename, flags);
    if (!dib) return nullptr;

    BYTE *bits = FreeImage_GetBits(dib);
    unsigned int width = FreeImage_GetWidth(dib);
    unsigned int height = FreeImage_GetHeight(dib);
    if (bits == 0 || width == 0 || height == 0) {
        FreeImage_Unload(dib);
        return nullptr;
    }

    return dib;
}

inline bool save_image(char const *filename, FIBITMAP *dib, int flags)
{
    if (!dib) return false;

    FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(filename);
    return FreeImage_Save(fif, dib, filename, flags);
}

inline double to_gray(BYTE const *data, int x, int y, int pitch, int chNum)
{
    double gray = 0;
    for (int c = 0; c < chNum; ++c) {
        gray += data[y*pitch + x*chNum + c];
    }
    return gray / chNum;
}

inline void set_pixel(BYTE *data, BYTE v, int x, int y, int pitch, int chNum)
{
    for (int c = 0; c < chNum; ++c) {
        data[y*pitch + x*chNum + c] = v;
    }
}

#endif

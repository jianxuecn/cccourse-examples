#ifndef __imgutils_h
#define __imgutils_h

#include <FreeImage.h>

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

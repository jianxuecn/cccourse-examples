#ifndef __sobel_h
#define __sobel_h

/* Definition of Sobel filter in horizontal direction */
int const sobel_weight_x[3][3] = {
    { -1, 0, 1 },
    { -2, 0, 2 },
    { -1, 0, 1 }
};

/* Definition of Sobel filter in vertical direction */
int const sobel_weight_y[3][3] = {
    { -1, -2, -1 },
    {  0,  0,  0 },
    {  1,  2,  1 }
};

#endif

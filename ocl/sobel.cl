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
__kernel void k_sobel4(__global float *dataOut,
                       const __global uchar4 *dataIn,
                       const int imageWidth,
                       const int imageHeight)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);

    float temp = 0.0f;
    float hSum[3] = { 0.0f, 0.0f, 0.0f };
    float vSum[3] = { 0.0f, 0.0f, 0.0f };

    int pixOffset = (y-1) * imageWidth + x-1;

    // NW
    if (x > 0 && y > 0) {
        hSum[0] += (float)dataIn[pixOffset].x;    // horizontal gradient of Red
        hSum[1] += (float)dataIn[pixOffset].y;    // horizontal gradient of Green
        hSum[2] += (float)dataIn[pixOffset].z;    // horizontal gradient of Blue
        vSum[0] -= (float)dataIn[pixOffset].x;    // vertical gradient of Red
        vSum[1] -= (float)dataIn[pixOffset].y;    // vertical gradient of Green
        vSum[2] -= (float)dataIn[pixOffset].z;    // vertical gradient of Blue   
    }
    pixOffset ++;

    // N
    if (y > 0) {
        vSum[0] -= (float)(dataIn[pixOffset].x << 1);  // vertical gradient of Red
        vSum[1] -= (float)(dataIn[pixOffset].y << 1);  // vertical gradient of Green
        vSum[2] -= (float)(dataIn[pixOffset].z << 1);  // vertical gradient of Blue
    }
    pixOffset ++;

    // NE
    if (y > 0 && x < imageWidth-1) {
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
    pixOffset ++;

    // C
    pixOffset ++;

    // E
    if (x < imageWidth-1) {
        hSum[0] -= (float)(dataIn[pixOffset].x << 1);  // vertical gradient of Red
        hSum[1] -= (float)(dataIn[pixOffset].y << 1);  // vertical gradient of Green
        hSum[2] -= (float)(dataIn[pixOffset].z << 1);  // vertical gradient of Blue
    }
    pixOffset += imageWidth - 2;

    // SW
    if (x > 0 && y < imageHeight-1) {
        hSum[0] += (float)dataIn[pixOffset].x;    // horizontal gradient of Red
        hSum[1] += (float)dataIn[pixOffset].y;    // horizontal gradient of Green
        hSum[2] += (float)dataIn[pixOffset].z;    // horizontal gradient of Blue
        vSum[0] += (float)dataIn[pixOffset].x;    // vertical gradient of Red
        vSum[1] += (float)dataIn[pixOffset].y;    // vertical gradient of Green
        vSum[2] += (float)dataIn[pixOffset].z;    // vertical gradient of Blue
    }
    pixOffset ++;

    // S
    if (y < imageHeight-1) {
        vSum[0] += (float)(dataIn[pixOffset].x << 1);  // vertical gradient of Red
        vSum[1] += (float)(dataIn[pixOffset].y << 1);  // vertical gradient of Green
        vSum[2] += (float)(dataIn[pixOffset].z << 1);  // vertical gradient of Blue
    }
    pixOffset ++;

    // SE
    if (x < imageWidth-1 && y < imageHeight-1) {
        hSum[0] -= (float)dataIn[pixOffset].x;    // horizontal gradient of Red
        hSum[1] -= (float)dataIn[pixOffset].y;    // horizontal gradient of Green
        hSum[2] -= (float)dataIn[pixOffset].z;    // horizontal gradient of Blue
        vSum[0] += (float)dataIn[pixOffset].x;    // vertical gradient of Red
        vSum[1] += (float)dataIn[pixOffset].y;    // vertical gradient of Green
        vSum[2] += (float)dataIn[pixOffset].z;    // vertical gradient of Blue
    }

    // Weighted combination of Root-Sum-Square per-color-band H & V gradients for each of RGB
    temp =  sqrt((hSum[0] * hSum[0]) + (vSum[0] * vSum[0])) / 3.0f;
    temp += sqrt((hSum[1] * hSum[1]) + (vSum[1] * vSum[1])) / 3.0f;
    temp += sqrt((hSum[2] * hSum[2]) + (vSum[2] * vSum[2])) / 3.0f;

    int outIdx = y * imageWidth + x;
    dataOut[outIdx] = temp;
}

__kernel void k_sobel3(__global float *dataOut,
                       const __global uchar *dataIn,
                       const int imageWidth,
                       const int imageHeight,
                       const int pitch)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);

    float temp = 0.0f;
    float hSum[3] = { 0.0f, 0.0f, 0.0f };
    float vSum[3] = { 0.0f, 0.0f, 0.0f };

    int pixOffset = (y-1) * pitch + (x-1)*3;

    // NW
    if (x > 0 && y > 0) {
        hSum[0] += (float)dataIn[pixOffset+0];    // horizontal gradient of Red
        hSum[1] += (float)dataIn[pixOffset+1];    // horizontal gradient of Green
        hSum[2] += (float)dataIn[pixOffset+2];    // horizontal gradient of Blue
        vSum[0] -= (float)dataIn[pixOffset+0];    // vertical gradient of Red
        vSum[1] -= (float)dataIn[pixOffset+1];    // vertical gradient of Green
        vSum[2] -= (float)dataIn[pixOffset+2];    // vertical gradient of Blue   
    }
    pixOffset += 3;

    // N
    if (y > 0) {
        vSum[0] -= (float)(dataIn[pixOffset+0] << 1);  // vertical gradient of Red
        vSum[1] -= (float)(dataIn[pixOffset+1] << 1);  // vertical gradient of Green
        vSum[2] -= (float)(dataIn[pixOffset+2] << 1);  // vertical gradient of Blue
    }
    pixOffset +=3;

    // NE
    if (y > 0 && x < imageWidth-1) {
        hSum[0] -= (float)dataIn[pixOffset+0];    // horizontal gradient of Red
        hSum[1] -= (float)dataIn[pixOffset+1];    // horizontal gradient of Green
        hSum[2] -= (float)dataIn[pixOffset+2];    // horizontal gradient of Blue
        vSum[0] -= (float)dataIn[pixOffset+0];    // vertical gradient of Red
        vSum[1] -= (float)dataIn[pixOffset+1];    // vertical gradient of Green
        vSum[2] -= (float)dataIn[pixOffset+2];    // vertical gradient of Blue
    }
    pixOffset += pitch - 6;

    // W
    if (x > 0) {
        hSum[0] += (float)(dataIn[pixOffset+0] << 1);  // vertical gradient of Red
        hSum[1] += (float)(dataIn[pixOffset+1] << 1);  // vertical gradient of Green
        hSum[2] += (float)(dataIn[pixOffset+2] << 1);  // vertical gradient of Blue
    }
    pixOffset += 3;

    // C
    pixOffset += 3;

    // E
    if (x < imageWidth-1) {
        hSum[0] -= (float)(dataIn[pixOffset+0] << 1);  // vertical gradient of Red
        hSum[1] -= (float)(dataIn[pixOffset+1] << 1);  // vertical gradient of Green
        hSum[2] -= (float)(dataIn[pixOffset+2] << 1);  // vertical gradient of Blue
    }
    pixOffset += pitch - 6;

    // SW
    if (x > 0 && y < imageHeight-1) {
        hSum[0] += (float)dataIn[pixOffset+0];    // horizontal gradient of Red
        hSum[1] += (float)dataIn[pixOffset+1];    // horizontal gradient of Green
        hSum[2] += (float)dataIn[pixOffset+2];    // horizontal gradient of Blue
        vSum[0] += (float)dataIn[pixOffset+0];    // vertical gradient of Red
        vSum[1] += (float)dataIn[pixOffset+1];    // vertical gradient of Green
        vSum[2] += (float)dataIn[pixOffset+2];    // vertical gradient of Blue
    }
    pixOffset += 3;

    // S
    if (y < imageHeight-1) {
        vSum[0] += (float)(dataIn[pixOffset+0] << 1);  // vertical gradient of Red
        vSum[1] += (float)(dataIn[pixOffset+1] << 1);  // vertical gradient of Green
        vSum[2] += (float)(dataIn[pixOffset+2] << 1);  // vertical gradient of Blue
    }
    pixOffset += 3;

    // SE
    if (x < imageWidth-1 && y < imageHeight-1) {
        hSum[0] -= (float)dataIn[pixOffset+0];    // horizontal gradient of Red
        hSum[1] -= (float)dataIn[pixOffset+1];    // horizontal gradient of Green
        hSum[2] -= (float)dataIn[pixOffset+2];    // horizontal gradient of Blue
        vSum[0] += (float)dataIn[pixOffset+0];    // vertical gradient of Red
        vSum[1] += (float)dataIn[pixOffset+1];    // vertical gradient of Green
        vSum[2] += (float)dataIn[pixOffset+2];    // vertical gradient of Blue
    }

    // Weighted combination of Root-Sum-Square per-color-band H & V gradients for each of RGB
    temp =  sqrt((hSum[0] * hSum[0]) + (vSum[0] * vSum[0])) / 3.0f;
    temp += sqrt((hSum[1] * hSum[1]) + (vSum[1] * vSum[1])) / 3.0f;
    temp += sqrt((hSum[2] * hSum[2]) + (vSum[2] * vSum[2])) / 3.0f;

    int outIdx = y * imageWidth + x;
    dataOut[outIdx] = temp;
}

__kernel void k_sobel1(__global float *dataOut,
                       const __global uchar *dataIn,
                       const int imageWidth,
                       const int imageHeight,
                       const int pitch)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);

    float temp = 0.0f;
    float hSum = 0.0f;
    float vSum = 0.0f;

    int pixOffset = (y-1) * pitch + x-1;

    // NW
    if (x > 0 && y > 0) {
        hSum += (float)dataIn[pixOffset];    // horizontal gradient of Red
        vSum -= (float)dataIn[pixOffset];    // vertical gradient of Red
    }
    pixOffset ++;

    // N
    if (y > 0) {
        vSum -= (float)(dataIn[pixOffset] << 1);  // vertical gradient of Red
    }
    pixOffset ++;

    // NE
    if (y > 0 && x < imageWidth-1) {
        hSum -= (float)dataIn[pixOffset];    // horizontal gradient of Red
        vSum -= (float)dataIn[pixOffset];    // vertical gradient of Red
    }
    pixOffset += pitch - 2;

    // W
    if (x > 0) {
        hSum += (float)(dataIn[pixOffset] << 1);  // vertical gradient of Red
    }
    pixOffset ++;

    // C
    pixOffset ++;

    // E
    if (x < imageWidth-1) {
        hSum -= (float)(dataIn[pixOffset] << 1);  // vertical gradient of Red
    }
    pixOffset += pitch - 2;

    // SW
    if (x > 0 && y < imageHeight-1) {
        hSum += (float)dataIn[pixOffset];    // horizontal gradient of Red
        vSum += (float)dataIn[pixOffset];    // vertical gradient of Red
    }
    pixOffset ++;

    // S
    if (y < imageHeight-1) {
        vSum += (float)(dataIn[pixOffset] << 1);  // vertical gradient of Red
    }
    pixOffset ++;

    // SE
    if (x < imageWidth-1 && y < imageHeight-1) {
        hSum -= (float)dataIn[pixOffset];    // horizontal gradient of Red
        vSum += (float)dataIn[pixOffset];    // vertical gradient of Red
    }

    // Weighted combination of Root-Sum-Square H & V gradients
    temp =  sqrt((hSum * hSum) + (vSum * vSum));

    int outIdx = y * imageWidth + x;
    dataOut[outIdx] = temp;
}

__kernel void k_min_max(__global float *minOut,
                        __global float *maxOut,
                        const __global float *dataIn,
                        const unsigned int n,
                        __local float *minShared,
                        __local float *maxShared)
{
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);

    minShared[tid] = (i < n) ? dataIn[i] : FLT_MAX;
    maxShared[tid] = (i < n) ? dataIn[i] : FLT_MIN;

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared memory
    for (unsigned int s=get_local_size(0)/2; s>0; s>>=1) {
        if (tid < s) {
            if (minShared[tid+s] < minShared[tid]) minShared[tid] = minShared[tid+s];
            if (maxShared[tid+s] > maxShared[tid]) maxShared[tid] = maxShared[tid+s];            
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        unsigned int outIdx = get_group_id(0);
        minOut[outIdx] = minShared[0];
        maxOut[outIdx] = maxShared[0];
    }
}

__kernel void k_min_max_iter(__global float *minInOut,
                             __global float *maxInOut,
                             const unsigned int n,
                             __local float *minShared,
                             __local float *maxShared)
{
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);

    minShared[tid] = (i < n) ? minInOut[i] : FLT_MAX;
    maxShared[tid] = (i < n) ? maxInOut[i] : FLT_MIN;

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared memory
    for (unsigned int s=get_local_size(0)/2; s>0; s>>=1) {
        if (tid < s) {
            if (minShared[tid+s] < minShared[tid]) minShared[tid] = minShared[tid+s];
            if (maxShared[tid+s] > maxShared[tid]) maxShared[tid] = maxShared[tid+s];            
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        unsigned int outIdx = get_group_id(0);
        minInOut[outIdx] = minShared[0];
        maxInOut[outIdx] = maxShared[0];
    }
}

__kernel void k_min(__global float *minOut,
                    __global float *dataIn,
                    const unsigned int n,
                    __local float *minShared)
{
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);

    minShared[tid] = (i < n) ? dataIn[i] : FLT_MAX;

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared memory
    for (unsigned int s=get_local_size(0)/2; s>0; s>>=1) {
        if (tid < s) {
            if (minShared[tid+s] < minShared[tid]) minShared[tid] = minShared[tid+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        unsigned int outIdx = get_group_id(0);
        minOut[outIdx] = minShared[0];
    }
}

__kernel void k_max(__global float *maxOut,
                    __global float *dataIn,
                    const unsigned int n,
                    __local float *maxShared)
{
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);

    maxShared[tid] = (i < n) ? dataIn[i] : FLT_MIN;

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared memory
    for (unsigned int s=get_local_size(0)/2; s>0; s>>=1) {
        if (tid < s) {
            if (maxShared[tid+s] > maxShared[tid]) maxShared[tid] = maxShared[tid+s];            
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        unsigned int outIdx = get_group_id(0);
        maxOut[outIdx] = maxShared[0];
    }
}

__kernel void k_scale_pixels4(__global uchar4 *dataOut,
                              const __global float *dataIn,
                              const float dataInMin,
                              const float dataInMax,
                              const unsigned int n)
{
    unsigned int i = get_global_id(0);
    
    if (i < n) {
        float temp = dataIn[i];
        temp = 255.0f * (temp - dataInMin) / (dataInMax - dataInMin);
        dataOut[i] = (uchar4)temp;
        dataOut[i].w = 255;
    }
}

__kernel void k_scale_pixels3(__global uchar *dataOut,
                              const __global float *dataIn,
                              const float dataInMin,
                              const float dataInMax,
                              const int imageWidth,
                              const int imageHeight,
                              const int pitch)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    int posIn = y*imageWidth + x;
    int posOut = y*pitch + x*3;
    float temp = dataIn[posIn];
    temp = 255.0f * (temp - dataInMin) / (dataInMax - dataInMin);
    dataOut[posOut+0] = (uchar)temp;
    dataOut[posOut+1] = (uchar)temp;
    dataOut[posOut+2] = (uchar)temp;
}

__kernel void k_scale_pixels1(__global uchar *dataOut,
                              const __global float *dataIn,
                              const float dataInMin,
                              const float dataInMax,
                              const int imageWidth,
                              const int imageHeight,
                              const int pitch)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    int posIn = y*imageWidth + x;
    int posOut = y*pitch + x;
    float temp = dataIn[posIn];
    temp = 255.0f * (temp - dataInMin) / (dataInMax - dataInMin);
    dataOut[posOut] = (uchar)temp;
}

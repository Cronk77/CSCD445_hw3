#include <stdio.h>
#include "kernel1.h"


//extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
    extern __shared__ float s_data[];

    // TODO, implement this kernel below
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    i = i + 1;
    
    // global thread(data) column index
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    j = j + 1;

    if( i >= width - 1|| j >= width - 1 || i < 1 || j < 1 ) return;

    s_data[blockDim.y * threadIdx.x + (threadIdx.y + 1)] = g_dataA[i * floatpitch + j]; //middle
    s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y + 1)] = g_dataA[(i + 1) * floatpitch + j]; //south
    s_data[blockDim.y * (threadIdx.x - 1) + (threadIdx.y + 1)] = g_dataA[(i - 1) * floatpitch + j]; //north

    if(threadIdx.y == blockDim.y - 1)
    {
        s_data[blockDim.y * threadIdx.x + (threadIdx.y + 2)] = g_dataA[ i * pitch + (j + 1)]; //E
        s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y + 2)] = g_dataA[ (i + 1) * pitch + (j + 1)]; //SE
        s_data[blockDim.y * (threadIdx.x - 1) + (threadIdx.y + 2)] = g_dataA[ (i - 1) * pitch + (j + 1)]; //NE

    }else if(threadIdx.y == 1)
    {
        s_data[blockDim.y * threadIdx.x + threadIdx.y] = g_dataA[ i * pitch + (j - 1)]; //W
        s_data[blockDim.y * (threadIdx.x + 1) + threadIdx.y] = g_dataA[ (i + 1) * pitch + (j - 1)]; //SW
        s_data[blockDim.y * (threadIdx.x - 1) + threadIdx.y] = g_dataA[ (i - 1) * pitch + (j - 1)]; //NW
    }

    __syncthreads();

    float middle = s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y + 1)];
    float south = s_data[blockDim.y * (threadIdx.x + 2) + (threadIdx.y + 1)];
    float north = s_data[blockDim.y * threadIdx.x + (threadIdx.y + 1)];

    int colNum = threadIdx.y % 2; // 3 is the number of columns per jacobi value and we view two of them at first so we only look at each one twice

    float jacobiValue = 1.0;

    //if we are at the first value then we grab the previous value as well otherwise we add up all the values and multiply by .95 to get the jacobi value
    if(colNum == 0)
    {
        float prevMiddle = s_data[blockDim.y * (threadIdx.x + 1) + threadIdx.y];
        float prevSouth = s_data[blockDim.y * (threadIdx.x + 2) + threadIdx.y];
        float prevNorth = s_data[blockDim.y * threadIdx.x + threadIdx.y];

        s_data[blockDim.y * (threadIdx.x + 1) + threadIdx.y + 1] = runJacobi(1, north, middle, south); // isMiddle is 1 because we are at the middle number
        s_data[blockDim.y * (threadIdx.x + 1) + threadIdx.y] = runJacobi(0, prevNorth, prevMiddle, prevSouth); // isMiddle is 0 because we are not at the middle numbe

    }else
    {
        jacobiValue = runJacobi(0, north, middle, south);
        jacobiValue += s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y - 1)]
        jacobiValue += s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y - 2)]
        jacobiValue = jacobiValue * 0.95f;
    }

    __syncthreads();

    //if we have a correct jacobiValue, store it at the middle location
    if(colNum == 1)
    {
        s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y - 1)] = jacobiValue;
    }

    /move data to output array
    g_dataB[blockDim.y * threadIdx.x + threadIdx.y] = s_data[blockDim.y * threadIdx.x + threadIdx.y];
}

__device__ float runJacobi(int isMiddle, float northValue, float middleValue, float southValue)
{
    if(!isMiddle)
    {
        return 0.1f * northValue + 0.1f * middleValue + 0.1f * southValue;

    }else
    {
        return 0.1f * northValue + 0.2f * middleValue + 0.1f * southValue;
    }

}


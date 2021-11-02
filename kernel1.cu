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
    
    // global thread(data) column index
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if( i >= width - 1|| j >= width - 1 || i < 1 || j < 1 ) return;

   // int colNum = j % 3; // 3 is the number of rows we are looking at each time
    
    int northValue = g_dataA[(i-1) * pitch +  j];
    int middleValue = g_dataA[i * pitch + j];
    int southValue = g_dataA[(i+1) * pitch +  j];

    s_data[(i - 1) * pitch + j] = northValue;
    s_data[i * pitch + j] = middleValue;
    s_data[(i + 1) * pitch + j] = southValue;

    if(threadIdx.x == blockDim.x-1 && i + 2 < width && ((i + 3) * pitch + j) < width)
    {
        s_data[(i) * pitch + j] = g_dataA[(i) * pitch +  j];
        s_data[(i + 1) * pitch + j] = g_dataA[(i + 1) * pitch +  j];
        s_data[(i + 2) * pitch + j] = g_dataA[(i + 2) * pitch +  j];

        s_data[(i + 1) * pitch + j] = g_dataA[(i + 1) * pitch +  j];
        s_data[(i + 2) * pitch + j] = g_dataA[(i + 2) * pitch +  j];
        s_data[(i + 3) * pitch + j] = g_dataA[(i + 3) * pitch +  j];
    }

    _syncthreads();
    //float jacobiValue = runJacobi(colNum, northValue, middleValue, southValue);

    //s_data[0] += jacobiValue;
}

__device__ float runJacobi(int colNum, int northValue, int middleValue, int southValue)
{


}


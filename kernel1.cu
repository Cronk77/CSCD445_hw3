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
    
    //get the top, middle and bottom value at the given i and j location
    int northValue = g_dataA[(i-1) * floatpitch +  j];
    int middleValue = g_dataA[i * floatpitch + j];
    int southValue = g_dataA[(i+1) * floatpitch +  j];

    //save the values from the columns we created into shared memory
    s_data[(i - 1) * floatpitch + j] = northValue;
    s_data[i * floatpitch + j] = middleValue;
    s_data[(i + 1) * floatpitch + j] = southValue;

    //check for out of bounds and grab the 2 extra columns we need an will miss in our current block
    if(threadIdx.x == blockDim.x-1 && i + 2 < width && ((i + 3) * floatpitch + j) < width)
    {
        s_data[(i) * floatpitch + j] = g_dataA[(i) * floatpitch +  j];
        s_data[(i + 1) * floatpitch + j] = g_dataA[(i + 1) * floatpitch +  j];
        s_data[(i + 2) * floatpitch + j] = g_dataA[(i + 2) * floatpitch +  j];

        s_data[(i + 1) * floatpitch + j] = g_dataA[(i + 1) * floatpitch +  j];
        s_data[(i + 2) * floatpitch + j] = g_dataA[(i + 2) * floatpitch +  j];
        s_data[(i + 3) * floatpitch + j] = g_dataA[(i + 3) * floatpitch +  j];
    }

    __syncthreads();
    //float jacobiValue = runJacobi(colNum, northValue, middleValue, southValue);

    //s_data[0] += jacobiValue;
}

__device__ float runJacobi(int colNum, int northValue, int middleValue, int southValue)
{


}


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

    if(i >= width - 1|| j >= width - 1 || i < 1 || j < 1 ) return;

    printf("3");
    if(threadIdx.y != 0)
    {
        s_data[blockDim.y * threadIdx.x + threadIdx.y] = g_dataA[i * floatpitch + j]; //middle
        s_data[blockDim.y * (threadIdx.x + 1) + threadIdx.y] = g_dataA[(i + 1) * floatpitch + j]; //south
        s_data[blockDim.y * (threadIdx.x - 1) + threadIdx.y] = g_dataA[(i - 1) * floatpitch + j]; //north
        
        if(threadIdx.y == blockDim.y - 1 || i == width - 2)
        {
            s_data[blockDim.y * threadIdx.x + (threadIdx.y + 1)] = g_dataA[ i * floatpitch + (j + 1)]; //E
            s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y + 1)] = g_dataA[ (i + 1) * floatpitch + (j + 1)]; //SE
            s_data[blockDim.y * (threadIdx.x - 1) + (threadIdx.y + 1)] = g_dataA[ (i - 1) * floatpitch + (j + 1)]; //NE
            printf("0");

        }else if(threadIdx.y == 1)
        {
            s_data[blockDim.y * threadIdx.x + (threadIdx.y - 1)] = g_dataA[ i * floatpitch + (j - 1)]; //W
            s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y - 1)] = g_dataA[ (i + 1) * floatpitch + (j - 1)]; //SW
            s_data[blockDim.y * (threadIdx.x - 1) + (threadIdx.y - 1)] = g_dataA[ (i - 1) * floatpitch + (j - 1)]; //NW
            printf("1");
        }
    }

    __syncthreads();

    if(threadIdx.y > 0 && threadIdx.y < width - 1 && threadIdx.x > 0 && threadIdx.x < width - 1)
    {
        g_dataB[i * floatpitch + j] = (
                              0.2f * s_data[blockDim.y * threadIdx.x + threadIdx.y]              +       //itself
                              0.1f * s_data[blockDim.y * (threadIdx.x - 1) + threadIdx.y]       +       //N
                              0.1f * s_data[blockDim.y * (threadIdx.x - 1) + (threadIdx.y + 1)] +       //NE
                              0.1f * s_data[blockDim.y * threadIdx.x + (threadIdx.y + 1)]       +       //E
                              0.1f * s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y + 1)] +       //SE
                              0.1f * s_data[blockDim.y * (threadIdx.x + 1) + threadIdx.y]       +       //S
                              0.1f * s_data[blockDim.y * (threadIdx.x + 1) + (threadIdx.y - 1)] +       //SW
                              0.1f * s_data[blockDim.y * threadIdx.x + (threadIdx.y - 1)]       +       //W
                              0.1f * s_data[blockDim.y * (threadIdx.x - 1) + (threadIdx.y - 1)]         //NW
                           ) * 0.95f;
    }
}


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

    if(i >= width - 1 || j >= width - 1 || i < 1 || j < 1 ) return;

    //if we are in the first column of blocks
    /*if(blockIdx.x == 0){

        if(threadIdx.x == 1)
        {
            //grab our current location values and store at location 1 in shared memory
            s_data[threadIdx.x + blockDim.x] = g_dataA[ i * floatpitch + j]; //middle
            s_data[threadIdx.x + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + j]; //S
            s_data[threadIdx.x] = g_dataA[ (i - 1) * floatpitch + j]; //N

            printf("first: %f %f %f\n\n", s_data[threadIdx.x], s_data[threadIdx.x + blockDim.x], s_data[threadIdx.x + (2 * blockDim.x)]);

            //grab our previous location values and store at previous location in shared memory
            s_data[(threadIdx.x - 1) + blockDim.x] = g_dataA[ i * floatpitch + (j - 2)]; //W
            s_data[(threadIdx.x - 1) + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + (j - 2)]; //SW
            s_data[(threadIdx.x - 1)] = g_dataA[ (i - 1) * floatpitch + (j - 2)]; //NW

            printf("second: %f %f %f\n", s_data[threadIdx.x -1], s_data[(threadIdx.x - 1) + blockDim.x], s_data[(threadIdx.x - 1)+ (2 * blockDim.x)]);
        } else if(threadIdx.x == blockDim.x - 1)
        {
            //grab our current values at the end of the block and store at the current location in memory
            s_data[threadIdx.x + blockDim.x] = g_dataA[ i * floatpitch + j]; //middle
            s_data[threadIdx.x + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + j]; //S
            s_data[threadIdx.x] = g_dataA[ (i - 1) * floatpitch + j]; //N

            //grab our next location values and store at the next location of shared memory
            s_data[(threadIdx.x + 1) + blockDim.x] = g_dataA[ i * floatpitch + (j + 1)]; //E
            s_data[(threadIdx.x + 1) + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + (j + 1)]; //SE
            s_data[(threadIdx.x + 1)] = g_dataA[ (i - 1) * floatpitch + (j + 1)]; //NE
        }else
        {
            //if we are not at the start or at the end then we just store oue current location values at the current location in shared memory
            s_data[threadIdx.x + blockDim.x] = g_dataA[ i * floatpitch + j]; //middle
            s_data[threadIdx.x + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + j]; //S
            s_data[threadIdx.x] = g_dataA[ (i - 1) * floatpitch + j]; //N
        }
    }else
    { */
        if(threadIdx.x == 0)
        {
            //grab current value in the global memory and store at next value in shared memory because it should be at 1
            s_data[(threadIdx.x + 1) + blockDim.x] = g_dataA[ i * floatpitch + j]; //middle
            s_data[(threadIdx.x + 1) + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + j]; //S
            s_data[threadIdx.x + 1] = g_dataA[ (i - 1) * floatpitch + j]; //N

            //grab our previous location values and store at current location in shared memory because it should be at 0
            s_data[threadIdx.x + blockDim.x] = g_dataA[ i * floatpitch + (j - 1)]; //W
            s_data[threadIdx.x + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + (j - 1)]; //SW
            s_data[threadIdx.x] = g_dataA[ (i - 1) * floatpitch + (j - 1)]; //NW
        }else if(i == width - 2 || threadIdx.x == width )
        {
            //grab current value in the global memory and store at next value in shared memory because it should be at 1
            s_data[(threadIdx.x + 1) + blockDim.x] = g_dataA[ i * floatpitch + j]; //middle
            s_data[(threadIdx.x + 1) + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + j]; //S
            s_data[threadIdx.x + 1] = g_dataA[ (i - 1) * floatpitch + j]; //N

            //grab our next location values and store at the +2 location of shared memory
            s_data[(threadIdx.x + 2) + blockDim.x] = g_dataA[ i * floatpitch + (j + 1)]; //E
            s_data[(threadIdx.x + 2) + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + (j + 1)]; //SE
            s_data[(threadIdx.x + 2)] = g_dataA[ (i - 1) * floatpitch + (j + 1)]; //NE

        }else
        {
            //grab current value in the global memory and store at next value in shared memory because it should be at 1
            s_data[(threadIdx.x + 1) + blockDim.x] = g_dataA[ i * floatpitch + j]; //middle
            s_data[(threadIdx.x + 1) + (2 * blockDim.x)] = g_dataA[ (i + 1) * floatpitch + j]; //S
            s_data[threadIdx.x + 1] = g_dataA[ (i - 1) * floatpitch + j]; //N
        }
   // }

    __syncthreads();

    if(threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) return;

    //if(blockIdx.x == 0 && threadIdx.x == 0){
    //    printf("%f\n%f\n%f\n\n", s_data[threadIdx.x], s_data[threadIdx.x + blockDim.x], s_data[threadIdx.x + (2 * blockDim.x)]);
    // }

    g_dataB[i * floatpitch + j] = (
                            0.2f * s_data[threadIdx.x + blockDim.x]             +       //itself
                            0.1f * s_data[threadIdx.x]                          +       //N
                            0.1f * s_data[threadIdx.x + 1]                      +       //NE
                            0.1f * s_data[(threadIdx.x + 1) + blockDim.x]       +       //E
                            0.1f * s_data[(threadIdx.x + 1) + (2 * blockDim.x)] +       //SE
                            0.1f * s_data[threadIdx.x + (2 * blockDim.x)]       +       //S
                            0.1f * s_data[(threadIdx.x - 1) + (2 * blockDim.x)] +       //SW
                            0.1f * s_data[(threadIdx.x - 1) + blockDim.x]       +       //W
                            0.1f * s_data[threadIdx.x - 1]                              //NW
                        ) * 0.95f;
}


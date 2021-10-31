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
    
}

__device__ float runJacobi(int currRow, int currCol, float* g_dataA, float* g_dataB, int pitch)
{

      g_dataB[currRow * pitch + currCol] = (
                              0.2f * g_dataA[currRow * pitch + currCol]             +   //itself
                              0.1f * g_dataA[(currRow - 1) * pitch +  currCol   ]   +   //N
                              0.1f * g_dataA[(currRow - 1) * pitch + (currCol + 1)] +   //NE
                              0.1f * g_dataA[ currRow    * pitch + (currCol + 1)]   +   //E
                              0.1f * g_dataA[(currRow + 1) * pitch + (currCol + 1)] +   //SE
                              0.1f * g_dataA[(currRow + 1) * pitch +  currCol   ]   +   //S
                              0.1f * g_dataA[(currRow + 1) * pitch + (currCol - 1)] +   //SW
                              0.1f * g_dataA[ currRow    * pitch + (currCol - 1)]   +   //W
                              0.1f * g_dataA[(currRow - 1) * pitch + (currCol - 1)]     //NW
                           ) * 0.95f;
}


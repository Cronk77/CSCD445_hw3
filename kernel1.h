#ifndef KERNEL1_H
#define KERNEL1_H


__global__ void k1( float* g_dataA, float* g_dataB, int dataWidth, int width);

__device__ float runJacobi(int currRow, int currCol, float* g_dataA, float* g_dataB, int pitch);


#endif

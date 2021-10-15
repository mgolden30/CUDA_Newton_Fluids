#ifndef FOURIER_FILTER
#define FOURIER_FILTER


#include <cufft.h>
#include <cuda_runtime.h>
#include "common.h"

#define NX   N
#define NK   ( (NX)/2 + 1  )    //Points in k. Starts at 0, goes to NK/2 due to symmetry. For Fourier transform.
#define GP     ( (NX)*(NX) )      //Overall number of spatial gridpoints
#define GPK    ( (NK)*(NX) )      //Overall number of points in Fourier space

#define BEGIN_TWOD_FOURIER \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;\
        while( idx < GPK )\
        {\
            int k = idx / NK ; \
            int l = idx % NK ;
#define END_TWOD_FOURIER  idx += TOTAL_THREADS;}
#define IDXK(k,l)    (k)*(NK)+(l)
#define ACTUAL_K\
                  double actual_k;\
                  if(k<NK)\
                  {\
                      actual_k = k;\
                  }\
                  else\
                  {\
                      actual_k = k-NX;\
                  }





void fourier_filter_state( double *state_d );
__global__ void filter( cufftDoubleComplex *fft_data_d );


#endif

#include "fourier_filter.h"

void fourier_filter_state( double *state_d ){
  /*
  A simple Fourier filter
  */

  cufftDoubleComplex *fft_data_d;
  cudaMalloc( &fft_data_d, sizeof(cufftDoubleComplex)*N*(N/2+1) );
  
  double *u_d = &state_d[    0];
  double *v_d = &state_d[  N*N];
  double *w_d = &state_d[2*N*N];

  cufftHandle plan_D2Z;
  cufftHandle plan_Z2D;
  cufftPlan2d( &plan_D2Z, N, N, CUFFT_D2Z );
  cufftPlan2d( &plan_Z2D, N, N, CUFFT_Z2D );
  
  cufftExecD2Z( plan_D2Z, u_d, fft_data_d );
  filter<<<BLOCKS,THREADS>>>(fft_data_d);
  cufftExecZ2D( plan_Z2D, fft_data_d, u_d );

  cufftExecD2Z( plan_D2Z, v_d, fft_data_d );
  filter<<<BLOCKS,THREADS>>>(fft_data_d);
  cufftExecZ2D( plan_Z2D, fft_data_d, v_d );

  cufftExecD2Z( plan_D2Z, w_d, fft_data_d );
  filter<<<BLOCKS,THREADS>>>(fft_data_d);
  cufftExecZ2D( plan_Z2D, fft_data_d, w_d );

  cufftDestroy(plan_D2Z);
  cufftDestroy(plan_Z2D);
  cudaFree( fft_data_d );
}

__global__ void filter( cufftDoubleComplex *fft_data_d ){
    BEGIN_TWOD_FOURIER
    ACTUAL_K
        double magnitude = actual_k * actual_k + l * l;
        
        //Cutoff is defined in common.h
        if( magnitude > FOURIER_CUTOFF + 0.001 ){
          fft_data_d[IDXK(k,l)].x = 0.;
          fft_data_d[IDXK(k,l)].y = 0.;
	}
        else{
          //Otherwise divide by gridpoints!
	  fft_data_d[IDXK(k,l)].x = fft_data_d[IDXK(k,l)].x/N/N;
          fft_data_d[IDXK(k,l)].y = fft_data_d[IDXK(k,l)].y/N/N;
	}

    END_TWOD_FOURIER
}

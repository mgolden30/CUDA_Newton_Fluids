/*
PURPOSE:
To find stationary solutions to the Navier-Stokes equations in 2D. 
While all of us grad students have some MATLAB implementations of such a code, they are very computationally intensive. 
The purpose of this code specifically is to rely on CUDA to speed things up.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

//My own code
#include "objective_function.h"
#include "common.h"
#include "IO.h"
#include "fourier_filter.h"
#include "finite_difference_macros.h"

//CUDA KERNELS
__global__ void equilibrium_objective_kernel   (double *output, double *state, void *params);
__global__ void equilibrium_action_of_J_kernel (double *output, double *input, double *state, void *params);
__global__ void equilibrium_action_of_Jt_kernel(double *output, double *input, double *state, void *params);
__global__ void initial_data( double *state );


int main( int argc, char **argv){
  //INITIALIZE OBJECTIVE FUNCTION
  objective_function ns_equilibria;
  ns_equilibria.input_dim      = 3*N*N;
  ns_equilibria.output_dim     = 4*N*N; 
  ns_equilibria.objective_func = (equilibrium_objective_kernel);
  ns_equilibria.action_of_J    = (equilibrium_action_of_J_kernel);
  ns_equilibria.action_of_Jt   = (equilibrium_action_of_Jt_kernel);

  //INITIALIZE INITIAL GUESS
  double *state_d;
  double *F_d;
  cudaMalloc( &state_d, (ns_equilibria.input_dim) *sizeof(double) );
  cudaMalloc( &F_d,     (ns_equilibria.output_dim)*sizeof(double) );
  initial_data<<<BLOCKS,THREADS>>>( state_d );

  //Newton parameters
  int max_iterations = MAX_ITERATIONS;
  double threshold = 1e-7;
  void *params = NULL;
  int inner = INNER;
  int outer = OUTER;

  double *gmres_residual_series = (double*) malloc( max_iterations*sizeof(double) );
  double *normF_series          = (double*) malloc( max_iterations*sizeof(double) );
  double *normJtF_series        = (double*) malloc( max_iterations*sizeof(double) );

  //Launch Newton
  cuda_newton_gmres( &ns_equilibria, state_d, &max_iterations, threshold, params, inner, outer, gmres_residual_series, normF_series, normJtF_series );

  EVAL   (&ns_equilibria,    F_d,       state_d, params);
  //EVAL_J (&ns_equilibria,   J1, ones, state, params);
  //EVAL_Jt(&ns_equilibria, JtJ1, J1,   state, params);
  
  double *state = (double*) malloc( (ns_equilibria.input_dim) *sizeof(double) );
  double *F     = (double*) malloc( (ns_equilibria.output_dim)*sizeof(double) );
  cudaMemcpy( state, state_d, (ns_equilibria.input_dim) *sizeof(double), cudaMemcpyDeviceToHost );
  cudaMemcpy( F, F_d,         (ns_equilibria.output_dim)*sizeof(double), cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize(); //Make sure memory has finished copying back to host before writing it to a file



  //Write vector to a binary file for Matlab to read
  FILE *solution_file = fopen("solution.bin", "w");
  int my_N = N;
  fwrite( &my_N,                 sizeof(int),    1                      ,  solution_file );
  fwrite( &max_iterations,       sizeof(int),    1                      ,  solution_file );
  fwrite( state,                 sizeof(double), ns_equilibria.input_dim,  solution_file );
  fwrite(     F,                 sizeof(double), ns_equilibria.output_dim, solution_file );
  fwrite( gmres_residual_series, sizeof(double), max_iterations,           solution_file );
  fwrite( normF_series,          sizeof(double), max_iterations,           solution_file );
  fwrite( normJtF_series,        sizeof(double), max_iterations,           solution_file );
  fclose(solution_file);



  //Free memory
  free(state);
  free(    F);
  free( gmres_residual_series);
  free( normF_series);
  free( normJtF_series);

  cudaFree(state_d);
  cudaFree(    F_d);
  
  return 0;
}


__global__ void equilibrium_objective_kernel(double *F, double *state, void *params){
  /*
  PURPOSE:
  A kernel for computing the objective function on my GPU

  INPUT:
  F - pointer to output array
  state - pointer to input array
  Both of these should be pointing to GPU memory

  OUTPUT:
  F - now has objective function in it
  */

  //First figure out your thread ID
  int idx = blockIdx.x *blockDim.x + threadIdx.x;
  int N2  = N*N;
  int tt  = THREADS*BLOCKS; //Total threads
  double dx = 2*M_PI/N;

  double *u = &state[0*N2];
  double *v = &state[1*N2];
  double *w = &state[2*N2];

  while( idx < N2 ){
    int i = idx / N;
    int j = idx % N;

    F[        IDX(i,j) ] = Dxa(u,i,j) + Dya(v,i,j); //Divergence 1
    F[   N2 + IDX(i,j) ] = Dxb(u,i,j) + Dyb(v,i,j); //Divergence 2
    F[ 2*N2 + IDX(i,j) ] = w[IDX(i,j)] - Dxa(v,i,j) + Dya(u,i,j); //Definition of vorticity
    F[ 3*N2 + IDX(i,j) ] = u[IDX(i,j)]*Dxa(w,i,j) + v[IDX(i,j)]*Dya(w,i,j); //equilibrium
    
    idx = idx + tt; //Go to next gridpoint
  }
}

__global__ void equilibrium_action_of_J_kernel(double *dF, double *input, double *state, void *params){
  /*
  PURPOSE:
  A kernel for action of J function on my GPU

  INPUT:
  output - pointer to output array
  state - pointer to input array
  state - pointer to input array
  Both of these should be pointing to GPU memory

  OUTPUT:
  F - now has objective function in it
  */

  //First figure out your thread ID
  int idx = blockIdx.x *blockDim.x + threadIdx.x;
  int N2  = N*N;
  int tt  = THREADS*BLOCKS; //Total threads
  double dx = 2*M_PI/N;

  double *u = &state[0*N2];
  double *v = &state[1*N2];
  double *w = &state[2*N2];

  double *du = &input[0*N2];
  double *dv = &input[1*N2];
  double *dw = &input[2*N2];

  while( idx < N2 ){
    int i = idx / N;
    int j = idx % N;

    //Divergence 1
    dF[      IDX(i,j) ] = Dxa(du,i,j) + Dya(dv,i,j);
    
    //Divergence 2
    dF[ N2 + IDX(i,j) ] = Dxb(du,i,j) + Dyb(dv,i,j);

    //Vorticity constraint
    dF[ 2*N2 + IDX(i,j) ] = dw[IDX(i,j)] - Dxa(dv,i,j) + Dya(du,i,j);

    //Equilibrium constraint
    dF[ 3*N2 + IDX(i,j) ] =   u[IDX(i,j)]*Dxa(dw,i,j)
                            + v[IDX(i,j)]*Dya(dw,i,j)
                            +du[IDX(i,j)]*Dxa( w,i,j)
                            +dv[IDX(i,j)]*Dya( w,i,j);

    idx = idx + tt; //Go to next gridpoint
  }
}

__global__ void equilibrium_action_of_Jt_kernel(double *dF, double *input, double *state, void *params){
  /*
  PURPOSE:
  A kernel for action of transpose of J function on my GPU

  INPUT:
  output - pointer to output array
  state - pointer to input array
  state - pointer to input array
  Both of these should be pointing to GPU memory

  OUTPUT:
  F - now has objective function in it
  */

  //First figure out your thread ID
  int idx = blockIdx.x *blockDim.x + threadIdx.x;
  int N2  = N*N;
  int tt  = THREADS*BLOCKS; //Total threads
  double dx = 2*M_PI/N;

  double *u = &state[0*N2];
  double *v = &state[1*N2];
  double *w = &state[2*N2];

  double *dx1 = &input[0*N2];
  double *dx2 = &input[1*N2];
  double *dx3 = &input[2*N2];
  double *dx4 = &input[3*N2];

  while( idx < N2 ){
    int i = idx / N;
    int j = idx % N;

    dF[        IDX(i,j) ] = - Dxa(dx1,i,j) - Dxb(dx2,i-1,j-1) - Dya(dx3,i,j) + dx4[IDX(i,j)]*Dxa(w,i,j);
    dF[   N2 + IDX(i,j) ] = - Dya(dx1,i,j) - Dyb(dx2,i-1,j-1) + Dxa(dx3,i,j) + dx4[IDX(i,j)]*Dya(w,i,j); 
    dF[ 2*N2 + IDX(i,j) ] = dx3[IDX(i,j)] - Dxa_prod(dx4,u,i,j) - Dya_prod(dx4,v,i,j);

    idx = idx + tt; //Go to next gridpoint
  }
}



__global__ void initial_data( double *state ){
  int idx = blockIdx.x *blockDim.x + threadIdx.x;
  int N2  = N*N;
  int tt  = THREADS*BLOCKS; //Total threads
  
  double *u = &state[0*N2];
  double *v = &state[1*N2];
  double *w = &state[2*N2];

  double dx = 2*M_PI/N;
  while( idx < N2 ){
    int i = idx / N; 
    int j = idx % N; assert(j >= 0 );
    
    double x = dx*i; 
    double y = dx*j;
    
    u[idx] = U0;
    v[idx] = V0;
    w[idx] = W0;

    idx = idx + tt; //Go to next gridpoint
  }
}

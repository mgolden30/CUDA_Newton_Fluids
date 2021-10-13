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

//CUDA stuff
#include <cuda_runtime.h>
#include <cublas_v2.h>

//GSL stuff
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

//My own code
#include "IO.c"

#define BLOCKS  16
#define THREADS 64

#define N    128
#define DAMP 0.3

#define IDX(i,j)      (i)*N+(j)                        //Indexing compatible with Matlab
#define TRUE_MOD(m,n) (((m)%(n))+(n))%(n)              //A modular operation that ensures you get a nonnegative result
#define IDXP(i,j)     IDX(TRUE_MOD(i,N),TRUE_MOD(j,N)) //Periodic indexing
#define SPACE_LOOP    for(int i=0; i<N; i++) for(int j=0; j<N; j++)

#define GPU_MEM_ALLOCATED     0
#define GPU_MEM_NOT_ALLOCATED 1

#define DEBUG 0

//SYSTEM PARAMETERS
#define FORCING 0 //(4*cos(4*j*dx))
#define NU      0 //(1/40)

typedef struct obj_func{
  int input_dim;
  int output_dim;
  void (*objective_func)(double *, double *, void *);       //pointer to the objective function
  void (*action_of_J )(double *, double *, double *, void *); //pointer to action of Jacobian
  void (*action_of_Jt)(double *, double *, double *, void *); //pointer to action of tranpose of Jacobian
}objective_function;


//Macros for ease of use
#define EVAL(   obj,output,input,params)       ((obj)->objective_func)<<<BLOCKS,THREADS>>>(output,input,params)
#define EVAL_J( obj,output,input,state,params) ((obj)->action_of_J   )<<<BLOCKS,THREADS>>>(output,input,state,params)
#define EVAL_Jt(obj,output,input,state,params) ((obj)->action_of_Jt  )<<<BLOCKS,THREADS>>>(output,input,state,params)

void cuda_newton_gmres( objective_function *obj, double *state, int max_iterations, double threshold, void *params, int inner, int outer );

//KERNELS
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
  int max_iterations = 10000;
  double threshold = 1e-4;
  void *params = NULL;
  int inner = 300;
  int outer = 1;

  //Launch Newton
  cuda_newton_gmres( &ns_equilibria, state_d, max_iterations, threshold, params, inner, outer );

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
  fwrite( &my_N, sizeof(int),    1                      ,  solution_file );
  fwrite( state, sizeof(double), ns_equilibria.input_dim,  solution_file );
  fwrite(     F, sizeof(double), ns_equilibria.output_dim, solution_file );
  fclose(solution_file);



  //Free memory
  free(state);
  free(    F);

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

    //Divergence 1
    F[      IDX(i,j) ] = ( u[IDXP(i+1,j)] - u[IDXP(i-1,j)] 
                        + v[IDXP(i,j+1)] - v[IDXP(i,j-1)] )/(2*dx);
    
    //Divergence 2
    F[ N2 + IDX(i,j) ] = ( u[IDXP(i+1,j)] + u[IDXP(i+1,j+1)] - u[IDXP(i,j)] - u[IDXP(i,j+1)]
                        + v[IDXP(i,j+1)] + v[IDXP(i+1,j+1)] - v[IDXP(i,j)] - v[IDXP(i+1,j)]
                        )/(2*dx);

    //Vorticity constraint
    F[ 2*N2 + IDX(i,j) ] = w[IDX(i,j)] - ( -u[IDXP(i,j+1)] + u[IDXP(i,j-1)] 
                                          +v[IDXP(i+1,j)] - v[IDXP(i-1,j)] )/(2*dx);

    //Equilibrium constraint
    F[ 3*N2 + IDX(i,j) ] = u[IDX(i,j)]*(w[IDXP(i+1,j)] - w[IDXP(i-1,j)])/(2*dx)
                          +v[IDX(i,j)]*(w[IDXP(i,j+1)] - w[IDXP(i,j-1)])/(2*dx)
                          -NU*(w[IDXP(i+1,j)] + w[IDXP(i-1,j)] + w[IDXP(i,j+1)] + w[IDX(i,j-1)] - 4*w[IDX(i,j)])/dx/dx - FORCING;

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
    dF[      IDX(i,j) ] = (du[IDXP(i+1,j)] - du[IDXP(i-1,j)] 
                          +dv[IDXP(i,j+1)] - dv[IDXP(i,j-1)] )/(2*dx);
    
    //Divergence 2
    dF[ N2 + IDX(i,j) ] = ( du[IDXP(i+1,j)] + du[IDXP(i+1,j+1)] - du[IDXP(i,j)] - du[IDXP(i,j+1)]
                          + dv[IDXP(i,j+1)] + dv[IDXP(i+1,j+1)] - dv[IDXP(i,j)] - dv[IDXP(i+1,j)]
                          )/(2*dx);

    //Vorticity constraint
    dF[ 2*N2 + IDX(i,j) ] = dw[IDX(i,j)] - ( -du[IDXP(i,j+1)] + du[IDXP(i,j-1)] 
                                             +dv[IDXP(i+1,j)] - dv[IDXP(i-1,j)] )/(2*dx);

    //Equilibrium constraint
    dF[ 3*N2 + IDX(i,j) ] =  u[IDX(i,j)]*(dw[IDXP(i+1,j)] - dw[IDXP(i-1,j)])/(2*dx)
                           + v[IDX(i,j)]*(dw[IDXP(i,j+1)] - dw[IDXP(i,j-1)])/(2*dx)
                           +du[IDX(i,j)]*(w[IDXP(i+1,j)] - w[IDXP(i-1,j)])/(2*dx)
                           +dv[IDX(i,j)]*(w[IDXP(i,j+1)] - w[IDXP(i,j-1)])/(2*dx)
                           -NU*(dw[IDXP(i+1,j)] + dw[IDXP(i-1,j)] + dw[IDXP(i,j+1)] + dw[IDX(i,j-1)] - 4*dw[IDX(i,j)])/dx/dx;

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

    dF[        IDX(i,j) ] = (
                             dx1[IDXP(i-1,j)] - dx1[IDXP(i+1,j )] 
                            +dx2[IDXP(i-1,j)] + dx2[IDXP(i-1,j-1)] - dx2[IDXP(i,j  )] - dx2[IDXP(i,j-1  )]
                            +dx3[IDXP(i,j-1)] - dx3[IDXP(i,j+1  )]
                            +dx4[IDXP(i,j)]*( w[IDXP(i+1,j)] - w[IDXP(i-1,j)] )
                            )/(2*dx);
    
    dF[   N2 + IDX(i,j) ] = (
                             dx1[IDXP(i,j-1)] - dx1[IDXP(i,j+1 )]
                            +dx2[IDXP(i,j-1)] + dx2[IDXP(i-1,j-1)] - dx2[IDXP(i,j)] - dx2[IDXP(i-1,j)]
                            -dx3[IDXP(i-1,j)] + dx3[IDXP(i+1,j  )]
                            +dx4[IDXP(i,j)]*( w[IDXP(i,j+1)] - w[IDXP(i,j-1)] )
                            )/(2*dx);

    double lap_dx4 = (dx4[IDXP(i+1,j)] + dx4[IDXP(i-1,j)] + dx4[IDXP(i,j+1)] + dx4[IDX(i,j-1)] - 4*dx4[IDX(i,j)])/dx/dx;
    dF[ 2*N2 + IDX(i,j) ] = dx3[IDX(i,j)] 
                          + (dx4[IDXP(i-1,j)]*u[IDXP(i-1,j)] - dx4[IDXP(i+1,j)]*u[IDXP(i+1,j)] + dx4[IDXP(i,j-1)]*v[IDXP(i,j-1)] - dx4[IDXP(i,j+1)]*v[IDXP(i,j+1)])/(2*dx)
                          - NU*lap_dx4;

    idx = idx + tt; //Go to next gridpoint
  }
}


void cuda_newton_gmres( objective_function *obj, double* state_d, int max_iterations, double threshold, void *params, int inner, int outer ){
  /*
  PURPOSE:
    The purpose of this function is to find a zero (or minimum) of an objective function depedend on a large number of real variables.
    Traditional generalized Newton's method relies on inverting the Jacobian or computing a psuedo-inverse. When the number of variables is large (>10^6),
    this computation is infeasible on useful timescales. GMRES is used to approximate a Newton step.

  INPUT:
    objective_function *obj - the function we are trying to minimize. 
    gsl_vector       *state - the initial guess for the minimal point
    int      max_iterations - the number of Newton steps this function is allowed to take.
    double        threshold - Newton ill exit if the norm of the objective function falls below this threshold
    void            *params - a pointer to parameters needed by the objective function. NULL if you don't need anything
    int               inner - number of inner iterations used in GMRES
  
  OUTPUT:
    double *state_d - will be overwritten to contain the new estimate of minimum
  */
 
  clock_t start, stop; //Used for timing GMRES
  start = clock();

  int input_dim  = obj->input_dim;
  int output_dim = obj->output_dim;

  if( output_dim < input_dim ){
    printf("GMRES did not launch: Your problem is underconstrained.\n");
    return;
  }
  int overconstrained = output_dim > input_dim;

  double normF, normJtF;
  double residual;
  double h_element;

  //CUBLAS requires a handle
  cublasHandle_t handle;
  cublasCreate( &handle );

  //GPU VECTORS
  double *F_d;   //objective function on the GPU
  double *JtF_d; //J'*F on the GPU. Used to minimize over-constrained problems
  double *Jq_d;
  double *JtJq_d;
  double *step_d;
  double *q_col_d;  //Doesn't need allocated
  double *q_col2_d; //Doesn't need allocated
  double *y_d;

  cudaMalloc( &F_d,    output_dim*sizeof(double) );
  cudaMalloc( &JtF_d,  input_dim *sizeof(double) );
  cudaMalloc( &Jq_d,   output_dim*sizeof(double) );
  cudaMalloc( &JtJq_d, input_dim *sizeof(double) );
  cudaMalloc( &step_d, input_dim *sizeof(double) );
  cudaMalloc( &y_d,         inner*sizeof(double) );
  
  //GPU MATRICES
  double *q_d;   //Matrix containing orthonormal basis vectors of krylov subspace. 
  cudaMalloc( &q_d, input_dim*(inner+1)*sizeof(double) );

  gsl_matrix *h     = gsl_matrix_calloc(inner+1,   inner  ); //Hessenberg form of matrix
  gsl_matrix *hth   = gsl_matrix_calloc(inner,     inner  ); //h'*h in matlab
  gsl_vector *b2    = gsl_vector_calloc(inner+1);  //The right hand side of GMRES. It's a trivial vector with one non-zero element.
  gsl_vector *htb2  = gsl_vector_calloc(inner);
  gsl_vector *y     = gsl_vector_calloc(inner);

  for(int i=0; i<max_iterations; i++){
    int outer_iterations = 0; //Reset each Newton step
    
    //Step 1: check |F| or |b| = |J'*F| to monitor convergence
    if(overconstrained){
      EVAL   (obj,   F_d,      state_d, params);
      EVAL_Jt(obj, JtF_d, F_d, state_d, params);
      cublasDnrm2( handle, output_dim,   F_d, 1, &normF   );
      cublasDnrm2( handle,  input_dim, JtF_d, 1, &normJtF );
      cudaDeviceSynchronize();

      printf( "Iteration %d: |F| = %.9e, |J'*F| = %.9e\n", i, normF, normJtF );
      if( normJtF < threshold ){
        //Only look for exit condition from normbJtF, which should be identically zero at a local minima
        printf("|J'*F| is less than specified threshold. Exiting Newton...\n");
	return;
      }
    }
    else{
      //Not overconstrained
      EVAL(obj, F_d, state_d, params);
      cublasDnrm2( handle, output_dim, F_d, 1, &normF );
      cudaDeviceSynchronize();
      
      printf("Iteration %d: |F| = %.9e\n", i, normF );
      if( normF < threshold ){
        printf("|F| is less than specified threshold. Exiting Newton...\n");
        return;
      }
    }
  
    //Simplifies code to introduce pointer "work". This is where A*(column of q) lives
    //It depends on if the problem is overconstrained
    double *work;
    work = overconstrained ? JtJq_d : Jq_d;
    
//Repeated iterations of GMRES start here with an updated step
restart:
    if(overconstrained){
      //In these lines, store "b - J'*J(step)" in the first column of q
      EVAL_J (obj, Jq_d,  step_d, state_d, params); // jq <- J(step)
      EVAL_Jt(obj, work,    Jq_d, state_d, params); // work <- J(step)
      double minus_one = -1;
      cublasDaxpy(handle, input_dim, &minus_one, JtF_d, 1, work, 1); //work <- work - JtF
    }
    else{
      //In these lines, store "f - J(step)" in the first column of q
      EVAL_J(obj, work, step_d, state_d, params); // jq <- J(step)
      double minus_one = -1;
      cublasDaxpy(handle, input_dim, &minus_one, F_d, 1, work, 1); //work <- work - F
    }
    
    //Compute norm
    cublasDnrm2( handle, input_dim, work, 1, &h_element );
    cudaDeviceSynchronize(); //Sadly we need to wait for this norm to be computed.
    double temp = -1/h_element;
    cublasDscal( handle, input_dim, &temp, work, 1 ); //work <- (F - J(step))/norm gives a unit vector!
    gsl_vector_set(b2, 0, h_element ); //This is the only non-zero element of b2
    
    //Set the first column of q to this unit vector
    cublasDcopy( handle, input_dim, work, 1, q_d, 1 ); //Copy this unit vector to the first column of q_d
    for(int j=0; j<inner; j++){
      q_col_d = &q_d[j*input_dim]; //pointer to the relevant column
      if(overconstrained){
        EVAL_J (obj,   Jq_d, q_col_d, state_d, params);
        EVAL_Jt(obj,   work,    Jq_d, state_d, params);
      }
      else{
        EVAL_J(obj, work, q_col_d, state_d, params);
      }
      for(int k=0; k<=j; k++){
        q_col2_d = &q_d[k*input_dim];
        cublasDdot( handle, input_dim, q_col2_d, 1, work, 1, &h_element);
        cudaDeviceSynchronize(); //Sadly we need to wait for this next step
	gsl_matrix_set(h, k, j, h_element); //I'll stick with CPU implementation for now
        //printf("h(%d,%d) = %.6e\n", k, j, h_element);
        temp = -h_element;
	cublasDaxpy( handle, input_dim, &temp, q_col2_d, 1, work, 1 );
      }

      cublasDnrm2( handle, input_dim, work, 1, &h_element );
      cudaDeviceSynchronize(); //Sadly we need to wait for this next step
      gsl_matrix_set(h, j+1, j, h_element);
      //printf("h(%d,%d) = %.6e\n", j+1, j, h_element);
      temp = 1/h_element;
      cublasDscal( handle, input_dim, &temp, work, 1 );
      cublasDcopy( handle, input_dim, work, 1, &q_d[(j+1)*input_dim], 1 );
    }

    if(DEBUG){ print_vector(b2, "b2"); }
    if(DEBUG){ print_matrix(h, "h"); }
    //print_matrix(q, "q");

    //Set hth to H'*H
    gsl_blas_dgemm( CblasTrans, CblasNoTrans, 1, h, h, 0, hth );
    
    //Set htb2 to H'*b2
    gsl_blas_dgemv( CblasTrans, 1, h, b2, 0, htb2 );

    //Solve the linear system with LU decomp
    //Stolen from gsl documentation
    //https://www.gnu.org/software/gsl/doc/html/linalg.html
    gsl_permutation *p = gsl_permutation_alloc(inner); int s;
    gsl_linalg_LU_decomp(hth, p, &s);
    gsl_linalg_LU_solve(hth, p, htb2, y);
    gsl_permutation_free(p);

    if(DEBUG){ print_vector(y, "y"); }

    //Use y to update step
    cudaMemcpy( y_d, y->data, inner*sizeof(double), cudaMemcpyHostToDevice );
    temp = 1;
    //Calling CUBLAS's gemv correctly is the trickiest part of this
    cublasDgemv( handle,         // handle is needed for all CUBLAS operations
		 CUBLAS_OP_N,    // CUBLAS_OP_N - no transpose, CUBLAS_OP_T - transpose
	         input_dim,      // number of rows of A
		 inner,          // number of columns of A
	         &temp,          // Alpha 
	         q_d,            // pointer to A 
		 input_dim,      // lda
	         y_d,            // pointer to x
		 1,              // incx
		 &temp,          // Beta 
	         step_d,         // pointer to y
		 1               // incy
               );
   
    //q->size2--;//Hide last column of q
    //gsl_blas_dgemv( CblasNoTrans, 1, q, y, 1, step ); //update step
    //q->size2++;//Restore last column of q
   
    //Check residual
    if(overconstrained){
      EVAL_J (obj,   Jq_d,  step_d, state_d, params);
      EVAL_Jt(obj, JtJq_d,  Jq_d,   state_d, params);
      double minus_one = -1;
      cublasDaxpy( handle, input_dim, &minus_one, JtF_d, 1, JtJq_d, 1);
      cublasDnrm2( handle, input_dim, JtJq_d, 1, &residual );
      cudaDeviceSynchronize();
      residual = residual/normJtF;
    }
    else{
      EVAL_J(obj, Jq_d, step_d, state_d, params); // jq <- J(step)
      double minus_one = -1;
      cublasDaxpy( handle, input_dim, &minus_one, F_d, 1, Jq_d, 1);
      cublasDnrm2( handle, input_dim, Jq_d, 1, &residual );
      cudaDeviceSynchronize();
      residual = residual/normF;
    }

    outer_iterations++; //congratulations you have completed an outer iteration
    if( residual > 1e-5 & outer_iterations < outer ){
      goto restart;
    }
    
    //Only display residual at end of iterations
    printf("GMRES residual = %e\n", residual);

    //Newton step
    //feel free to damp and use like -0.1
    temp = -DAMP; //How much to move along the Newton direction.
    cublasDaxpy( handle, input_dim, &temp, step_d, 1, state_d, 1);
  }

  //Vectors
  cudaFree( F_d    );
  cudaFree( JtF_d  );
  cudaFree( Jq_d   );
  cudaFree( JtJq_d );
  cudaFree( step_d );

  //Matrices
  cudaFree( q_d );

  gsl_matrix_free(h);
  gsl_matrix_free(hth);
  gsl_vector_free(y);
  gsl_vector_free(b2);
  gsl_vector_free(htb2);

  stop = clock();
  printf("Newton-GMRES took %f seconds. Have a nice day!\n", (double)(stop - start)/CLOCKS_PER_SEC );
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
    
    u[idx] = cos(x+2*y + 0.4);
    v[idx] = sin(3*y);
    w[idx] = cos(2*x+0.7)*cos(y-0.5) + sin(5*x);

    idx = idx + tt; //Go to next gridpoint
  }
}

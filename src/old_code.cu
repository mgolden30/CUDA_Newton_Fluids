#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

//GSL stuff
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

//CUDA stuff
#include <cuda_runtime.h>

#define BLOCKS  16
#define THREADS 64

#define N    512
#define DAMP 0.5

#define IDX(i,j)      (i)*N+(j)                        //Indexing compatible with Matlab
#define TRUE_MOD(m,n) (((m)%(n))+(n))%(n)              //A modular operation that ensures you get a nonnegative result
#define IDXP(i,j)     IDX(TRUE_MOD(i,N),TRUE_MOD(j,N)) //Periodic indexing
#define SPACE_LOOP    for(int i=0; i<N; i++) for(int j=0; j<N; j++)

#define GPU_MEM_ALLOCATED     0
#define GPU_MEM_NOT_ALLOCATED 1


//SYSTEM PARAMETERS
#define FORCING (4*cos(4*j*dx))
#define NU      (1/20)

typedef struct obj_func{
  int input_dim;
  int output_dim;
  void (*objective_func)(gsl_vector *, gsl_vector *, void *); //pointer to the objective function
  void (*action_of_J )(gsl_vector *, gsl_vector *, gsl_vector *, void *);   //pointer to action of Jacobian
  void (*action_of_Jt)(gsl_vector *, gsl_vector *, gsl_vector *, void *);   //pointer to action of tranpose of Jacobian
}objective_function;


//Macros for ease of use
#define EVAL(   obj,output,input,params)       ((obj)->objective_func)(output,input,params)
#define EVAL_J( obj,output,input,state,params) ((obj)->action_of_J   )(output,input,state,params)
#define EVAL_Jt(obj,output,input,state,params) ((obj)->action_of_Jt  )(output,input,state,params)


void newton_gmres( objective_function *obj, gsl_vector *state, int max_iterations, double threshold, void *params, int inner, int outer );

//Really these are for debugging GMRES
void print_norm( gsl_vector *v, char name[] );
void print_matrix( gsl_matrix *m, char name[] );
void print_vector( gsl_vector *v, char name[] );

void equilibria_objective_function(gsl_vector *output, gsl_vector *state, void *params);
void equilibria_action_of_J       (gsl_vector *output, gsl_vector *input, gsl_vector *state, void *params);
void equilibria_action_of_Jt      (gsl_vector *output, gsl_vector *input, gsl_vector *state, void *params);

//CUDA variants
void equilibria_objective_function_CUDA(gsl_vector *output, gsl_vector *state, void *params);
void equilibria_action_of_J_CUDA       (gsl_vector *output, gsl_vector *input, gsl_vector *state, void *params);
void equilibria_action_of_Jt_CUDA      (gsl_vector *output, gsl_vector *input, gsl_vector *state, void *params);
__global__ void equilibrium_objective_kernel   (double *F, double *state);
__global__ void equilibrium_action_of_J_kernel (double *output, double *input, double *state);
__global__ void equilibrium_action_of_Jt_kernel(double *output, double *input, double *state);

void initial_data( gsl_vector *state );


int main( int argc, char **argv){
  //INITIALIZE OBJECTIVE FUNCTION
  objective_function euler_equilibria;
  euler_equilibria.input_dim      = 3*N*N;
  euler_equilibria.output_dim     = 4*N*N;
  euler_equilibria.objective_func = (equilibria_objective_function);
  euler_equilibria.action_of_J    = (equilibria_action_of_J);
  euler_equilibria.action_of_Jt   = (equilibria_action_of_Jt);
 
  objective_function euler_equilibria_CUDA = euler_equilibria;
  euler_equilibria_CUDA.objective_func = (equilibria_objective_function_CUDA);
  euler_equilibria_CUDA.action_of_J    = (equilibria_action_of_J_CUDA);
  euler_equilibria_CUDA.action_of_Jt   = (equilibria_action_of_Jt_CUDA);
 

  //INITIALIZE INITIAL GUESS
  gsl_vector *state = gsl_vector_calloc(3*N*N);
  gsl_vector *f     = gsl_vector_calloc(4*N*N);
  gsl_vector *J1    = gsl_vector_calloc(4*N*N);
  gsl_vector *JtJ1  = gsl_vector_calloc(3*N*N);
  initial_data( state );

  //Newton parameters
  int max_iterations = 2;
  double threshold = 1e-4;
  void *params = NULL;
  int inner = 6;
  int outer = 1;

  /*
  int trials = 100;
  clock_t start, stop;

  start=clock(); 
  for(int i=0; i<trials; i++){
    EVAL   (&euler_equilibria, f,        state, params);
    EVAL_J (&euler_equilibria, J1,    f, state, params);
    EVAL_Jt(&euler_equilibria, JtJ1, J1, state, params);
  }
  stop=clock(); 
  print_norm( f, "f" );
  print_norm( J1, "Jf" );
  print_norm( JtJ1, "JtJf" );
  printf("process took %f seconds\n", (double)(stop - start)/CLOCKS_PER_SEC );


  start=clock(); 
  for(int i=0; i<trials; i++){
    EVAL   (&euler_equilibria_CUDA, f,         state, params);
    cudaDeviceSynchronize(); //Make sure it finishes
    EVAL_J (&euler_equilibria_CUDA, J1, state, state, params);
    cudaDeviceSynchronize(); //Make sure it finishes
    EVAL_Jt(&euler_equilibria_CUDA, JtJ1,  J1, state, params);
    cudaDeviceSynchronize(); //Make sure it finishes
  }
  stop=clock();
  print_norm( f, "f" );
  print_norm( J1, "Jf" );
  print_norm( JtJ1, "JtJf" );
  printf("process took %f seconds\n", (double)(stop - start)/CLOCKS_PER_SEC );

  return 0;
  */

  //Launch Newton
  newton_gmres( &euler_equilibria,      state, max_iterations, threshold, params, inner, outer );
  //newton_gmres( &euler_equilibria_CUDA, state, max_iterations, threshold, params, inner, outer );

  EVAL   (&euler_equilibria,    f,       state, params);
//  EVAL_J (&euler_equilibria,   J1, ones, state, params);
//  EVAL_Jt(&euler_equilibria, JtJ1, J1,   state, params);
  
  //Write vector to a binary file for Matlab to read
  FILE *out = fopen("solution.bin", "w");
  gsl_vector_fwrite( out, state );
  gsl_vector_fwrite( out, f );
  fclose(out);

  gsl_vector_free(state);

  return 0;
}


void print_matrix( gsl_matrix *m, char name[] ){
  printf("\n%s = \n", name);
  for(int i=0; i<(m->size1); i++){
    for(int j=0; j<(m->size2); j++){
      printf("%.6f ", gsl_matrix_get(m, i, j) );
    }
    printf("\n");
  }
  printf("\n");
}



void print_vector( gsl_vector *v, char name[] ){
  printf("\n%s = \n", name);
  for( int i=0; i<(v->size); i++){
    printf("%.6f\n", gsl_vector_get(v,i));
  }
  printf("\n");
}

__global__ void equilibrium_objective_kernel(double *F, double *state){
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

__global__ void equilibrium_action_of_J_kernel(double *dF, double *input, double *state){
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

__global__ void equilibrium_action_of_Jt_kernel(double *dF, double *input, double *state){
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



void equilibria_objective_function_CUDA(gsl_vector *output, gsl_vector *state, void *params){
  //Transfer state to GPU
  clock_t start, stop;

  static double *gpu_data;
  static double *gpu_output;
  static int allocated = GPU_MEM_NOT_ALLOCATED;
  if( allocated != GPU_MEM_ALLOCATED ){
    cudaError_t error_report;
    error_report = cudaMalloc( &gpu_data,   ( state->size)*sizeof(double) ); assert(error_report == cudaSuccess);
    error_report = cudaMalloc( &gpu_output, (output->size)*sizeof(double) ); assert(error_report == cudaSuccess);
    allocated = GPU_MEM_ALLOCATED;
  }
  cudaMemcpy( gpu_data, state->data, (state->size)*sizeof(double), cudaMemcpyHostToDevice );

  //Launch the CUDA kernel to compute the objective function
  start = clock();
  equilibrium_objective_kernel<<<BLOCKS,THREADS>>>(gpu_output, gpu_data);
  cudaDeviceSynchronize(); //Make sure it finishes
  stop = clock();
  printf("The actual GPU computation took %f seconds\n", (double)(stop - start)/CLOCKS_PER_SEC );


  //Transfer data back to cpu
  start = clock();  
  cudaMemcpy( output->data, gpu_output, (output->size)*sizeof(double), cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize(); //Make sure it finishes
  stop = clock();
  printf("Copying data took %f seconds\n", (double)(stop - start)/CLOCKS_PER_SEC );
}

void equilibria_action_of_J_CUDA(gsl_vector *output, gsl_vector *input, gsl_vector *state, void *params){
  //Transfer state to GPU
  static double *gpu_state;
  static double *gpu_input;
  static double *gpu_output;
  static int allocated = GPU_MEM_NOT_ALLOCATED;
  if( allocated != GPU_MEM_ALLOCATED ){
    cudaMalloc( &gpu_state,   ( state->size)*sizeof(double) );
    cudaMalloc( &gpu_input,   ( input->size)*sizeof(double) );
    cudaMalloc( &gpu_output,  (output->size)*sizeof(double) );
    allocated = GPU_MEM_ALLOCATED;
  }
  cudaMemcpy( gpu_state, state->data, (state->size)*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( gpu_input, input->data, (input->size)*sizeof(double), cudaMemcpyHostToDevice );

  //Launch the CUDA kernel to compute the objective function
  equilibrium_action_of_J_kernel<<<BLOCKS,THREADS>>>(gpu_output, gpu_input, gpu_state);
  cudaDeviceSynchronize(); //Make sure it finishes

  //Transfer data back to cpu
  cudaMemcpy( output->data, gpu_output, (output->size)*sizeof(double), cudaMemcpyDeviceToHost );
}


void equilibria_action_of_Jt_CUDA(gsl_vector *output, gsl_vector *input, gsl_vector *state, void *params){
  //Transfer state to GPU
  static double *gpu_state;
  static double *gpu_input;
  static double *gpu_output;
  static int allocated = GPU_MEM_NOT_ALLOCATED;
  if( allocated != GPU_MEM_ALLOCATED ){
    cudaError_t error_report;
    error_report = cudaMalloc( &gpu_state,   ( state->size)*sizeof(double) ); assert( error_report == cudaSuccess );
    error_report = cudaMalloc( &gpu_input,   ( input->size)*sizeof(double) ); assert( error_report == cudaSuccess );
    error_report = cudaMalloc( &gpu_output,  (output->size)*sizeof(double) ); assert( error_report == cudaSuccess );
    allocated = GPU_MEM_ALLOCATED;
  }
  cudaMemcpy( gpu_state, state->data, (state->size)*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( gpu_input, input->data, (input->size)*sizeof(double), cudaMemcpyHostToDevice );
  cudaDeviceSynchronize(); //Make sure it finishes

  //Launch the CUDA kernel to compute the objective function
  equilibrium_action_of_Jt_kernel<<<BLOCKS,THREADS>>>(gpu_output, gpu_input, gpu_state);
  cudaDeviceSynchronize(); //Make sure it finishes

  //Transfer data back to cpu
  cudaMemcpy( output->data, gpu_output, (output->size)*sizeof(double), cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize(); //Make sure it finishes
}



void equilibria_objective_function(gsl_vector *output, gsl_vector *state, void *params){
  gsl_vector u = *state; u.size = N*N; u.data = &u.data[0*N*N];
  gsl_vector v = *state; v.size = N*N; v.data = &v.data[1*N*N];
  gsl_vector w = *state; w.size = N*N; w.data = &w.data[2*N*N];

  double dx = 2*M_PI/N;
  double val;

  SPACE_LOOP{
    //Incompressibility 1
    val =   gsl_vector_get(&u, IDXP(i+1,j)) - gsl_vector_get(&u, IDXP(i-1,j))
          + gsl_vector_get(&v, IDXP(i,j+1)) - gsl_vector_get(&v, IDXP(i,j-1));
    val = val/(2*dx);
    gsl_vector_set( output, IDX(i,j), val );
    
    //Incompressibility 2
    val =   gsl_vector_get(&u, IDXP(i+1,j  )) + gsl_vector_get(&u, IDXP(i+1,j+1))
          - gsl_vector_get(&u, IDXP(i,  j  )) - gsl_vector_get(&u, IDXP(i,  j+1))
          + gsl_vector_get(&v, IDXP(i  ,j+1)) + gsl_vector_get(&v, IDXP(i+1,j+1))
          - gsl_vector_get(&v, IDXP(i,  j  )) - gsl_vector_get(&v, IDXP(i+1,j  ));
    val = val/(2*dx);
    gsl_vector_set( output, N*N + IDX(i,j), val );
  
    //Definition of vorticity
    val =   gsl_vector_get(&w, IDXP(i,j  )) 
          -(gsl_vector_get(&v, IDXP(i+1,j  ))
          - gsl_vector_get(&v, IDXP(i-1,j  ))
          - gsl_vector_get(&u, IDXP(i  ,j+1))
          + gsl_vector_get(&u, IDXP(i  ,j-1))
	   )/(2*dx);
    gsl_vector_set( output, 2*N*N + IDX(i,j), val );

    //Equilibrium condition
    val = gsl_vector_get(&u, IDX(i,j))*( gsl_vector_get(&w, IDXP(i+1,j)) - gsl_vector_get(&w, IDXP(i-1,j)) )
        + gsl_vector_get(&v, IDX(i,j))*( gsl_vector_get(&w, IDXP(i,j+1)) - gsl_vector_get(&w, IDXP(i,j-1)) );
    val = val/(2*dx);
    double laplacian_w = (  gsl_vector_get(&w, IDXP(i+1,j)) 
                           +gsl_vector_get(&w, IDXP(i-1,j))
                           +gsl_vector_get(&w, IDXP(i,j+1)) 
                           +gsl_vector_get(&w, IDXP(i,j-1)) 
                         -4*gsl_vector_get(&w, IDXP(i,j  ))
                         )/(dx*dx);
    val = val - NU*laplacian_w - FORCING;
    gsl_vector_set( output, 3*N*N + IDX(i,j), val );
  }
}



void equilibria_action_of_J(gsl_vector *output, gsl_vector *input, gsl_vector *state, void *params){
  gsl_vector u = *state; u.size = N*N; u.data = &u.data[0*N*N];
  gsl_vector v = *state; v.size = N*N; v.data = &v.data[1*N*N];
  gsl_vector w = *state; w.size = N*N; w.data = &w.data[2*N*N];

  gsl_vector du = *input; du.size = N*N; du.data = &du.data[0*N*N];
  gsl_vector dv = *input; dv.size = N*N; dv.data = &dv.data[1*N*N];
  gsl_vector dw = *input; dw.size = N*N; dw.data = &dw.data[2*N*N];

  double dx = 2*M_PI/N;

  double val;
  SPACE_LOOP{
    //Incompressibility 1
    val =   gsl_vector_get(&du, IDXP(i+1,j))
          - gsl_vector_get(&du, IDXP(i-1,j))
          + gsl_vector_get(&dv, IDXP(i,j+1))
          - gsl_vector_get(&dv, IDXP(i,j-1));
    val = val/(2*dx);
    gsl_vector_set( output, IDX(i,j), val );
    
    //Incompressibility 2
    val =   gsl_vector_get(&du, IDXP(i+1,j  )) 
          + gsl_vector_get(&du, IDXP(i+1,j+1))
          - gsl_vector_get(&du, IDXP(i,  j  ))
          - gsl_vector_get(&du, IDXP(i,  j+1))
          + gsl_vector_get(&dv, IDXP(i  ,j+1))
          + gsl_vector_get(&dv, IDXP(i+1,j+1))
          - gsl_vector_get(&dv, IDXP(i,  j  ))
          - gsl_vector_get(&dv, IDXP(i+1,j  ));
    val = val/(2*dx);
    gsl_vector_set( output, N*N + IDX(i,j), val );
  
    //Definition of vorticity
    val =   gsl_vector_get(&dw, IDXP(i,j  )) 
          -(gsl_vector_get(&dv, IDXP(i+1,j  ))
          - gsl_vector_get(&dv, IDXP(i-1,j  ))
          - gsl_vector_get(&du, IDXP(i  ,j+1))
          + gsl_vector_get(&du, IDXP(i  ,j-1)))/(2*dx);
    gsl_vector_set( output, 2*N*N + IDX(i,j), val );

    //Equilibrium condition
    val = gsl_vector_get(&du, IDX(i,j))*(gsl_vector_get(&w,  IDXP(i+1,j)) - gsl_vector_get(&w,  IDXP(i-1,j)))
        + gsl_vector_get(&dv, IDX(i,j))*(gsl_vector_get(&w,  IDXP(i,j+1)) - gsl_vector_get(&w,  IDXP(i,j-1)))
        + gsl_vector_get(&u,  IDX(i,j))*(gsl_vector_get(&dw, IDXP(i+1,j)) - gsl_vector_get(&dw, IDXP(i-1,j)))
        + gsl_vector_get(&v,  IDX(i,j))*(gsl_vector_get(&dw, IDXP(i,j+1)) - gsl_vector_get(&dw, IDXP(i,j-1)));
    val = val/(2*dx);
    double laplacian_dw = (  gsl_vector_get(&dw, IDXP(i+1,j)) 
                            +gsl_vector_get(&dw, IDXP(i-1,j))
                            +gsl_vector_get(&dw, IDXP(i,j+1)) 
                            +gsl_vector_get(&dw, IDXP(i,j-1)) 
                          -4*gsl_vector_get(&dw, IDXP(i,j  ))
                          )/(dx*dx);
    val = val - NU*laplacian_dw;
    gsl_vector_set( output, 3*N*N + IDX(i,j), val );
  }
}



void equilibria_action_of_Jt(gsl_vector *output, gsl_vector *input, gsl_vector *state, void *params){
  gsl_vector dx1 = *input; dx1.size = N*N; dx1.data = &dx1.data[0*N*N];
  gsl_vector dx2 = *input; dx2.size = N*N; dx2.data = &dx2.data[1*N*N];
  gsl_vector dx3 = *input; dx3.size = N*N; dx3.data = &dx3.data[2*N*N];
  gsl_vector dx4 = *input; dx4.size = N*N; dx4.data = &dx4.data[3*N*N];

  gsl_vector u = *state; u.size = N*N; u.data = &u.data[0*N*N];
  gsl_vector v = *state; v.size = N*N; v.data = &v.data[1*N*N];
  gsl_vector w = *state; w.size = N*N; w.data = &w.data[2*N*N];

  double dx = 2*M_PI/N;

  double val;
  SPACE_LOOP{
    val = gsl_vector_get(&dx1, IDXP(i-1,j)) - gsl_vector_get(&dx1, IDXP(i+1,j  ))
	 +gsl_vector_get(&dx2, IDXP(i-1,j)) + gsl_vector_get(&dx2, IDXP(i-1,j-1)) - gsl_vector_get(&dx2, IDXP(i,j)) - gsl_vector_get(&dx2, IDXP(i,j-1))
         +gsl_vector_get(&dx3, IDXP(i,j-1)) - gsl_vector_get(&dx3, IDXP(i,j+1))
	 +gsl_vector_get(&dx4, IDX(i,j  ))*( gsl_vector_get(&w, IDXP(i+1,j)) - gsl_vector_get(&w, IDXP(i-1,j)) );
    val = val/(2*dx);
    gsl_vector_set( output, IDX(i,j), val );


    val = gsl_vector_get(&dx1, IDXP(i,j-1)) - gsl_vector_get(&dx1, IDXP(i,j+1  ))
	 +gsl_vector_get(&dx2, IDXP(i,j-1)) + gsl_vector_get(&dx2, IDXP(i-1,j-1)) - gsl_vector_get(&dx2, IDX(i,j)) - gsl_vector_get(&dx2, IDXP(i-1,j))
         -gsl_vector_get(&dx3, IDXP(i-1,j)) + gsl_vector_get(&dx3, IDXP(i+1,j  ))
	 +gsl_vector_get(&dx4, IDXP(i,j  ))*( gsl_vector_get(&w, IDXP(i,j+1)) - gsl_vector_get(&w, IDXP(i,j-1)) );
    val = val/(2*dx);
    gsl_vector_set( output, N*N + IDX(i,j), val );

    double laplacian_dx4 = (  gsl_vector_get(&dx4, IDXP(i+1,j)) 
                             +gsl_vector_get(&dx4, IDXP(i-1,j))
                             +gsl_vector_get(&dx4, IDXP(i,j+1)) 
                             +gsl_vector_get(&dx4, IDXP(i,j-1)) 
                           -4*gsl_vector_get(&dx4, IDXP(i,j  ))
                           )/(dx*dx);
    val =  gsl_vector_get(&dx3, IDX(i,j))
	  +( gsl_vector_get(&dx4, IDXP(i-1,j))*gsl_vector_get(&u, IDXP(i-1,j))
            -gsl_vector_get(&dx4, IDXP(i+1,j))*gsl_vector_get(&u, IDXP(i+1,j))
            +gsl_vector_get(&dx4, IDXP(i,j-1))*gsl_vector_get(&v, IDXP(i,j-1))
            -gsl_vector_get(&dx4, IDXP(i,j+1))*gsl_vector_get(&v, IDXP(i,j+1))
	   )/(2*dx);
    val = val - NU*laplacian_dx4;
    gsl_vector_set( output, 2*N*N + IDX(i,j), val );
  }
}



void newton_gmres( objective_function *obj, gsl_vector *state, int max_iterations, double threshold, void *params, int inner, int outer ){
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
    gsl_vector       *state - will be overwritten to contain the new estimate of minimum
  */
  
  clock_t start, stop; //Used for timing GMRES
  start = clock();

  int input_dim  = obj->input_dim;
  int output_dim = obj->output_dim;

  if( output_dim < input_dim ){
    printf("Your problem is underconstrained. I have put no effort into making my code work in this case. Sorry.\n");
    return;
  }
  int overconstrained = output_dim > input_dim;

  double normf;
  double normb;
  double residual;
  double h_element;

  gsl_matrix *q     = gsl_matrix_calloc(input_dim, inner+1); //Orthonormal basis vectors
  gsl_matrix *h     = gsl_matrix_calloc(inner+1,   inner  ); //Hessenberg form of matrix
  gsl_matrix *hth   = gsl_matrix_calloc(inner,     inner  ); //h'*h in matlab
  
  gsl_vector *f     = gsl_vector_calloc(output_dim);
  gsl_vector *b     = gsl_vector_calloc(input_dim);
  gsl_vector *b2    = gsl_vector_calloc(inner+1);    //The right hand side of GMRES. It's a trivial vector with one non-zero element.
  gsl_vector *htb2  = gsl_vector_calloc(inner);      //The right hand side of GMRES. It's a trivial vector with one non-zero element.
  gsl_vector *y     = gsl_vector_calloc(inner);      //The right hand side of GMRES. It's a trivial vector with one non-zero element.
  gsl_vector *jq    = gsl_vector_calloc(output_dim); //Place to store action of jacobian
  gsl_vector *jtjq  = gsl_vector_calloc(input_dim);  //Place to store action of j'*j for over-constrained Newton
  gsl_vector *step  = gsl_vector_calloc(input_dim);  //Step direction produced by Newton
  gsl_vector *q_col = gsl_vector_calloc(input_dim);  //a column of q
  gsl_vector *q_col2= gsl_vector_calloc(input_dim);  //a column of q  

  for(int i=0; i<max_iterations; i++){
    int outer_iterations = 0; //Reset each Newton step
    
    //Step 1: check |F| or |b| = |J'*F| to monitor convergence
    if(overconstrained){
      EVAL   (obj,  f,state,params);
      EVAL_Jt(obj,b,f,state,params);
      normf = gsl_blas_dnrm2(f);
      normb = gsl_blas_dnrm2(b);
      printf("Iteration %d: |F| = %.9e, |J'*F| = %.9e\n", i, normf, normb);
      if( normb < threshold ){
        //Only look for exit condition from normb, which should be identically zero at a local minima
        printf("|J'*F| is less than specified threshold. Exiting Newton...\n");
	return;
      }
    }
    else{
      //Not overconstrained
      EVAL(obj,f,state,params);
      normf = gsl_blas_dnrm2(f);
      printf("Iteration %d: |F| = %.9e\n", i, normf);
      if( normf < threshold ){
        printf("|F| is less than specified threshold. Exiting Newton...\n");
        return;
      }
    }

    //Simplifies code to introduce pointer "work"
    gsl_vector *work;
    work = overconstrained ? jtjq : jq;

//Repeated iterations of GMRES start here with an updated step
restart:
    if(overconstrained){
      //In these lines, store "b - J'*J(step)" in the first column of q
      EVAL_J (obj,jq,  step,state,params); // jq <- J(step)
      EVAL_Jt(obj,work,  jq,state,params); // work <- J(step)
      gsl_vector_sub(work, b);             // work <- J'*J(step) - b
    }
    else{
      //In these lines, store "f - J(step)" in the first column of q
      EVAL_J(obj,jq,step,state,params); // jq <- J(step)
      gsl_vector_sub(work, f);          // jq <- J(step) - f
    }
 
    //Compute norm
    h_element = gsl_blas_dnrm2(work);
    gsl_vector_scale(work,-1/h_element); // jq <- (f - J(step))/norm      gives a unit vector!
    gsl_vector_set(b2, 0, h_element ); //This is the only non-zero element of b2
 
    //Set the first column of q to this unit vector
    gsl_matrix_set_col(q, 0, work);
    for(int j=0; j<inner; j++){
      gsl_matrix_get_col(q_col, q, j);
      if(overconstrained){
        EVAL_J (obj,   jq, q_col, state, params);
        EVAL_Jt(obj, work,    jq, state, params);
      }
      else{
        EVAL_J(obj, work, q_col, state, params);
      }
      for(int k=0; k<=j; k++){
        gsl_matrix_get_col(q_col2, q, k);
        gsl_blas_ddot( work, q_col2, &h_element );
        gsl_matrix_set(h, k, j, h_element);
        gsl_blas_daxpy(-h_element, q_col2, work);
      }
      h_element = gsl_blas_dnrm2(work);
      gsl_matrix_set(h, j+1, j, h_element);
      gsl_vector_scale( work, 1/h_element );
      gsl_matrix_set_col( q, j+1, work );
    }

    print_vector(b2, "b2");
    print_matrix(h, "h");
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

    print_vector(y, "y");

    //Use y to update step
    q->size2--;//Hide last column of q
    gsl_blas_dgemv( CblasNoTrans, 1, q, y, 1, step ); //update step
    q->size2++;//Restore last column of q


    //Check residual
    if(overconstrained){
      EVAL_J (obj,jq,  step,state,params); // jq   <- J(step)
      EVAL_Jt(obj,jtjq,  jq,state,params); // jtjq <- J'*J(step)
      gsl_vector_sub(jtjq, b);             // jtjq <- J'*J(step) - b
      residual = gsl_blas_dnrm2(jtjq)/normb;
    }
    else{
      EVAL_J(obj,jq,step,state,params); // jq <- J(step)
      gsl_vector_sub(jq, f);            // jq <- J(step) - f
      residual = gsl_blas_dnrm2(jq)/normf;
    }

    outer_iterations++; //congratulations you have completed an outer iteration
    if( residual > 1e-9 & outer_iterations < outer ){
      goto restart;
    }

    //Only display residual at end of iterations
    printf("GMRES residual = %e\n", residual);
    
    //Newton step
    //feel free to damp and use like -0.1
    gsl_blas_daxpy(-DAMP, step, state);
  }

  gsl_matrix_free(q);
  gsl_matrix_free(h);
  gsl_matrix_free(hth);
  gsl_vector_free(f);
  gsl_vector_free(y);
  gsl_vector_free(b);
  gsl_vector_free(b2);
  gsl_vector_free(htb2);
  gsl_vector_free(jq);
  gsl_vector_free(jtjq);
  gsl_vector_free(step);
  gsl_vector_free(q_col);
  gsl_vector_free(q_col2);

  stop = clock();
  printf("Newton-GMRES took %f seconds. Have a nice day!\n", (double)(stop - start)/CLOCKS_PER_SEC );
}

void print_norm( gsl_vector *v, char name[] ){
  double norm = gsl_blas_dnrm2(v);
  printf("|%s| = %.9e\n", name, norm);
}

void initial_data( gsl_vector *state ){
  gsl_vector u = *state; u.size = N*N; u.data = &u.data[0*N*N];
  gsl_vector v = *state; v.size = N*N; v.data = &v.data[1*N*N];
  gsl_vector w = *state; w.size = N*N; w.data = &w.data[2*N*N];

  double dx = 2*M_PI/N;

  #pragma omp parallel
  SPACE_LOOP{
    double x = dx*i;
    double y = dx*j;
    
    gsl_vector_set(&u, IDX(i,j), cos(x+y + 0.4) );
    gsl_vector_set(&v, IDX(i,j), sin(3*y - 0.1) );
    gsl_vector_set(&w, IDX(i,j), cos(x)*cos(y) );

    //gsl_vector_set(&u, IDX(i,j), cos(2*y - 1) );
    //gsl_vector_set(&v, IDX(i,j), cos(x + 0.1) + sin(2*x)*sin(y) );
    //gsl_vector_set(&w, IDX(i,j), cos(3*y) + sin(2*x-0.3) );
  }
}

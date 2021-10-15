#ifndef MY_OBJECTIVE_FUNCTION
#define MY_OBJECTIVE_FUNCTION


//GSL stuff
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

//For some reason it is important that common.h comes AFTER GSL headers
#include "common.h"
#include "IO.h"

//CUDA stuff
#include <cuda_runtime.h>
#include <cublas_v2.h>



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


void cuda_newton_gmres( objective_function *obj, double *state, int *max_iterations, double threshold, void *params, int inner, int outer, double *, double *, double * );

#endif

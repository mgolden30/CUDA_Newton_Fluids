#include "objective_function.h"
#include "fourier_filter.h"

void cuda_newton_gmres( objective_function *obj, double* state_d, int *max_iterations, double threshold, void *params, int inner, int outer, double *gmres_residual_series, double *normF_series, double *normJtF_series ){
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

  for(int i=0; i< *max_iterations; i++){
    int outer_iterations = 0; //Reset each Newton step
    
    //Step 1: check |F| or |b| = |J'*F| to monitor convergence
    if(overconstrained){
      EVAL   (obj,   F_d,      state_d, params);
      EVAL_Jt(obj, JtF_d, F_d, state_d, params);
      cublasDnrm2( handle, output_dim,   F_d, 1, &normF   );
      cublasDnrm2( handle,  input_dim, JtF_d, 1, &normJtF );
      cudaDeviceSynchronize();

      printf( "Iteration %d: |F| = %.9e, |J'*F| = %.9e\n", i, normF, normJtF );
      normF_series[i]   = normF;
      normJtF_series[i] = normJtF;

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
	(*max_iterations) = i+1;
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

    gmres_residual_series[i] = residual;

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

    fourier_filter_state( state_d );
    
    //check for exiting
    BREAK_IF_Q 
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

#ifndef MY_COMMON
#define MY_COMMON 

/* This header contains definitions that are useful for all source code.
 * Assume this is included in any .cu file
 */

#define N 32 //Number of points per side

//Forcing express in terms of i and j: (x,y) = (i,j)*dx
#define FORCING (4*cos(4*j*dx))
#define NU      (1/60)

#define U0   ( cos(x+2*y + 0.4) )
#define V0   ( sin(2*y) )
#define W0   ( cos(3*x+0.7)*cos(y-0.5) + sin(x) )



//Useful Macros
#define IDX(i,j)      (i)*N+(j)                        //Indexing compatible with Matlab
#define TRUE_MOD(m,n) (((m)%(n))+(n))%(n)              //A modular operation that ensures you get a nonnegative result
#define IDXP(i,j)     IDX(TRUE_MOD(i,N),TRUE_MOD(j,N)) //Periodic indexing
#define SPACE_LOOP    for(int i=0; i<N; i++) for(int j=0; j<N; j++)




//GMRES PARAMETERS
#define DAMP 0.5
#define INNER 250
#define OUTER 2
#define MAX_ITERATIONS 100

//Fourier filtering parameters
#define FOURIER_CUTOFF N*N/9 //N*N/9 is the usual 2/3 dealiasing


//CUDA parameters
#define BLOCKS  16
#define THREADS 64
#define TOTAL_THREADS   ( BLOCKS*THREADS )




#define DEBUG 0



#endif


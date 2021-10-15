#include "IO.h"

/* PURPOSE:
 * Here are functions for outputing data to the screen and reading/writing to files.
 */

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



void print_norm( gsl_vector *v, char name[] ){
  double norm = gsl_blas_dnrm2(v);
  printf("|%s| = %.9e\n", name, norm);
}

#ifndef MY_INPUT_OUTPUT
#define MY_INPUT_OUTPUT


//ncurses for unbuffered user input. Apparently this is not supported by ANSI C.
#include <curses.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

#include "common.h"

/* PURPOSE:
 * Here are functions for outputing data to the screen and reading/writing to files.
 */

void print_matrix( gsl_matrix *m, char name[] );
void print_vector( gsl_vector *v, char name[] );
void print_norm( gsl_vector *v, char name[] );

//Macro that breaks from a loop if the letter q is pressed. Done with ncurses
#define BREAK_IF_Q         WINDOW* my_window; char user_input;  my_window = initscr(); cbreak(); timeout(10); user_input = getch(); endwin(); /*printf("Read %c from user\n", user_input);*/ if( user_input == 'q' ){ printf("\nUser input to terminate...\n"); *max_iterations = i+1; break; }


#endif

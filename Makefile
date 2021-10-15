CC=nvcc
CFLAGS=-I./include -lcublas -lcufft -lgsl -lgslcblas -lm -O3 -DHAVE_INLINE -ltinfo -lncurses

all: euler_eq navier_stokes_eq

#################
# Object files
#################
obj/euler_eq.o: src/euler_eq.cu
	$(CC) $^ $(CFLAGS) -c -o $@
obj/navier_stokes_eq.o: src/navier_stokes_eq.cu
	$(CC) $^ $(CFLAGS) -c -o $@
obj/IO.o: src/IO.cu
	$(CC) $^ $(CFLAGS) -c -o $@
obj/objective_function.o: src/objective_function.cu
	$(CC) $^ $(CFLAGS) -c -o $@
obj/fourier_filter.o: src/fourier_filter.cu
	$(CC) $^ $(CFLAGS) -c -o $@




################
# Executable
################
euler_eq: obj/euler_eq.o obj/objective_function.o obj/IO.o obj/fourier_filter.o
	$(CC) $^ $(CFLAGS) -o $@

navier_stokes_eq: obj/navier_stokes_eq.o obj/objective_function.o obj/IO.o obj/fourier_filter.o
	$(CC) $^ $(CFLAGS) -o $@




clean:
	rm obj/*

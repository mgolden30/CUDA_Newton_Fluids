CC=nvcc
CFLAGS=-lcublas -lgsl -lgslcblas -lm -O3 -DHAVE_INLINE -ltinfo -lncurses

all: equilibrium_finder main

equilibrium_finder: src/equilibrium_finder.cu
	$(CC) $^ $(CFLAGS) -o $@

main: src/main.cu
	$(CC) $^ $(CFLAGS) -o $@

clean:
	rm main

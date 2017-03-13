DEBUG ?= 0
ifeq ($(DEBUG), 1)
	override CFLAGS+= -lm -Wall -Og -g3 -O0 -shared-libgcc
else
	override CFLAGS+= -march=native -O3 -lm
endif

CC= gcc-5
OPENMP= -fopenmp
DEPS= edge_detector/edge_detectors.h performance/performance.h utils/matrix_ops.h
OBJ= edge_detectors.o performance.o matrix_ops.o main.o

all: main clean

main.o: main.c $(DEPS)
	$(CC) -c $< $(CFLAGS) $(OPENMP)

matrix_ops.o: utils/matrix_ops.c $(DEPS)
	$(CC) -c $< $(CFLAGS)

performance.o: performance/performance.c $(DEPS)
	$(CC) -c $< $(CFLAGS) $(OPENMP)

edge_detectors.o: edge_detector/edge_detectors.c $(DEPS)
	$(CC) -c $< $(CFLAGS) $(OPENMP)

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(OPENMP)

clean:
	rm -f *.o

remove:
	rm -f *.o main

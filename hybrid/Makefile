DEBUG ?= 0
ifeq ($(DEBUG), 1)
	override CFLAGS+= -arch sm_30 -g -G -lstdc++ -Xcompiler -fopenmp -Xcompiler -lm -Xcompiler -Og -Xcompiler -Wall
else
	override CFLAGS+= -arch sm_30 -O2 -Xcompiler -fopenmp -Xcompiler -lm -Xcompiler -march=native
endif

CC= nvcc -ccbin=g++-5
DEPS= edge_detector/edge_detectors.cuh performance/performance.cuh utils/matrix_ops.cuh
OBJ= matrix_ops.o edge_detectors.o performance.o main.o

all: main clean

main.o: main.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

matrix_ops.o: utils/matrix_ops.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

performance.o: performance/performance.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

edge_detectors.o: edge_detector/edge_detectors.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o

remove:
	rm -f *.o main

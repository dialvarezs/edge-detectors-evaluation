DEBUG ?= 0
ifeq ($(DEBUG), 1)
    override CFLAGS+= -arch sm_30 -g -G -lstdc++ -Xcompiler -Og -Xcompiler -Wall
else
	override CFLAGS+= -arch sm_30 -O2 -Xcompiler -march=native
endif

CC= nvcc -ccbin=gcc-5
DEPS= edge_detector/edge_detectors_gpu.cuh performance/performance_gpu.cuh utils/matrix_ops.cuh
OBJ= edge_detectors_gpu.o performance_gpu.o matrix_ops.o main.o

all: main clean

main.o: main.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

matrix_ops.o: utils/matrix_ops.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

performance_gpu.o: performance/performance_gpu.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

edge_detectors_gpu.o: edge_detector/edge_detectors_gpu.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o

remove:
	rm -f *.o main

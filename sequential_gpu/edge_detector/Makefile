CC= nvcc -ccbin=gcc-5
CFLAGS= -lm -O2 -arch sm_30
DEPS= ../utils/matrix_ops.h edge_detectors_gpu.h
OBJ= edge_detectors_gpu.o matrix_ops.o test.o

all: edge_detector clean

%.o: %.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

matrix_ops.o: ../utils/matrix_ops.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

edge_detector: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o

remove:
	rm -f *.o edge_detector

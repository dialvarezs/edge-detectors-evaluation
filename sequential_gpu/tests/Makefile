CC= nvcc -ccbin=gcc-5
CFLAGS= -lm -O2 -arch sm_30
DEPS= ../utils/matrix_ops.h
OBJ= matrix_ops.o


all: noise_maker edge_detector performance clean

%.o: %.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

matrix_ops.o: ../utils/matrix_ops.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

noise_maker: test_noise_maker.o $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

edge_detectors_gpu.o: ../edge_detector/edge_detectors_gpu.cu ../edge_detector/edge_detectors_gpu.h $(DEPS)
	$(CC) -c $< $(CFLAGS)

edge_detector: test_edge_detector.o edge_detectors_gpu.o $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

performance_gpu.o: ../performance/performance_gpu.cu ../performance/performance_gpu.h $(DEPS)
	$(CC) -c $< $(CFLAGS)

performance: test_performance.o performance_gpu.o $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o

remove:
	rm -f *.o edge_detector noise_maker performance

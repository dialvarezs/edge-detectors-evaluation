CC= nvcc -ccbin=gcc-5
CFLAGS= -lm -O2 -arch sm_30
DEPS= ../utils/matrix_ops.h performance_gpu.h
OBJ= performance_gpu.o matrix_ops.o test.o

all: performance clean

%.o: %.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

matrix_ops.o: ../utils/matrix_ops.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

performance: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o

remove:
	rm -f *.o performance

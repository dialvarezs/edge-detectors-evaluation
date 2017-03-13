CC= nvcc -ccbin=gcc-5
CFLAGS= -lm -O2 -arch sm_30
DEPS= ../utils/matrix_ops.h
OBJ= matrix_ops.o test.o

all: noise_maker clean

%.o: %.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

matrix_ops.o: ../utils/matrix_ops.cu $(DEPS)
	$(CC) -c $< $(CFLAGS)

noise_maker: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o

remove:
	rm -f *.o noise_maker

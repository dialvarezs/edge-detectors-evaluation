DEBUG ?= 0
ifeq ($(DEBUG), 1)
    override CFLAGS+= -lm -Wall -g3 -shared-libgcc
else
	override CFLAGS+= -lm -O2
endif

CC= gcc-5
DEPS= ../utils/matrix_ops.h
OBJ= matrix_ops.o


all: noise_maker edge_detector performance clean

%.o: %.c $(DEPS)
	$(CC) -c $< $(CFLAGS)

matrix_ops.o: ../utils/matrix_ops.c $(DEPS)
	$(CC) -c $< $(CFLAGS)

noise_maker: test_noise_maker.o $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

edge_detectors.o: ../edge_detector/edge_detectors.c ../edge_detector/edge_detectors.h $(DEPS)
	$(CC) -c $< $(CFLAGS)

edge_detector: test_edge_detector.o edge_detectors.o $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

performance.o: ../performance/performance.c ../performance/performance.h $(DEPS)
	$(CC) -c $< $(CFLAGS)

performance: test_performance.o performance.o $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o

remove:
	rm -f *.o edge_detector noise_maker performance

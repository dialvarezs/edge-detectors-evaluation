DEBUG ?= 0
ifeq ($(DEBUG), 1)
	override CFLAGS+= -lm -Wall -Og -g3 -O0 -shared-libgcc
else
	override CFLAGS+= -march=native -O2 -lm
endif

CC= gcc
DEPS= edge_detector/edge_detectors.h performance/performance.h utils/matrix_ops.h
OBJ= edge_detectors.o performance.o matrix_ops.o

all: main clean

main.o: main.c $(DEPS)
	$(CC) -c $< $(CFLAGS)

matrix_ops.o: utils/matrix_ops.c $(DEPS)
	$(CC) -c $< $(CFLAGS)

performance.o: performance/performance.c $(DEPS)
	$(CC) -c $< $(CFLAGS)

edge_detectors.o: edge_detector/edge_detectors.c $(DEPS)
	$(CC) -c $< $(CFLAGS)

main: $(OBJ) main.o
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o

remove:
	rm -f *.o main
